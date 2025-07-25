import streamlit as st
import pandas as pd
from openai import OpenAI
import json
from typing import List, Dict, Any, Optional, Union
import time
from datetime import datetime
import uuid
import re

# Initialize OpenAI client
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {str(e)}")
    st.stop()

# Define CSV file structures with exact column names
CSV_STRUCTURES = {
    "sample_mortgage_accounts.csv": [
        "customer_id", "product_type", "account_status", "loan_open_date", "loan_balance"
    ],
    "sample_loan_repayments.csv": [
        "repayment_id", "customer_id", "loan_account_number", "repayment_date",
        "repayment_amount", "installment_number", "payment_method_status", "loan_type",
        "interest_component", "principal_component", "remaining_balance"
    ],
    "sample_telco_billing.csv": [
        "billing_id", "customer_id", "bill_date", "bill_amount", "plan_type",
        "data_used_gb", "voice_minutes", "sms_count", "channel"
    ],
    "sample_product_enrollments.csv": [
        "enrollment_id", "customer_id", "product_type", "product_name", "enrollment_date", "status"
    ],
    "sample_customer_profiles.csv": [
        "customer_id", "name", "email", "phone", "dob", "gender",
        "region", "segment", "household_id", "is_primary"
    ],
    "sample_savings_account_transactions.csv": [
        "transaction_id", "account_id", "customer_id", "amount", "date", "transaction_type"
    ],
    "sample_credit_card_transactions.csv": [
        "customer_id", "card_number", "transaction_date", "transaction_amount", "transaction_type"
    ]
}

# Create mapping for case-insensitive matching
COLUMN_MAPPING = {}
for file, columns in CSV_STRUCTURES.items():
    for col in columns:
        COLUMN_MAPPING[col.lower()] = col

def clean_user_input(text: str) -> str:
    """Clean user input by removing extra spaces between characters"""
    text = ' '.join(text.split())
    return text.replace(" o n ", " on ").replace(" A N D ", " AND ").replace(" O R ", " OR ")

def extract_logical_structure(user_input: str) -> str:
    """Extract and clarify the logical structure from user input"""
    prompt = f"""
    The user provided this requirement: "{user_input}"
    
    Analyze the logical structure and rewrite it in a clear format with proper parentheses showing the intended grouping.
    Use only AND/OR operators and parentheses. Don't add any explanations.
    
    Example:
    Input: "A and B or C"
    Output: "(A AND B) OR C"
    
    Input: "A or B and C"
    Output: "A OR (B AND C)"
    
    Now process this input:
    Input: "{user_input}"
    Output: """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a logical expression analyzer. Your task is to rewrite expressions with proper parentheses."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error analyzing logical structure: {str(e)}")
        return user_input

def generate_prompt_guidance(logical_structure: str, user_input: str) -> str:
    """Generate guidance for the AI with proper logical grouping"""
    available_data = "\n".join([f"- {f}: {', '.join(cols)}" for f, cols in CSV_STRUCTURES.items()])
    
    base_prompt = f"""
    You are a financial rule generation assistant. Create rules based on this logical structure:
    {logical_structure}
    
    Original user requirement: "{user_input}"
    
    CRITICAL INSTRUCTIONS:
    1. Use ONLY the exact column names from these data sources:
    {available_data}
    2. Follow the logical structure EXACTLY as provided
    3. For time references like "last month", use "Rolling 30 days"
    4. For amount aggregations, use "sum" function
    
    Output JSON matching this schema:
    {{
        "rules": [
            {{
                "id": "generated_id",
                "ruleType": "condition" or "conditionGroup",
                // For conditions:
                "dataSource": "source_name",
                "field": "column_name",
                "eligibilityPeriod": "time_period or N/A",
                "function": "aggregation_function or N/A",
                "operator": "comparison_operator",
                "value": "comparison_value",
                // For condition groups:
                "conditions": [ array of conditions/groups ],
                "groupConnector": "AND" or "OR"
            }}
        ],
        "logical_structure": "{logical_structure}"
    }}

    Respond ONLY with the JSON output.
    """
    return base_prompt

def validate_and_correct_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and correct the rule structure"""
    if not rule or "rules" not in rule:
        return {"rules": [], "logical_structure": ""}
    
    def process_items(items):
        for item in items:
            if "id" not in item:
                item["id"] = str(uuid.uuid4())
            if item.get("ruleType") == "conditionGroup":
                if "conditions" not in item:
                    item["conditions"] = []
                if "groupConnector" not in item:
                    item["groupConnector"] = "AND"
                process_items(item["conditions"])
    
    process_items(rule["rules"])
    return rule

def generate_rule_with_openai(logical_structure: str, user_input: str) -> Dict[str, Any]:
    """Generate rule with proper logical grouping"""
    prompt = generate_prompt_guidance(logical_structure, user_input)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You create precise JSON rules with proper logical grouping."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        json_str = response_content[response_content.find('{'):response_content.rfind('}')+1]
        rule = json.loads(json_str)
        return validate_and_correct_rule(rule)
    except Exception as e:
        st.error(f"Error generating rule: {str(e)}")
        return None

def render_condition(rule_item: Dict[str, Any], key_prefix: str = "") -> None:
    """Render a single condition in the UI"""
    cols = st.columns(7)
    with cols[0]:
        selected_ds = st.selectbox(
            "Data Source",
            options=list(CSV_STRUCTURES.keys()),
            index=list(CSV_STRUCTURES.keys()).index(rule_item["dataSource"]) 
            if rule_item["dataSource"] in CSV_STRUCTURES else 0,
            key=f"{key_prefix}_ds"
        )
    with cols[1]:
        columns = CSV_STRUCTURES.get(selected_ds, [])
        selected_field = st.selectbox(
            "Field", 
            options=columns,
            index=columns.index(rule_item["field"]) 
            if rule_item["field"] in columns else 0,
            key=f"{key_prefix}_field"
        )
    with cols[2]:
        st.selectbox("eligibilityPeriod", 
                    ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days"],
                    index=0 if rule_item.get("eligibilityPeriod") == "N/A" else 1,
                    key=f"{key_prefix}_period")
    with cols[3]:
        st.selectbox("function", 
                    ["N/A", "sum", "count", "avg", "max", "min"],
                    index=0 if rule_item.get("function") == "N/A" else 1,
                    key=f"{key_prefix}_func")
    with cols[4]:
        operator_options = ["=", ">", "<", ">=", "<=", "!=", "contains"]
        operator_index = operator_options.index(rule_item["operator"]) if rule_item["operator"] in operator_options else 0
        st.selectbox("Operator", 
                    operator_options,
                    index=operator_index,
                    key=f"{key_prefix}_op")
    with cols[5]:
        st.text_input("Value", value=str(rule_item.get("value", "")), 
                    key=f"{key_prefix}_val")
    with cols[6]:
        if rule_item.get("connector"):
            st.selectbox("Connector", 
                        ["AND", "OR"],
                        index=0 if rule_item["connector"] == "AND" else 1,
                        key=f"{key_prefix}_conn")

def render_condition_group(group: Dict[str, Any], group_index: int) -> None:
    """Render a condition group in the UI"""
    with st.expander(f"Group {group_index + 1} ({group['groupConnector']})", expanded=True):
        for i, condition in enumerate(group.get("conditions", [])):
            if condition.get("ruleType") == "condition":
                render_condition(condition, key_prefix=f"group_{group['id']}_cond_{i}")
            elif condition.get("ruleType") == "conditionGroup":
                render_condition_group(condition, i)

def display_rule_ui(rule: Dict[str, Any]) -> None:
    """Display the rule in the UI"""
    if not rule or "rules" not in rule:
        return
    
    st.subheader("Rule Structure")
    st.code(rule.get("logical_structure", ""), language="text")
    
    st.subheader("Conditions")
    for i, rule_item in enumerate(rule["rules"]):
        if rule_item.get("ruleType") == "condition":
            with st.expander(f"Condition {i+1}", expanded=True):
                render_condition(rule_item, key_prefix=f"cond_{i}")
        elif rule_item.get("ruleType") == "conditionGroup":
            render_condition_group(rule_item, i)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can help create mortgage holder rules. What criteria would you like to use?"}
        ]
    if "current_rule" not in st.session_state:
        st.session_state.current_rule = None
    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = ""
    if "awaiting_structure_confirmation" not in st.session_state:
        st.session_state.awaiting_structure_confirmation = False
    if "awaiting_rule_confirmation" not in st.session_state:
        st.session_state.awaiting_rule_confirmation = False
    if "proposed_structure" not in st.session_state:
        st.session_state.proposed_structure = ""
    if "confirmed_rule" not in st.session_state:
        st.session_state.confirmed_rule = False

def display_chat_message(role: str, content: str):
    """Display a chat message"""
    with st.chat_message(role):
        st.markdown(content)

def handle_user_confirmation(confirmation: bool, confirmation_type: str):
    """Handle user confirmation"""
    if confirmation_type == "structure":
        if confirmation:
            st.session_state.awaiting_structure_confirmation = False
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Great! Based on your confirmation, I'll generate rules for: {st.session_state.proposed_structure}"
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Please rephrase your requirements with clearer logical grouping."
            })
            st.session_state.awaiting_structure_confirmation = False
    elif confirmation_type == "rule":
        if confirmation:
            st.session_state.confirmed_rule = True
            st.session_state.awaiting_rule_confirmation = False
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Rule confirmed! Here's your final rule:"
            })
        else:
            st.session_state.awaiting_rule_confirmation = False
            st.session_state.messages.append({
                "role": "assistant",
                "content": "What changes would you like to make to the rule?"
            })

def generate_new_rule():
    """Generate a new rule based on current state"""
    if not st.session_state.user_prompt:
        return
    
    # Step 1: Extract and confirm logical structure
    if not st.session_state.proposed_structure:
        with st.spinner("Analyzing logical structure..."):
            st.session_state.proposed_structure = extract_logical_structure(st.session_state.user_prompt)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"To confirm, is this the logical structure you intended?\n\n```\n{st.session_state.proposed_structure}\n```\n\nPlease respond with 'yes' or 'no'."
            })
            st.session_state.awaiting_structure_confirmation = True
        return
    
    # Step 2: Generate rule after structure confirmation
    if not st.session_state.awaiting_rule_confirmation and not st.session_state.current_rule:
        with st.spinner("Generating rule with proper grouping..."):
            rule = generate_rule_with_openai(
                st.session_state.proposed_structure,
                st.session_state.user_prompt
            )
            if rule:
                st.session_state.current_rule = rule
                rule_preview = json.dumps(rule, indent=2)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I've generated this rule:\n\n```json\n{rule_preview}\n```\n\nDoes this meet your requirements? (yes/no)"
                })
                st.session_state.awaiting_rule_confirmation = True
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn't generate a valid rule. Please provide more details."
                })

def main():
    st.set_page_config(page_title="Mortgage Rule Generator", layout="wide")
    st.title("🏦 Interactive Mortgage Rule Generator")
    
    # Custom CSS
    st.markdown("""
    <style>
        .stChatFloatingInputContainer { bottom: 20px; }
        .stChatMessage { padding: 12px; border-radius: 8px; margin-bottom: 12px; }
        .assistant-message { background-color: #f0f2f6; }
        .user-message { background-color: #e3f2fd; }
        .stExpander { margin-bottom: 15px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.current_rule:
            display_rule_ui(st.session_state.current_rule)
            
            if st.session_state.confirmed_rule:
                st.success("✅ Final Rule Confirmed")
                st.json(st.session_state.current_rule)
                
                json_str = json.dumps(st.session_state.current_rule, indent=2)
                st.download_button(
                    label="Download Rule JSON",
                    data=json_str,
                    file_name=f"mortgage_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                if st.button("Create New Rule"):
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Let's create a new rule. What criteria would you like to use?"}
                    ]
                    st.session_state.current_rule = None
                    st.session_state.user_prompt = ""
                    st.session_state.proposed_structure = ""
                    st.session_state.confirmed_rule = False
                    st.rerun()
    
    with col2:
        st.subheader("Conversation")
        
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        if prompt := st.chat_input("Type your message here..."):
            cleaned_prompt = clean_user_input(prompt)
            st.session_state.messages.append({"role": "user", "content": cleaned_prompt})
            
            if st.session_state.awaiting_structure_confirmation:
                if "yes" in cleaned_prompt.lower():
                    handle_user_confirmation(True, "structure")
                else:
                    handle_user_confirmation(False, "structure")
                    st.session_state.proposed_structure = ""
                st.rerun()
            
            elif st.session_state.awaiting_rule_confirmation:
                if "yes" in cleaned_prompt.lower():
                    handle_user_confirmation(True, "rule")
                else:
                    handle_user_confirmation(False, "rule")
                    st.session_state.current_rule = None
                st.rerun()
            
            else:
                st.session_state.user_prompt = cleaned_prompt
                st.session_state.current_rule = None
                st.session_state.proposed_structure = ""
                st.session_state.confirmed_rule = False
                generate_new_rule()
                st.rerun()
        
        generate_new_rule()

if __name__ == "__main__":
    main()
