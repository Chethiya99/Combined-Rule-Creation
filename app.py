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

def generate_alternative_structures(user_input: str) -> List[str]:
    """Generate multiple possible logical structures from user input"""
    prompt = f"""
    The user provided this requirement: "{user_input}"
    
    Generate 3 different possible logical interpretations with proper parentheses showing different grouping possibilities.
    Use only AND/OR operators and parentheses. Number each option.
    
    Example:
    Input: "A and B or C"
    Output:
    1. (A AND B) OR C
    2. A AND (B OR C)
    3. A AND B OR C
    
    Now process this input:
    Input: "{user_input}"
    Output:
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You generate multiple possible logical groupings for expressions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=200
        )
        output = response.choices[0].message.content.strip()
        # Parse the numbered options
        options = [line.split('. ', 1)[1] for line in output.split('\n') if '. ' in line]
        return options[:3]  # Return up to 3 options
    except Exception as e:
        st.error(f"Error generating alternatives: {str(e)}")
        return [user_input]  # Fallback to original input

def generate_prompt_guidance(logical_structure: str, user_input: str) -> str:
    """Generate guidance for the AI with proper logical grouping"""
    available_data = "\n".join([f"- {f}: {', '.join(cols)}" for f, cols in CSV_STRUCTURES.items()])
    
    base_prompt = f"""
    You are a financial rule generation assistant. Create rules based on this confirmed logical structure:
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
    if "proposed_structures" not in st.session_state:
        st.session_state.proposed_structures = []
    if "current_structure_index" not in st.session_state:
        st.session_state.current_structure_index = 0
    if "confirmed_structure" not in st.session_state:
        st.session_state.confirmed_structure = ""
    if "confirmed_rule" not in st.session_state:
        st.session_state.confirmed_rule = False
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Simple Mode"

def display_chat_message(role: str, content: str):
    """Display a chat message"""
    with st.chat_message(role):
        st.markdown(content)

def handle_structure_confirmation(confirmation: bool):
    """Handle user confirmation of logical structure"""
    if confirmation:
        st.session_state.confirmed_structure = st.session_state.proposed_structures[st.session_state.current_structure_index]
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Great! I'll generate rules for this structure:\n\n```\n{st.session_state.confirmed_structure}\n```"
        })
        st.session_state.awaiting_structure_confirmation = False
    else:
        # Move to next alternative structure
        st.session_state.current_structure_index += 1
        if st.session_state.current_structure_index < len(st.session_state.proposed_structures):
            show_next_alternative()
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I've shown all possible interpretations. Please rephrase your requirements with clearer logical grouping."
            })
            st.session_state.awaiting_structure_confirmation = False
            reset_structure_state()

def handle_rule_confirmation(confirmation: bool):
    """Handle user confirmation of generated rule"""
    if confirmation:
        st.session_state.confirmed_rule = True
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Rule confirmed! Here's your final rule:"
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "What changes would you like to make to the rule?"
        })
        st.session_state.current_rule = None
    st.session_state.awaiting_rule_confirmation = False

def show_next_alternative():
    """Show the next alternative structure to the user"""
    structure = st.session_state.proposed_structures[st.session_state.current_structure_index]
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"Does this structure match your intention?\n\n```\n{structure}\n```\n\nPlease respond with 'yes' or 'no'."
    })
    st.session_state.awaiting_structure_confirmation = True

def reset_structure_state():
    """Reset structure-related session state"""
    st.session_state.proposed_structures = []
    st.session_state.current_structure_index = 0
    st.session_state.confirmed_structure = ""

def generate_new_rule():
    """Generate a new rule based on current state"""
    if not st.session_state.user_prompt:
        return
    
    # Guided Mode: Generate and confirm logical structure
    if st.session_state.active_tab == "Guided Mode":
        if not st.session_state.proposed_structures and not st.session_state.confirmed_structure:
            with st.spinner("Analyzing possible logical structures..."):
                st.session_state.proposed_structures = generate_alternative_structures(st.session_state.user_prompt)
                if st.session_state.proposed_structures:
                    show_next_alternative()
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I couldn't interpret the logical structure. Please provide more specific requirements."
                    })
            return
    
    # Generate rule after structure confirmation (Guided) or directly (Simple)
    if (st.session_state.active_tab == "Simple Mode" or 
        (st.session_state.confirmed_structure and st.session_state.active_tab == "Guided Mode")) and \
        not st.session_state.current_rule and not st.session_state.awaiting_rule_confirmation:
        
        with st.spinner("Generating rule..."):
            # For Simple Mode, generate structure directly
            if st.session_state.active_tab == "Simple Mode":
                st.session_state.confirmed_structure = st.session_state.user_prompt
            
            rule = generate_rule_with_openai(
                st.session_state.confirmed_structure,
                st.session_state.user_prompt
            )
            
            if rule:
                st.session_state.current_rule = rule
                rule_preview = json.dumps(rule, indent=2)
                
                if st.session_state.active_tab == "Guided Mode":
                    msg = f"I've generated this rule based on your confirmed structure:\n\n```json\n{rule_preview}\n```\n\nDoes this meet your requirements? (yes/no)"
                else:
                    msg = f"I've generated this rule directly from your input:\n\n```json\n{rule_preview}\n```\n\nDoes this meet your requirements? (yes/no)"
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg
                })
                st.session_state.awaiting_rule_confirmation = True
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn't generate a valid rule. Please provide more details."
                })

def render_chat_interface():
    """Render the chat interface based on active tab"""
    st.subheader("Conversation")
    
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    if prompt := st.chat_input("Type your message here..."):
        cleaned_prompt = clean_user_input(prompt)
        st.session_state.messages.append({"role": "user", "content": cleaned_prompt})
        
        if st.session_state.awaiting_structure_confirmation:
            if "yes" in cleaned_prompt.lower():
                handle_structure_confirmation(True)
            elif "no" in cleaned_prompt.lower():
                handle_structure_confirmation(False)
            st.rerun()
        
        elif st.session_state.awaiting_rule_confirmation:
            if "yes" in cleaned_prompt.lower():
                handle_rule_confirmation(True)
            elif "no" in cleaned_prompt.lower():
                handle_rule_confirmation(False)
            st.rerun()
        
        else:
            st.session_state.user_prompt = cleaned_prompt
            st.session_state.current_rule = None
            reset_structure_state()
            st.session_state.confirmed_rule = False
            generate_new_rule()
            st.rerun()
    
    generate_new_rule()

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
        .structure-option { 
            background-color: #f5f5f5; 
            padding: 10px; 
            margin: 5px 0; 
            border-radius: 5px; 
            border-left: 4px solid #4285f4;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 20px;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #f0f2f6;
            font-weight: 600;
        }
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
                    reset_structure_state()
                    st.session_state.confirmed_rule = False
                    st.rerun()
    
    with col2:
        # Tab selection
        tab1, tab2 = st.tabs(["Simple Mode", "Guided Mode"])
        
        with tab1:
            st.session_state.active_tab = "Simple Mode"
            st.info("💡 Use Simple Mode for straightforward rules without structural suggestions")
            render_chat_interface()
        
        with tab2:
            st.session_state.active_tab = "Guided Mode"
            st.info("💡 Use Guided Mode for complex rules with multiple logical interpretations")
            render_chat_interface()

if __name__ == "__main__":
    main()
