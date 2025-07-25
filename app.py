import streamlit as st
import pandas as pd
import openai
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

# Initialize OpenAI client
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
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

def analyze_logical_structure(text: str) -> Dict[str, Any]:
    """Analyze user input to detect logical structure with better precision"""
    text_lower = text.lower()
    structure = {
        "primary_connector": "AND",  # Default
        "secondary_connector": None,
        "has_complex_conditions": False,
        "condition_groups": []
    }
    
    # Split into main parts
    parts = []
    if " and " in text_lower and " or " in text_lower:
        structure["has_complex_conditions"] = True
        # Split by the primary connector first
        if text_lower.count(" and ") > text_lower.count(" or "):
            structure["primary_connector"] = "AND"
            parts = [part.strip() for part in text.split(" AND ")]
        else:
            structure["primary_connector"] = "OR"
            parts = [part.strip() for part in text.split(" OR ")]
        
        # Further split parts containing the secondary connector
        processed_parts = []
        for part in parts:
            if (" and " in part.lower() if structure["primary_connector"] == "OR" else 
                " or " in part.lower()):
                subparts = [subpart.strip() for subpart in (
                    part.split(" OR ") if structure["primary_connector"] == "AND" 
                    else part.split(" AND ")
                )]
                processed_parts.append(subparts)
            else:
                processed_parts.append(part)
        structure["condition_groups"] = processed_parts
    elif " and " in text_lower:
        structure["primary_connector"] = "AND"
        parts = [part.strip() for part in text.split(" AND ")]
        structure["condition_groups"] = parts
    elif " or " in text_lower:
        structure["primary_connector"] = "OR"
        parts = [part.strip() for part in text.split(" OR ")]
        structure["condition_groups"] = parts
    else:
        structure["condition_groups"] = [text.strip()]
    
    return structure

def generate_prompt_guidance(user_input: str, modification_request: Optional[str] = None) -> str:
    """Generate guidance for the AI with improved logical structure handling"""
    available_data = "\n".join([f"- {f}: {', '.join(cols)}" for f, cols in CSV_STRUCTURES.items()])
    logic = analyze_logical_structure(user_input)
    
    base_prompt = f"""
    You are a financial rule generation assistant. Your task is to create rules for mortgage holders based on available data sources.

    CRITICAL INSTRUCTIONS:
    1. Use ONLY these exact column names from the available data sources
    2. Field names are case-sensitive and must match exactly as provided
    3. For time periods like "last month", use "Rolling 30 days" as eligibilityPeriod
    4. For amounts, use exact column names like "transaction_amount" or "loan_balance"
    5. For status checks, use exact column names like "account_status"
    6. Detected logical structure:
       - Primary connector: {logic['primary_connector']}
       - Condition groups: {logic['condition_groups']}

    Available data sources and EXACT columns:
    {available_data}

    User requirement: "{user_input}"
    """
    
    if modification_request:
        base_prompt += f"\nRequested modifications: {modification_request}"
    
    base_prompt += """
    Create the rule by:
    1. Mapping each condition group to the appropriate data source and column
    2. Using proper operators based on the condition text (>, <, =, etc.)
    3. Structuring the rules to match the detected logical structure
    4. Formatting the output as JSON with this structure:

    For simple conditions:
    {
        "rules": [
            {{"dataSource": "...", "field": "...", "operator": "...", "value": "...", "connector": "AND/OR"}},
            {{"dataSource": "...", "field": "...", "operator": "...", "value": "...", "connector": null}}
        ]
    }

    For complex conditions:
    {
        "rules": [
            {{
                "ruleType": "conditionGroup",
                "connector": "primary_connector",
                "conditions": [
                    {{"dataSource": "...", "field": "...", "operator": "...", "value": "..."}},
                    {{
                        "ruleType": "conditionGroup",
                        "connector": "secondary_connector",
                        "conditions": [
                            {{"dataSource": "...", "field": "...", "operator": "...", "value": "..."}},
                            {{"dataSource": "...", "field": "...", "operator": "...", "value": "..."}}
                        ]
                    }}
                ]
            }}
        ]
    }

    Respond ONLY with valid JSON. No additional text or explanations.
    """
    
    return base_prompt

def validate_rule_structure(rule: Dict[str, Any]) -> bool:
    """Validate the rule structure meets all requirements"""
    if not rule or "rules" not in rule:
        return False
    
    for rule_item in rule["rules"]:
        if rule_item.get("ruleType") == "conditionGroup":
            conditions = rule_item.get("conditions", [])
            if not conditions:
                return False
            for cond in conditions:
                if cond.get("ruleType") == "conditionGroup":
                    if not validate_rule_structure({"rules": cond.get("conditions", [])}):
                        return False
                else:
                    if not validate_condition(cond):
                        return False
        else:
            if not validate_condition(rule_item):
                return False
    
    return True

def validate_condition(condition: Dict[str, Any]) -> bool:
    """Validate an individual condition"""
    required_fields = ["dataSource", "field", "operator", "value"]
    if not all(key in condition for key in required_fields):
        return False
    if condition["dataSource"] not in CSV_STRUCTURES:
        return False
    if condition["field"] not in CSV_STRUCTURES[condition["dataSource"]]:
        return False
    return True

def generate_rule_with_openai(user_input: str, modification_request: Optional[str] = None) -> Dict[str, Any]:
    """Generate rule using OpenAI with enhanced error handling"""
    prompt = generate_prompt_guidance(user_input, modification_request)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial rule expert that creates precise JSON rules using exact column names and proper logical structure."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        
        # Extract JSON from response
        try:
            json_str = response_content[response_content.find('{'):response_content.rfind('}')+1]
            rule = json.loads(json_str)
            
            # Validate the rule structure
            if not validate_rule_structure(rule):
                st.error("Generated rule failed validation. Please check your input and try again.")
                return None
                
            return rule
            
        except json.JSONDecodeError:
            st.error("The AI response contained invalid JSON. Please try again.")
            return None
            
    except Exception as e:
        st.error(f"Error communicating with OpenAI: {str(e)}")
        return None

def display_rule_ui(rule: Dict[str, Any]) -> None:
    """Display the rule in the UI with improved rendering"""
    if not rule or "rules" not in rule:
        st.warning("No valid rule generated yet")
        return
    
    st.subheader("Generated Rule Conditions")
    
    def render_condition(condition, index, parent_index=None, is_nested=False):
        """Recursively render conditions with proper formatting"""
        key_prefix = f"{parent_index}_" if parent_index is not None else ""
        key = f"{key_prefix}{index}"
        
        if condition.get("ruleType") == "conditionGroup":
            with st.expander(f"Condition Group ({condition.get('connector', 'AND')})", expanded=True):
                for i, cond in enumerate(condition.get("conditions", [])):
                    render_condition(cond, i, key, True)
        else:
            with st.expander(f"Condition {key}", expanded=True):
                cols = st.columns([2, 2, 2, 2, 2, 2, 1])
                with cols[0]:
                    st.selectbox(
                        "Data Source",
                        options=list(CSV_STRUCTURES.keys()),
                        index=list(CSV_STRUCTURES.keys()).index(condition["dataSource"]),
                        key=f"ds_{key}"
                    )
                with cols[1]:
                    columns = CSV_STRUCTURES[condition["dataSource"]]
                    st.selectbox(
                        "Field",
                        options=columns,
                        index=columns.index(condition["field"]),
                        key=f"field_{key}"
                    )
                with cols[2]:
                    st.selectbox(
                        "Function",
                        options=["N/A", "sum", "count", "avg", "max", "min"],
                        index=0 if condition.get("function", "N/A") == "N/A" else 1,
                        key=f"func_{key}"
                    )
                with cols[3]:
                    st.selectbox(
                        "Period",
                        options=["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days"],
                        index=["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days"].index(
                            condition.get("eligibilityPeriod", "N/A")
                        ),
                        key=f"period_{key}"
                    )
                with cols[4]:
                    st.selectbox(
                        "Operator",
                        options=["=", ">", "<", ">=", "<=", "!=", "contains"],
                        index=["=", ">", "<", ">=", "<=", "!=", "contains"].index(
                            condition["operator"]
                        ) if condition["operator"] in ["=", ">", "<", ">=", "<=", "!=", "contains"] else 0,
                        key=f"op_{key}"
                    )
                with cols[5]:
                    st.text_input(
                        "Value",
                        value=str(condition["value"]),
                        key=f"val_{key}"
                    )
                if not is_nested and "connector" in condition:
                    with cols[6]:
                        st.selectbox(
                            "Connector",
                            options=["AND", "OR"],
                            index=0 if condition["connector"] == "AND" else 1,
                            key=f"conn_{key}"
                        )
    
    for i, rule_item in enumerate(rule["rules"]):
        render_condition(rule_item, i)

# [Rest of the functions (initialize_session_state, display_chat_message, etc.) remain the same as in previous version]

def main():
    st.set_page_config(page_title="Mortgage Rule Generator", layout="wide")
    st.title("🏦 Mortgage Rule Generator with OpenAI")
    
    # Custom CSS
    st.markdown("""
    <style>
        .stChatFloatingInputContainer { bottom: 20px; }
        .stChatMessage { padding: 12px; border-radius: 8px; margin-bottom: 12px; }
        .assistant-message { background-color: #f0f2f6; }
        .user-message { background-color: #e3f2fd; }
        .stTextInput input, .stSelectbox select { font-size: 14px !important; }
        .stExpander { margin-bottom: 15px; border: 1px solid #e0e0e0; border-radius: 8px; }
        .stExpander .streamlit-expanderHeader { font-weight: bold; background-color: #f5f5f5; padding: 10px 15px; }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.current_rule:
            display_rule_ui(st.session_state.current_rule)
            
            if st.session_state.confirmed:
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
                    st.session_state.confirmed = False
                    st.session_state.user_prompt = ""
                    st.rerun()
    
    with col2:
        st.subheader("Rule Assistant")
        
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        if prompt := st.chat_input("Type your message here..."):
            cleaned_prompt = clean_user_input(prompt)
            st.session_state.messages.append({"role": "user", "content": cleaned_prompt})
            display_chat_message("user", cleaned_prompt)
            
            if not st.session_state.user_prompt:
                st.session_state.user_prompt = cleaned_prompt
                generate_new_rule()
                st.rerun()
            elif st.session_state.awaiting_confirmation:
                if "yes" in cleaned_prompt.lower() or "correct" in cleaned_prompt.lower():
                    handle_user_confirmation(True)
                else:
                    handle_user_confirmation(False)
                st.rerun()
            elif st.session_state.awaiting_modification:
                generate_new_rule()
                st.rerun()
            else:
                st.session_state.user_prompt = cleaned_prompt
                st.session_state.current_rule = None
                st.session_state.confirmed = False
                generate_new_rule()
                st.rerun()

if __name__ == "__main__":
    main()

