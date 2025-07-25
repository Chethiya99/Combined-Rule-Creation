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

# Configuration
CSV_DIR = "data"  # Directory containing CSV files

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

def detect_logical_structure(text: str) -> Dict[str, Any]:
    """Analyze user input to detect logical structure"""
    text_lower = text.lower()
    structure = {
        "primary_connector": "AND",  # Default
        "secondary_connector": None,
        "has_complex_conditions": False
    }
    
    # Check for complex conditions with both AND and OR
    if " and " in text_lower and " or " in text_lower:
        structure["has_complex_conditions"] = True
        # Determine which connector appears first to set primary
        and_pos = text_lower.find(" and ")
        or_pos = text_lower.find(" or ")
        if and_pos < or_pos:
            structure["primary_connector"] = "AND"
            structure["secondary_connector"] = "OR"
        else:
            structure["primary_connector"] = "OR"
            structure["secondary_connector"] = "AND"
    elif " or " in text_lower:
        structure["primary_connector"] = "OR"
    
    return structure

def generate_prompt_guidance(user_input: str, modification_request: Optional[str] = None) -> str:
    """Generate guidance for the AI with emphasis on exact column names and logical structure"""
    available_data = "\n".join([f"- {f}: {', '.join(cols)}" for f, cols in CSV_STRUCTURES.items()])
    logic = detect_logical_structure(user_input)
    
    base_prompt = f"""
    You are a financial rule generation assistant. Your task is to help create rules for mortgage holders based on available data sources.

    CRITICAL INSTRUCTIONS:
    1. You MUST use ONLY the exact column names from the available data sources
    2. Field names are case-sensitive and must match exactly as provided
    3. Pay special attention to logical connectors (AND/OR) in the user's requirements
    4. Detected logical structure in user input: Primary connector - {logic['primary_connector']}, Secondary - {logic['secondary_connector']}
    5. For complex conditions with both AND and OR, create a conditionGroup with proper nesting
    6. For simple AND/OR conditions, create separate conditions with connectors
    7. For amounts, use exact column names like "transaction_amount" or "loan_balance"
    8. For status checks, use exact column names like "account_status"

    Available data sources and their EXACT columns:
    {available_data}

    The user has provided this requirement: "{user_input}"
    """
    
    if modification_request:
        base_prompt += f"\nThe user requested these modifications: {modification_request}"
    
    base_prompt += """
    Analyze this requirement and:
    1. Identify which data sources are needed
    2. Use ONLY the exact column names from the sources
    3. Create proper logical structure based on detected connectors
    4. Include all these fields for each condition:
       - dataSource (file name exactly as shown)
       - field (column name exactly as shown)
       - eligibilityPeriod (use "Rolling 30 days" for time-based conditions, otherwise "N/A")
       - function (use "sum", "count", "avg" where appropriate, otherwise "N/A")
       - operator (use "=", ">", "<", ">=", "<=", "!=" as appropriate)
       - value (use exact values from user request)
    5. For complex conditions, use this structure:
        {
            "rules": [
                {
                    "ruleType": "conditionGroup",
                    "connector": "primary_connector",
                    "conditions": [
                        { /* first condition */ },
                        { 
                            "ruleType": "conditionGroup",
                            "connector": "secondary_connector",
                            "conditions": [
                                { /* nested condition */ },
                                { /* nested condition */ }
                            ]
                        }
                    ]
                }
            ]
        }
    6. For simple conditions, use this structure:
        {
            "rules": [
                { /* first condition with connector */ },
                { /* second condition with connector */ },
                { /* last condition with null connector */ }
            ]
        }

    Respond ONLY with the JSON output. Do not include any additional explanation or markdown formatting.
    The rule should precisely match the user's logical requirements.
    """
    
    return base_prompt

def validate_rule_structure(rule: Dict[str, Any]) -> bool:
    """Validate the rule structure meets our requirements"""
    if not rule or "rules" not in rule:
        return False
    
    for rule_item in rule["rules"]:
        if rule_item.get("ruleType") == "conditionGroup":
            conditions = rule_item.get("conditions", [])
            if not conditions:
                return False
            for cond in conditions:
                if cond.get("ruleType") == "conditionGroup":
                    nested_conds = cond.get("conditions", [])
                    if not nested_conds:
                        return False
        else:
            if not all(key in rule_item for key in ["dataSource", "field", "operator", "value"]):
                return False
            if rule_item["dataSource"] not in CSV_STRUCTURES:
                return False
            if rule_item["field"] not in CSV_STRUCTURES[rule_item["dataSource"]]:
                return False
    
    return True

def generate_rule_with_openai(user_input: str, modification_request: Optional[str] = None) -> Dict[str, Any]:
    """Use OpenAI to generate a rule based on user input"""
    prompt = generate_prompt_guidance(user_input, modification_request)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial rule generation expert that creates precise JSON rules with proper logical structure using EXACT column names."
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
        
        # Clean the response to extract just the JSON
        json_str = response_content[response_content.find('{'):response_content.rfind('}')+1]
        rule = json.loads(json_str)
        
        # Validate the rule structure
        if not validate_rule_structure(rule):
            st.error("Generated rule doesn't meet validation requirements")
            return None
            
        return rule
    
    except json.JSONDecodeError:
        st.error("Failed to parse AI response as valid JSON")
        return None
    except Exception as e:
        st.error(f"Error generating rule: {str(e)}")
        return None

# [Rest of your functions remain exactly the same...]
# display_rule_ui()
# initialize_session_state()
# display_chat_message()
# handle_user_confirmation()
# generate_new_rule()
# main()

def main():
    st.set_page_config(page_title="Mortgage Rule Generator", layout="wide")
    st.title("🏦 Mortgage Rule Generator with OpenAI")
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
        .stChatFloatingInputContainer {
            bottom: 20px;
        }
        .stChatMessage {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .assistant-message {
            background-color: #f0f2f6;
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .stTextInput input, .stSelectbox select {
            font-size: 14px !important;
        }
        .stExpander {
            margin-bottom: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        .stExpander .streamlit-expanderHeader {
            font-weight: bold;
            background-color: #f5f5f5;
            padding: 10px 15px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Create main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display rule UI
        if st.session_state.current_rule:
            display_rule_ui(st.session_state.current_rule)
            
            # Show final JSON if confirmed
            if st.session_state.confirmed:
                st.success("✅ Final Rule Confirmed")
                st.json(st.session_state.current_rule)
                
                # Add download button
                json_str = json.dumps(st.session_state.current_rule, indent=2)
                st.download_button(
                    label="Download Rule JSON",
                    data=json_str,
                    file_name=f"mortgage_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                if st.button("Create New Rule"):
                    # Reset for new rule
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Let's create a new rule. What criteria would you like to use?"}
                    ]
                    st.session_state.current_rule = None
                    st.session_state.confirmed = False
                    st.session_state.user_prompt = ""
                    st.rerun()
    
    with col2:
        # Display chat messages
        st.subheader("Rule Assistant")
        
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        # Handle user input
        if prompt := st.chat_input("Type your message here..."):
            # Clean the user input first
            cleaned_prompt = clean_user_input(prompt)
            st.session_state.messages.append({"role": "user", "content": cleaned_prompt})
            display_chat_message("user", cleaned_prompt)
            
            # Determine what to do based on current state
            if not st.session_state.user_prompt:
                # First prompt - generate initial rule
                st.session_state.user_prompt = cleaned_prompt
                generate_new_rule()
                st.rerun()
            
            elif st.session_state.awaiting_confirmation:
                # User is responding to confirmation question
                if "yes" in cleaned_prompt.lower() or "correct" in cleaned_prompt.lower():
                    handle_user_confirmation(True)
                else:
                    handle_user_confirmation(False)
                st.rerun()
            
            elif st.session_state.awaiting_modification:
                # User is providing modification details
                generate_new_rule()
                st.rerun()
            
            else:
                # New conversation
                st.session_state.user_prompt = cleaned_prompt
                st.session_state.current_rule = None
                st.session_state.confirmed = False
                generate_new_rule()
                st.rerun()

if __name__ == "__main__":
    main()
