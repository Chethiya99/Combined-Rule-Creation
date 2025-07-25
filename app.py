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

def detect_logical_structure(text: str) -> Dict[str, Any]:
    """Analyze user input to detect logical structure"""
    text_lower = text.lower()
    structure = {
        "primary_connector": "AND",  # Default
        "secondary_connector": None,
        "has_complex_conditions": False,
        "connectors": []
    }
    
    # Check for explicit connector mentions
    if " and " in text_lower and " or " in text_lower:
        structure["has_complex_conditions"] = True
        # Count occurrences to determine primary connector
        and_count = text_lower.count(" and ")
        or_count = text_lower.count(" or ")
        structure["primary_connector"] = "AND" if and_count > or_count else "OR"
        structure["secondary_connector"] = "OR" if structure["primary_connector"] == "AND" else "AND"
    elif " and " in text_lower:
        structure["primary_connector"] = "AND"
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
    5. For complex conditions with both AND and OR, create proper nested condition groups
    6. For simple conditions, use direct connectors between conditions
    7. For amounts, use exact column names like "transaction_amount" or "loan_balance"
    8. For status checks, use exact column names like "account_status"
    9. For time periods like "last month", use "Rolling 30 days" as eligibilityPeriod

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
    3. Create proper logical structure matching the user's phrasing
    4. Include all these fields for each condition:
       - dataSource (file name exactly as shown)
       - field (column name exactly as shown)
       - eligibilityPeriod (use "Rolling 30 days" for time-based conditions, otherwise "N/A")
       - function (use "sum", "count", "avg" where appropriate, otherwise "N/A")
       - operator (use "=", ">", "<", ">=", "<=", "!=" as appropriate)
       - value (use exact values from user request)
    5. Output format examples:

    For simple AND:
    {
        "rules": [
            { /* first condition with connector */ },
            { /* second condition with null connector */ }
        ]
    }

    For complex AND/OR:
    {
        "rules": [
            {
                "ruleType": "conditionGroup",
                "connector": "AND",
                "conditions": [
                    { /* first condition */ },
                    {
                        "ruleType": "conditionGroup",
                        "connector": "OR",
                        "conditions": [
                            { /* nested condition */ },
                            { /* nested condition */ }
                        ]
                    }
                ]
            }
        ]
    }

    Respond ONLY with the JSON output. Do not include any additional explanation or markdown formatting.
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
                    if not validate_rule_structure({"rules": cond.get("conditions", [])}):
                        return False
        else:
            required_fields = ["dataSource", "field", "operator", "value"]
            if not all(key in rule_item for key in required_fields):
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
            model="gpt-4",
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

def display_rule_ui(rule: Dict[str, Any]) -> None:
    """Display the rule in the UI with all required fields"""
    if not rule or "rules" not in rule:
        st.warning("No valid rule generated yet")
        return
    
    st.subheader("Rule Conditions")
    st.markdown("Define the logical conditions for this rule to apply.")
    
    # Priority checkbox
    st.checkbox("Enable priority order and drag & drop", value=False, key="priority_order")
    
    # Track condition IDs to avoid duplicates
    condition_ids = set()
    
    def render_condition(condition, index, parent_index=None, is_nested=False):
        """Recursively render conditions and condition groups"""
        key_suffix = f"{parent_index}_{index}" if parent_index is not None else str(index)
        
        if condition.get("ruleType") == "conditionGroup":
            with st.expander(f"Condition Group {key_suffix}", expanded=True):
                st.markdown(f"#### Group Connector: {condition.get('connector', 'AND')}")
                for i, cond in enumerate(condition.get("conditions", [])):
                    render_condition(cond, i, key_suffix, True)
        else:
            # Generate unique ID for each condition
            cond_id = f"{condition.get('dataSource','')}_{condition.get('field','')}_{index}"
            if cond_id in condition_ids:
                cond_id = f"{cond_id}_{parent_index}"
            condition_ids.add(cond_id)
            
            with st.expander(f"Condition {cond_id}", expanded=True):
                cols = st.columns(7)
                with cols[0]:
                    # Data source dropdown with exact file names
                    selected_ds = st.selectbox(
                        "Data Source",
                        options=list(CSV_STRUCTURES.keys()),
                        index=list(CSV_STRUCTURES.keys()).index(condition["dataSource"]) 
                        if condition["dataSource"] in CSV_STRUCTURES else 0,
                        key=f"ds_{cond_id}"
                    )
                with cols[1]:
                    # Field dropdown with exact column names for selected data source
                    columns = CSV_STRUCTURES.get(selected_ds, [])
                    selected_field = st.selectbox(
                        "Field", 
                        options=columns,
                        index=columns.index(condition["field"]) 
                        if condition["field"] in columns else 0,
                        key=f"field_{cond_id}"
                    )
                with cols[2]:
                    period_options = ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"]
                    period_index = period_options.index(condition["eligibilityPeriod"]) if condition.get("eligibilityPeriod") in period_options else 0
                    st.selectbox("eligibilityPeriod", 
                                period_options,
                                index=period_index,
                                key=f"period_{cond_id}")
                with cols[3]:
                    func_options = ["N/A", "sum", "count", "avg", "max", "min"]
                    func_index = func_options.index(condition["function"]) if condition.get("function") in func_options else 0
                    st.selectbox("function", 
                                func_options,
                                index=func_index,
                                key=f"func_{cond_id}")
                with cols[4]:
                    op_options = ["=", ">", "<", ">=", "<=", "!=", "contains"]
                    op_index = op_options.index(condition["operator"]) if condition.get("operator") in op_options else 0
                    st.selectbox("Operator", 
                                op_options,
                                index=op_index,
                                key=f"op_{cond_id}")
                with cols[5]:
                    st.text_input("Value", value=str(condition.get("value", "")), 
                                key=f"val_{cond_id}")
                
                if not is_nested and index < len(rule["rules"]) - 1:
                    with cols[6]:
                        st.selectbox("Connector", 
                                    ["AND", "OR"],
                                    index=0 if condition.get("connector", "AND") == "AND" else 1,
                                    key=f"conn_{cond_id}")
    
    # Render all rules
    for i, rule_item in enumerate(rule["rules"]):
        render_condition(rule_item, i)

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can help you create mortgage holder rules. What criteria would you like to use?"}
        ]
    if "current_rule" not in st.session_state:
        st.session_state.current_rule = None
    if "confirmed" not in st.session_state:
        st.session_state.confirmed = False
    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = ""
    if "awaiting_confirmation" not in st.session_state:
        st.session_state.awaiting_confirmation = False
    if "awaiting_modification" not in st.session_state:
        st.session_state.awaiting_modification = False

def display_chat_message(role: str, content: str):
    """Display a chat message in the UI"""
    with st.chat_message(role):
        if role == "user":
            content = clean_user_input(content)
        st.markdown(content)

def handle_user_confirmation(confirmation: bool):
    """Handle user confirmation or modification request"""
    if confirmation:
        st.session_state.confirmed = True
        st.session_state.awaiting_confirmation = False
        st.session_state.messages.append({"role": "assistant", "content": "Great! Here's your final rule:"})
    else:
        st.session_state.awaiting_confirmation = False
        st.session_state.awaiting_modification = True
        st.session_state.messages.append({"role": "assistant", "content": "What changes would you like to make to the rule?"})

def generate_new_rule():
    """Generate a new rule based on current state"""
    modification_request = None
    if st.session_state.awaiting_modification and st.session_state.messages[-1]["role"] == "user":
        modification_request = clean_user_input(st.session_state.messages[-1]["content"])
    
    with st.spinner("Generating rule..."):
        new_rule = generate_rule_with_openai(
            st.session_state.user_prompt,
            modification_request
        )
        
        if new_rule:
            st.session_state.current_rule = new_rule
            rule_preview = json.dumps(new_rule, indent=2)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I've generated this rule:\n\n```json\n{rule_preview}\n```\n\nDoes this meet your requirements?"
            })
            st.session_state.awaiting_confirmation = True
            st.session_state.awaiting_modification = False
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I couldn't generate a valid rule. Could you please provide more details?"
            })

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
