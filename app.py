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


# ==============================================
# Simple Cases Tab Functions
# ==============================================

def generate_prompt_guidance_simple(user_input: str, modification_request: Optional[str] = None) -> str:
    """Generate guidance for the AI with emphasis on exact column names and condition groups"""
    available_data = "\n".join([f"- {f}: {', '.join(cols)}" for f, cols in CSV_STRUCTURES.items()])
    
    base_prompt = f"""
    You are a financial rule generation assistant. Your task is to help create rules for mortgage holders based on available data sources.

    CRITICAL INSTRUCTIONS:
    1. You MUST use ONLY the exact column names from the available data sources
    2. Field names are case-sensitive and must match exactly as provided
    3. If a similar concept exists but with different naming, use the provided column name
    4. For simple AND/OR conditions, create separate conditions with appropriate connectors
    5. For complex nested logic, use conditionGroup with proper nesting
    6. For amounts, use exact column names like "transaction_amount"
    7. For status checks, use exact column names like "account_status"

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
    3. Create conditions or condition groups as needed
    4. Include all these fields for each condition:
       - dataSource (file name exactly as shown)
       - field (column name exactly as shown)
       - eligibilityPeriod (use "Rolling 30 days" for time-based conditions, otherwise "N/A")
       - function (use "sum", "count", "avg" where appropriate, otherwise "N/A")
       - operator (use "=", ">", "<", ">=", "<=", "!=" as appropriate)
       - value (use exact values from user request)
    5. For condition groups, include:
       - conditions (array of conditions or nested condition groups)
       - connector ("AND" or "OR")
    6. Output the rule in JSON format matching this schema:
        {
            "rules": [
                {
                    "id": "generated_id",
                    "ruleType": "condition" or "conditionGroup",
                    // For conditions:
                    "dataSource": "source_name",
                    "field": "column_name",
                    "eligibilityPeriod": "time_period or N/A",
                    "function": "aggregation_function or N/A",
                    "operator": "comparison_operator",
                    "value": "comparison_value",
                    "priority": null,
                    "connector": "AND" or "OR" or null
                    // For condition groups:
                    "conditions": [ array of conditions/groups ],
                    "groupConnector": "AND" or "OR"
                }
            ]
        }

    Respond ONLY with the JSON output. Do not include any additional explanation or markdown formatting.
    The rule should be as specific as possible to match the user's requirements.
    """
    
    return base_prompt

def validate_and_correct_rule_simple(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and correct the rule structure"""
    if not rule or "rules" not in rule:
        return rule
    
    # Ensure all items have IDs and proper structure
    for rule_item in rule["rules"]:
        if "id" not in rule_item:
            rule_item["id"] = str(uuid.uuid4())
        
        if rule_item.get("ruleType") == "conditionGroup":
            if "conditions" not in rule_item:
                rule_item["conditions"] = []
            if "groupConnector" not in rule_item:
                rule_item["groupConnector"] = "AND"
    
    return rule

def generate_rule_with_openai_simple(user_input: str, modification_request: Optional[str] = None) -> Dict[str, Any]:
    """Use OpenAI to generate a rule based on user input"""
    prompt = generate_prompt_guidance_simple(user_input, modification_request)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial rule generation expert that creates precise JSON rules using EXACT column names from provided data sources."
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
        
        # Validate and correct the rule structure
        return validate_and_correct_rule_simple(rule)
    
    except Exception as e:
        st.error(f"Error generating rule: {str(e)}")
        return None

def render_condition_simple(rule_item: Dict[str, Any], key_prefix: str = "") -> None:
    """Render a single condition in the UI"""
    cols = st.columns(7)
    with cols[0]:
        # Data source dropdown with exact file names
        selected_ds = st.selectbox(
            "Data Source",
            options=list(CSV_STRUCTURES.keys()),
            index=list(CSV_STRUCTURES.keys()).index(rule_item["dataSource"]) 
            if rule_item["dataSource"] in CSV_STRUCTURES else 0,
            key=f"{key_prefix}_ds_simple"
        )
    with cols[1]:
        # Field dropdown with exact column names for selected data source
        columns = CSV_STRUCTURES.get(selected_ds, [])
        selected_field = st.selectbox(
            "Field", 
            options=columns,
            index=columns.index(rule_item["field"]) 
            if rule_item["field"] in columns else 0,
            key=f"{key_prefix}_field_simple"
        )
    with cols[2]:
        st.selectbox("eligibilityPeriod", 
                    ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"],
                    index=0 if rule_item.get("eligibilityPeriod") == "N/A" else 1,
                    key=f"{key_prefix}_period_simple")
    with cols[3]:
        st.selectbox("function", 
                    ["N/A", "sum", "count", "avg", "max", "min"],
                    index=0 if rule_item.get("function") == "N/A" else 1,
                    key=f"{key_prefix}_func_simple")
    with cols[4]:
        # Operator selection with correct default
        operator_options = ["=", ">", "<", ">=", "<=", "!=", "contains"]
        operator_index = operator_options.index(rule_item["operator"]) if rule_item["operator"] in operator_options else 0
        st.selectbox("Operator", 
                    operator_options,
                    index=operator_index,
                    key=f"{key_prefix}_op_simple")
    with cols[5]:
        # Display the exact value from the rule
        st.text_input("Value", value=str(rule_item.get("value", "")), 
                    key=f"{key_prefix}_val_simple")
    with cols[6]:
        # Connector for conditions (not for the last item)
        st.selectbox("Connector", 
                    ["AND", "OR", "NONE"],
                    index=0 if rule_item.get("connector", "AND") == "AND" else 
                          (1 if rule_item.get("connector") == "OR" else 2),
                    key=f"{key_prefix}_conn_simple")

def render_condition_group_simple(group: Dict[str, Any], group_index: int) -> None:
    """Render a condition group in the UI"""
    with st.expander(f"Condition Group {group_index + 1}", expanded=True):
        # Group connector selection
        group_connector = st.selectbox(
            "Group Connector",
            ["AND", "OR"],
            index=0 if group.get("groupConnector", "AND") == "AND" else 1,
            key=f"group_{group_index}_connector_simple"
        )
        
        # Render each condition in the group
        for i, condition in enumerate(group.get("conditions", [])):
            if condition.get("ruleType") == "condition":
                render_condition_simple(condition, key_prefix=f"group_{group_index}_cond_{i}_simple")
            elif condition.get("ruleType") == "conditionGroup":
                render_condition_group_simple(condition, i)

def display_rule_ui_simple(rule: Dict[str, Any]) -> None:
    """Display the rule in the UI with support for condition groups"""
    if not rule or "rules" not in rule:
        st.warning("No valid rule generated yet")
        return
    
    st.subheader("Rule Conditions")
    st.markdown("Define the logical conditions for this rule to apply.")
    
    # Priority checkbox
    st.checkbox("Enable priority order and drag & drop", value=False, key="priority_order_simple")
    
    # Main rule display
    for i, rule_item in enumerate(rule["rules"]):
        if rule_item.get("ruleType") == "condition":
            with st.expander(f"Condition {i+1}", expanded=True):
                render_condition_simple(rule_item, key_prefix=f"cond_{i}_simple")
        elif rule_item.get("ruleType") == "conditionGroup":
            render_condition_group_simple(rule_item, i)

def initialize_session_state_simple():
    """Initialize all session state variables for simple cases"""
    if "messages_simple" not in st.session_state:
        st.session_state.messages_simple = [
            {"role": "assistant", "content": "Hello! I can help you create mortgage holder rules. What criteria would you like to use?"}
        ]
    if "current_rule_simple" not in st.session_state:
        st.session_state.current_rule_simple = None
    if "confirmed_simple" not in st.session_state:
        st.session_state.confirmed_simple = False
    if "user_prompt_simple" not in st.session_state:
        st.session_state.user_prompt_simple = ""
    if "awaiting_confirmation_simple" not in st.session_state:
        st.session_state.awaiting_confirmation_simple = False
    if "awaiting_modification_simple" not in st.session_state:
        st.session_state.awaiting_modification_simple = False

def display_chat_message_simple(role: str, content: str):
    """Display a chat message in the UI"""
    with st.chat_message(role):
        if role == "user":
            content = clean_user_input(content)
        st.markdown(content)

def handle_user_confirmation_simple(confirmation: bool):
    """Handle user confirmation or modification request"""
    if confirmation:
        st.session_state.confirmed_simple = True
        st.session_state.awaiting_confirmation_simple = False
        st.session_state.messages_simple.append({"role": "assistant", "content": "Great! Here's your final rule:"})
    else:
        st.session_state.awaiting_confirmation_simple = False
        st.session_state.awaiting_modification_simple = True
        st.session_state.messages_simple.append({"role": "assistant", "content": "What changes would you like to make to the rule?"})

def generate_new_rule_simple():
    """Generate a new rule based on current state"""
    modification_request = None
    if st.session_state.awaiting_modification_simple and st.session_state.messages_simple[-1]["role"] == "user":
        modification_request = clean_user_input(st.session_state.messages_simple[-1]["content"])
    
    with st.spinner("Generating rule..."):
        new_rule = generate_rule_with_openai_simple(
            st.session_state.user_prompt_simple,
            modification_request
        )
        
        if new_rule:
            st.session_state.current_rule_simple = new_rule
            rule_preview = json.dumps(new_rule, indent=2)
            st.session_state.messages_simple.append({
                "role": "assistant",
                "content": f"I've generated this rule:\n\n```json\n{rule_preview}\n```\n\nDoes this meet your requirements?"
            })
            st.session_state.awaiting_confirmation_simple = True
            st.session_state.awaiting_modification_simple = False
        else:
            st.session_state.messages_simple.append({
                "role": "assistant",
                "content": "I couldn't generate a valid rule. Could you please provide more details?"
            })

# ==============================================
# Complex Cases Tab Functions (Updated)
# ==============================================

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

def generate_prompt_guidance_complex(logical_structure: str, user_input: str) -> str:
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
    5. For BETWEEN conditions, create two separate conditions with >= and <=
    6. Include proper connectors (AND/OR) at all levels
    
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
                "connector": "AND" or "OR" or null,
                // For condition groups:
                "conditions": [ array of conditions/groups ],
                "groupConnector": "AND" or "OR"
            }}
        ],
        "logical_structure": "{logical_structure}",
        "topLevelConnector": "AND" or "OR" (for rules at top level)
    }}

    Respond ONLY with the JSON output.
    """
    return base_prompt

def validate_and_correct_rule_complex(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and correct the rule structure"""
    if not rule or "rules" not in rule:
        return {"rules": [], "logical_structure": "", "topLevelConnector": "OR"}
    
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
    
    # Ensure top level connector exists
    if "topLevelConnector" not in rule:
        rule["topLevelConnector"] = "OR" if len(rule["rules"]) > 1 else None
    
    return rule

def generate_rule_with_openai_complex(logical_structure: str, user_input: str) -> Dict[str, Any]:
    """Generate rule with proper logical grouping"""
    prompt = generate_prompt_guidance_complex(logical_structure, user_input)
    
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
        return validate_and_correct_rule_complex(rule)
    except Exception as e:
        st.error(f"Error generating rule: {str(e)}")
        return None

def render_condition_complex(rule_item: Dict[str, Any], key_prefix: str = "") -> None:
    """Render a single condition in the UI"""
    cols = st.columns(7)
    with cols[0]:
        selected_ds = st.selectbox(
            "Data Source",
            options=list(CSV_STRUCTURES.keys()),
            index=list(CSV_STRUCTURES.keys()).index(rule_item["dataSource"]) 
            if rule_item["dataSource"] in CSV_STRUCTURES else 0,
            key=f"{key_prefix}_ds_complex"
        )
    with cols[1]:
        columns = CSV_STRUCTURES.get(selected_ds, [])
        selected_field = st.selectbox(
            "Field", 
            options=columns,
            index=columns.index(rule_item["field"]) 
            if rule_item["field"] in columns else 0,
            key=f"{key_prefix}_field_complex"
        )
    with cols[2]:
        st.selectbox("eligibilityPeriod", 
                    ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days"],
                    index=0 if rule_item.get("eligibilityPeriod") == "N/A" else 1,
                    key=f"{key_prefix}_period_complex")
    with cols[3]:
        st.selectbox("function", 
                    ["N/A", "sum", "count", "avg", "max", "min"],
                    index=0 if rule_item.get("function") == "N/A" else 1,
                    key=f"{key_prefix}_func_complex")
    with cols[4]:
        operator_options = ["=", ">", "<", ">=", "<=", "!=", "contains"]
        operator_index = operator_options.index(rule_item["operator"]) if rule_item["operator"] in operator_options else 0
        st.selectbox("Operator", 
                    operator_options,
                    index=operator_index,
                    key=f"{key_prefix}_op_complex")
    with cols[5]:
        st.text_input("Value", value=str(rule_item.get("value", "")), 
                    key=f"{key_prefix}_val_complex")
    with cols[6]:
        if rule_item.get("connector"):
            st.selectbox("Connector", 
                        ["AND", "OR"],
                        index=0 if rule_item.get("connector") == "AND" else 1,
                        key=f"{key_prefix}_conn_complex")

def render_condition_group_complex(group: Dict[str, Any], group_index: int, is_top_level: bool = False) -> None:
    """Render a condition group in the UI"""
    group_title = f"Group {group_index + 1}" if not is_top_level else "Top Level Group"
    
    with st.expander(f"{group_title} ({group['groupConnector']})", expanded=True):
        for i, condition in enumerate(group.get("conditions", [])):
            if condition.get("ruleType") == "condition":
                render_condition_complex(condition, key_prefix=f"group_{group['id']}_cond_{i}_complex")
            elif condition.get("ruleType") == "conditionGroup":
                render_condition_group_complex(condition, i)
            
            # Show connector after each condition except the last one
            if i < len(group.get("conditions", [])) - 1 and "connector" in condition:
                st.markdown(f"<div class='connector-display'>{condition['connector']}</div>", unsafe_allow_html=True)

def display_rule_ui_complex(rule: Dict[str, Any]) -> None:
    """Display the rule in the UI with proper connectors"""
    if not rule or "rules" not in rule:
        return
    
    st.subheader("Rule Structure")
    st.code(rule.get("logical_structure", ""), language="text")
    
    # Show top level connector if exists
    if "topLevelConnector" in rule and len(rule["rules"]) > 1:
        st.markdown(f"<div class='connector-display'>Top Level Connector: {rule['topLevelConnector']}</div>", unsafe_allow_html=True)
    
    st.subheader("Conditions")
    
    # Create a top-level group if there are multiple rules with a connector
    if len(rule["rules"]) > 1:
        top_level_group = {
            "id": "top_level",
            "ruleType": "conditionGroup",
            "conditions": rule["rules"],
            "groupConnector": rule.get("topLevelConnector", "OR")
        }
        render_condition_group_complex(top_level_group, 0, is_top_level=True)
    else:
        # Single rule, render directly
        rule_item = rule["rules"][0]
        if rule_item.get("ruleType") == "condition":
            with st.expander(f"Condition 1", expanded=True):
                render_condition_complex(rule_item, key_prefix=f"cond_0_complex")
        elif rule_item.get("ruleType") == "conditionGroup":
            render_condition_group_complex(rule_item, 0)

def initialize_session_state_complex():
    """Initialize session state variables for complex cases"""
    if "messages_complex" not in st.session_state:
        st.session_state.messages_complex = [
            {"role": "assistant", "content": "Hello! I can help create mortgage holder rules. What criteria would you like to use?"}
        ]
    if "current_rule_complex" not in st.session_state:
        st.session_state.current_rule_complex = None
    if "user_prompt_complex" not in st.session_state:
        st.session_state.user_prompt_complex = ""
    if "awaiting_structure_confirmation_complex" not in st.session_state:
        st.session_state.awaiting_structure_confirmation_complex = False
    if "awaiting_rule_confirmation_complex" not in st.session_state:
        st.session_state.awaiting_rule_confirmation_complex = False
    if "proposed_structures_complex" not in st.session_state:
        st.session_state.proposed_structures_complex = []
    if "current_structure_index_complex" not in st.session_state:
        st.session_state.current_structure_index_complex = 0
    if "confirmed_structure_complex" not in st.session_state:
        st.session_state.confirmed_structure_complex = ""
    if "confirmed_rule_complex" not in st.session_state:
        st.session_state.confirmed_rule_complex = False

def display_chat_message_complex(role: str, content: str):
    """Display a chat message"""
    with st.chat_message(role):
        st.markdown(content)

def handle_structure_confirmation_complex(confirmation: bool):
    """Handle user confirmation of logical structure"""
    if confirmation:
        st.session_state.confirmed_structure_complex = st.session_state.proposed_structures_complex[st.session_state.current_structure_index_complex]
        st.session_state.messages_complex.append({
            "role": "assistant",
            "content": f"Great! I'll generate rules for this structure:\n\n```\n{st.session_state.confirmed_structure_complex}\n```"
        })
        st.session_state.awaiting_structure_confirmation_complex = False
    else:
        # Move to next alternative structure
        st.session_state.current_structure_index_complex += 1
        if st.session_state.current_structure_index_complex < len(st.session_state.proposed_structures_complex):
            show_next_alternative_complex()
        else:
            st.session_state.messages_complex.append({
                "role": "assistant",
                "content": "I've shown all possible interpretations. Please rephrase your requirements with clearer logical grouping."
            })
            st.session_state.awaiting_structure_confirmation_complex = False
            reset_structure_state_complex()

def handle_rule_confirmation_complex(confirmation: bool):
    """Handle user confirmation of generated rule"""
    if confirmation:
        st.session_state.confirmed_rule_complex = True
        st.session_state.messages_complex.append({
            "role": "assistant",
            "content": "Rule confirmed! Here's your final rule:"
        })
    else:
        st.session_state.messages_complex.append({
            "role": "assistant",
            "content": "What changes would you like to make to the rule?"
        })
        st.session_state.current_rule_complex = None
    st.session_state.awaiting_rule_confirmation_complex = False

def show_next_alternative_complex():
    """Show the next alternative structure to the user"""
    structure = st.session_state.proposed_structures_complex[st.session_state.current_structure_index_complex]
    st.session_state.messages_complex.append({
        "role": "assistant",
        "content": f"Does this structure match your intention?\n\n```\n{structure}\n```\n\nPlease respond with 'yes' or 'no'."
    })
    st.session_state.awaiting_structure_confirmation_complex = True

def reset_structure_state_complex():
    """Reset structure-related session state"""
    st.session_state.proposed_structures_complex = []
    st.session_state.current_structure_index_complex = 0
    st.session_state.confirmed_structure_complex = ""

def generate_new_rule_complex():
    """Generate a new rule based on current state"""
    if not st.session_state.user_prompt_complex:
        return
    
    # Step 1: Generate and confirm logical structure
    if not st.session_state.proposed_structures_complex and not st.session_state.confirmed_structure_complex:
        with st.spinner("Analyzing possible logical structures..."):
            st.session_state.proposed_structures_complex = generate_alternative_structures(st.session_state.user_prompt_complex)
            if st.session_state.proposed_structures_complex:
                show_next_alternative_complex()
            else:
                st.session_state.messages_complex.append({
                    "role": "assistant",
                    "content": "I couldn't interpret the logical structure. Please provide more specific requirements."
                })
        return
    
    # Step 2: Generate rule after structure confirmation
    if st.session_state.confirmed_structure_complex and not st.session_state.current_rule_complex and not st.session_state.awaiting_rule_confirmation_complex:
        with st.spinner("Generating rule with confirmed structure..."):
            rule = generate_rule_with_openai_complex(
                st.session_state.confirmed_structure_complex,
                st.session_state.user_prompt_complex
            )
            if rule:
                st.session_state.current_rule_complex = rule
                rule_preview = json.dumps(rule, indent=2)
                st.session_state.messages_complex.append({
                    "role": "assistant",
                    "content": f"I've generated this rule based on your confirmed structure:\n\n```json\n{rule_preview}\n```\n\nDoes this meet your requirements? (yes/no)"
                })
                st.session_state.awaiting_rule_confirmation_complex = True
            else:
                st.session_state.messages_complex.append({
                    "role": "assistant",
                    "content": "I couldn't generate a valid rule. Please provide more details."
                })

def main():
    st.set_page_config(page_title="Mortgage Rule Generator", layout="wide")
    st.title("🏦 Mortgage Rule Generator with GPT-4o")
    
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
        .condition-group {
            border-left: 4px solid #4285f4;
            padding-left: 12px;
            margin-left: 8px;
            margin-bottom: 15px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #f0f2f6;
        }
        .connector-display {
            background-color: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            margin: 8px 0;
            text-align: center;
            font-weight: bold;
            border: 1px solid #e0e0e0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session states
    initialize_session_state_complex()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Simple Cases", "Complex Cases"])
    with tab1:
        # Simple Cases Tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display rule UI
            if st.session_state.current_rule_simple:
                display_rule_ui_simple(st.session_state.current_rule_simple)
                
                # Show final JSON if confirmed
                if st.session_state.confirmed_simple:
                    st.success("✅ Final Rule Confirmed")
                    st.json(st.session_state.current_rule_simple)
                    
                    # Add download button
                    json_str = json.dumps(st.session_state.current_rule_simple, indent=2)
                    st.download_button(
                        label="Download Rule JSON",
                        data=json_str,
                        file_name=f"mortgage_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_simple"
                    )
                    
                    if st.button("Create New Rule", key="new_rule_simple"):
                        # Reset for new rule
                        st.session_state.messages_simple = [
                            {"role": "assistant", "content": "Let's create a new rule. What criteria would you like to use?"}
                        ]
                        st.session_state.current_rule_simple = None
                        st.session_state.confirmed_simple = False
                        st.session_state.user_prompt_simple = ""
                        st.rerun()
        
        with col2:
            # Display chat messages
            st.subheader("Rule Assistant")
            
            for message in st.session_state.messages_simple:
                display_chat_message_simple(message["role"], message["content"])
            
            # Handle user input
            if prompt := st.chat_input("Type your message here...", key="chat_input_simple"):
                # Clean the user input first
                cleaned_prompt = clean_user_input(prompt)
                st.session_state.messages_simple.append({"role": "user", "content": cleaned_prompt})
                display_chat_message_simple("user", cleaned_prompt)
                
                # Determine what to do based on current state
                if not st.session_state.user_prompt_simple:
                    # First prompt - generate initial rule
                    st.session_state.user_prompt_simple = cleaned_prompt
                    generate_new_rule_simple()
                    st.rerun()
                
                elif st.session_state.awaiting_confirmation_simple:
                    # User is responding to confirmation question
                    if "yes" in cleaned_prompt.lower() or "correct" in cleaned_prompt.lower():
                        handle_user_confirmation_simple(True)
                    else:
                        handle_user_confirmation_simple(False)
                    st.rerun()
                
                elif st.session_state.awaiting_modification_simple:
                    # User is providing modification details
                    generate_new_rule_simple()
                    st.rerun()
                
                else:
                    # New conversation
                    st.session_state.user_prompt_simple = cleaned_prompt
                    st.session_state.current_rule_simple = None
                    st.session_state.confirmed_simple = False
                    generate_new_rule_simple()
                    st.rerun()
    
    with tab2:
        # Complex Cases Tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.session_state.current_rule_complex:
                display_rule_ui_complex(st.session_state.current_rule_complex)
                
                if st.session_state.confirmed_rule_complex:
                    st.success("✅ Final Rule Confirmed")
                    st.json(st.session_state.current_rule_complex)
                    
                    json_str = json.dumps(st.session_state.current_rule_complex, indent=2)
                    st.download_button(
                        label="Download Rule JSON",
                        data=json_str,
                        file_name=f"mortgage_rule_complex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_complex"
                    )
                    
                    if st.button("Create New Rule", key="new_rule_complex"):
                        st.session_state.messages_complex = [
                            {"role": "assistant", "content": "Let's create a new rule. What criteria would you like to use?"}
                        ]
                        st.session_state.current_rule_complex = None
                        st.session_state.user_prompt_complex = ""
                        reset_structure_state_complex()
                        st.session_state.confirmed_rule_complex = False
                        st.rerun()
        
        with col2:
            st.subheader("Conversation")
            
            for message in st.session_state.messages_complex:
                display_chat_message_complex(message["role"], message["content"])
            
            if prompt := st.chat_input("Type your message here...", key="chat_input_complex"):
                cleaned_prompt = clean_user_input(prompt)
                st.session_state.messages_complex.append({"role": "user", "content": cleaned_prompt})
                
                if st.session_state.awaiting_structure_confirmation_complex:
                    if "yes" in cleaned_prompt.lower():
                        handle_structure_confirmation_complex(True)
                    elif "no" in cleaned_prompt.lower():
                        handle_structure_confirmation_complex(False)
                    st.rerun()
                
                elif st.session_state.awaiting_rule_confirmation_complex:
                    if "yes" in cleaned_prompt.lower():
                        handle_rule_confirmation_complex(True)
                    elif "no" in cleaned_prompt.lower():
                        handle_rule_confirmation_complex(False)
                    st.rerun()
                
                else:
                    st.session_state.user_prompt_complex = cleaned_prompt
                    st.session_state.current_rule_complex = None
                    reset_structure_state_complex()
                    st.session_state.confirmed_rule_complex = False
                    generate_new_rule_complex()
                    st.rerun()
            
            generate_new_rule_complex()

if __name__ == "__main__":
    main()
