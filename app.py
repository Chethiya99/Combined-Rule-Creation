import streamlit as st
import pandas as pd
import openai
import groq
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

# Initialize API clients
try:
    openai.api_key = st.secrets.get("OPENAI_API_KEY", "")
    groq_client = groq.Client(api_key=st.secrets.get("GROQ_API_KEY", ""))
except Exception as e:
    st.warning(f"API initialization warning: {str(e)}")

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

def generate_prompt_guidance(user_input: str, modification_request: Optional[str] = None) -> str:
    """Generate guidance for the AI with emphasis on exact column names"""
    available_data = "\n".join([f"- {f}: {', '.join(cols)}" for f, cols in CSV_STRUCTURES.items()])
    
    base_prompt = f"""
You are a financial rule generation assistant. Your task is to help create rules for mortgage holders based on available data sources.

CRITICAL INSTRUCTIONS:
1. You MUST use ONLY the exact column names from the available data sources
2. Field names are case-sensitive and must match exactly as provided
3. If a similar concept exists but with different naming, use the provided column name
4. For conditions with multiple criteria:
   - Create separate conditions for each criteria
   - Use 'AND' connector between conditions that must both be true
   - Use 'OR' connector for alternative conditions
   - The last condition should have connector: null
5. For amounts, use exact column names like "transaction_amount"
6. For status checks, use exact column names like "account_status"
7. For time-based conditions, use "Rolling 30 days" for "last month"

AVAILABLE DATA SOURCES AND COLUMNS:
{available_data}

USER REQUIREMENT: "{user_input}"
"""
    
    if modification_request:
        base_prompt += f"\nMODIFICATION REQUEST: {modification_request}"
    
    # Add examples of complex conditions
    base_prompt += """

EXAMPLES OF COMPLEX CONDITIONS:

Example 1: "Customers with active mortgages AND loan balance > 100,000"
[
  {{
    "dataSource": "sample_mortgage_accounts.csv",
    "field": "account_status",
    "operator": "=",
    "value": "Active",
    "connector": "AND"
  }},
  {{
    "dataSource": "sample_mortgage_accounts.csv",
    "field": "loan_balance",
    "operator": ">",
    "value": "100000",
    "connector": null
  }}
]

Example 2: "Customers who spent over $2500 on credit cards last month OR have loan balances > 1000"
[
  {{
    "dataSource": "sample_credit_card_transactions.csv",
    "field": "transaction_date",
    "eligibilityPeriod": "Rolling 30 days",
    "function": "sum",
    "field": "transaction_amount",
    "operator": ">",
    "value": "2500",
    "connector": "OR"
  }},
  {{
    "dataSource": "sample_mortgage_accounts.csv",
    "field": "loan_balance",
    "operator": ">",
    "value": "1000",
    "connector": null
  }}
]

Example 3: "Active mortgage AND (loan balance > 100,000 OR credit card spend > $5000 last month)"
[
  {{
    "dataSource": "sample_mortgage_accounts.csv",
    "field": "account_status",
    "operator": "=",
    "value": "Active",
    "connector": "AND"
  }},
  {{
    "dataSource": "sample_mortgage_accounts.csv",
    "field": "loan_balance",
    "operator": ">",
    "value": "100000",
    "connector": "OR"
  }},
  {{
    "dataSource": "sample_credit_card_transactions.csv",
    "field": "transaction_date",
    "eligibilityPeriod": "Rolling 30 days",
    "function": "sum",
    "field": "transaction_amount",
    "operator": ">",
    "value": "5000",
    "connector": null
  }}
]

OUTPUT REQUIREMENTS:
- Create a JSON array of condition objects
- Each condition must include:
  - dataSource (exact file name)
  - field (exact column name)
  - eligibilityPeriod ("Rolling 30 days" for time-based, else "N/A")
  - function ("sum", "count", "avg" where appropriate, else "N/A")
  - operator ("=", ">", "<", ">=", "<=", "!=", "contains")
  - value (exact comparison value)
  - connector ("AND", "OR", or null for last condition)
- Omit any fields not relevant to the condition
- Output ONLY the JSON array, no additional text

RESPOND ONLY WITH THE JSON ARRAY OF CONDITIONS. No markdown, no explanations.
"""
    
    return base_prompt

def validate_rule_structure(rule: Dict[str, Any]) -> bool:
    """Validate the rule structure meets requirements"""
    if not rule or "rules" not in rule:
        return False
    
    required_fields = ["dataSource", "field", "operator", "value", "connector"]
    
    for condition in rule["rules"]:
        # Check required fields
        if not all(field in condition for field in required_fields):
            return False
            
        # Validate data source exists
        if condition["dataSource"] not in CSV_STRUCTURES:
            return False
            
        # Validate field exists in data source
        if condition["field"] not in CSV_STRUCTURES[condition["dataSource"]]:
            return False
            
        # Validate operator
        valid_operators = ["=", ">", "<", ">=", "<=", "!=", "contains"]
        if condition["operator"] not in valid_operators:
            return False
            
        # Validate connector
        if condition["connector"] not in ["AND", "OR", None]:
            return False
            
    # Last condition must have null connector
    if rule["rules"][-1]["connector"] is not None:
        return False
        
    return True

def fix_rule_structure(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Fix common issues in rule structure"""
    if not rule or "rules" not in rule:
        return {"rules": []}
    
    # Ensure last condition has null connector
    if rule["rules"] and rule["rules"][-1].get("connector") is not None:
        rule["rules"][-1]["connector"] = None
        
    # Add missing fields with defaults
    for condition in rule["rules"]:
        condition.setdefault("eligibilityPeriod", "N/A")
        condition.setdefault("function", "N/A")
        condition.setdefault("ruleType", "condition")
        condition.setdefault("priority", None)
        condition.setdefault("id", f"cond_{hash(json.dumps(condition))}")
        
    return rule

def generate_rule_openai(user_input: str, modification_request: Optional[str] = None) -> Dict[str, Any]:
    """Use OpenAI to generate a rule based on user input"""
    if not openai.api_key:
        st.error("OpenAI API key is missing")
        return None
        
    prompt = generate_prompt_guidance(user_input, modification_request)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise financial rule generator. Output ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        response_content = response.choices[0].message.content
        
        # Extract JSON from response
        try:
            # First try parsing as full JSON
            rule_data = json.loads(response_content)
            if "rules" not in rule_data:
                # Wrap in rules object if needed
                rule_data = {"rules": rule_data}
        except:
            # Extract JSON substring
            match = re.search(r'\[.*\]', response_content, re.DOTALL)
            if match:
                rule_data = {"rules": json.loads(match.group())}
            else:
                st.error("Failed to extract JSON from response")
                return None
                
        # Validate and fix structure
        if not validate_rule_structure(rule_data):
            st.warning("Rule structure validation failed, attempting to fix...")
            rule_data = fix_rule_structure(rule_data)
            
        return rule_data
    
    except Exception as e:
        st.error(f"OpenAI error: {str(e)}")
        return None

def generate_rule_llama(user_input: str, modification_request: Optional[str] = None) -> Dict[str, Any]:
    """Use Groq/Llama to generate a rule based on user input"""
    if not hasattr(groq_client, 'chat') or not st.secrets.get("GROQ_API_KEY"):
        st.error("Groq API is not configured properly")
        return None
        
    prompt = generate_prompt_guidance(user_input, modification_request)
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise financial rule generator. Output ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        response_content = chat_completion.choices[0].message.content
        
        # Extract JSON from response
        try:
            rule_data = json.loads(response_content)
            if "rules" not in rule_data:
                rule_data = {"rules": rule_data}
        except:
            match = re.search(r'\[.*\]', response_content, re.DOTALL)
            if match:
                rule_data = {"rules": json.loads(match.group())}
            else:
                st.error("Failed to extract JSON from response")
                return None
                
        # Validate and fix structure
        if not validate_rule_structure(rule_data):
            st.warning("Rule structure validation failed, attempting to fix...")
            rule_data = fix_rule_structure(rule_data)
            
        return rule_data
    
    except Exception as e:
        st.error(f"Llama error: {str(e)}")
        return None

def display_rule_ui(rule: Dict[str, Any]) -> None:
    """Display the rule in the UI with all required fields"""
    if not rule or "rules" not in rule or not rule["rules"]:
        st.warning("No valid rule generated yet")
        return
    
    st.subheader("Rule Conditions")
    st.markdown("Define the logical conditions for this rule to apply.")
    
    # Main rule display
    for i, rule_item in enumerate(rule["rules"]):
        with st.expander(f"Condition {i+1}", expanded=True):
            cols = st.columns(7)
            with cols[0]:
                selected_ds = st.selectbox(
                    "Data Source",
                    options=list(CSV_STRUCTURES.keys()),
                    index=list(CSV_STRUCTURES.keys()).index(rule_item["dataSource"]) 
                    if rule_item["dataSource"] in CSV_STRUCTURES else 0,
                    key=f"ds_{i}"
                )
            with cols[1]:
                columns = CSV_STRUCTURES.get(selected_ds, [])
                selected_field = st.selectbox(
                    "Field", 
                    options=columns,
                    index=columns.index(rule_item["field"]) 
                    if rule_item["field"] in columns else 0,
                    key=f"field_{i}"
                )
            with cols[2]:
                period_val = rule_item.get("eligibilityPeriod", "N/A")
                period_idx = 0 if period_val == "N/A" else 1
                st.selectbox("Eligibility Period", 
                            ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"],
                            index=period_idx,
                            key=f"period_{i}")
            with cols[3]:
                func_val = rule_item.get("function", "N/A")
                func_idx = 0 if func_val == "N/A" else ["sum", "count", "avg", "max", "min"].index(func_val) + 1
                st.selectbox("Function", 
                            ["N/A", "sum", "count", "avg", "max", "min"],
                            index=func_idx,
                            key=f"func_{i}")
            with cols[4]:
                operator_options = ["=", ">", "<", ">=", "<=", "!=", "contains"]
                operator_idx = operator_options.index(rule_item["operator"]) if rule_item["operator"] in operator_options else 0
                st.selectbox("Operator", 
                            operator_options,
                            index=operator_idx,
                            key=f"op_{i}")
            with cols[5]:
                st.text_input("Value", value=str(rule_item.get("value", "")), 
                            key=f"val_{i}")
            
            if i < len(rule["rules"]) - 1:
                with cols[6]:
                    connector_val = rule_item.get("connector", "AND")
                    connector_idx = 0 if connector_val == "AND" else 1
                    st.selectbox("Connector", 
                                ["AND", "OR"],
                                index=connector_idx,
                                key=f"conn_{i}")

def initialize_session_state(tab_prefix: str):
    """Initialize session state for specific tab"""
    prefix = f"{tab_prefix}_"
    
    if f"{prefix}messages" not in st.session_state:
        st.session_state[f"{prefix}messages"] = [
            {"role": "assistant", "content": "Hello! What mortgage rule criteria would you like to create?"}
        ]
    if f"{prefix}current_rule" not in st.session_state:
        st.session_state[f"{prefix}current_rule"] = None
    if f"{prefix}confirmed" not in st.session_state:
        st.session_state[f"{prefix}confirmed"] = False
    if f"{prefix}user_prompt" not in st.session_state:
        st.session_state[f"{prefix}user_prompt"] = ""
    if f"{prefix}awaiting_confirmation" not in st.session_state:
        st.session_state[f"{prefix}awaiting_confirmation"] = False
    if f"{prefix}awaiting_modification" not in st.session_state:
        st.session_state[f"{prefix}awaiting_modification"] = False

def display_chat_message(role: str, content: str):
    """Display a chat message in the UI"""
    with st.chat_message(role):
        st.markdown(clean_user_input(content)) if role == "user" else content

def handle_user_confirmation(tab_prefix: str, confirmation: bool):
    """Handle user confirmation or modification request"""
    prefix = f"{tab_prefix}_"
    
    if confirmation:
        st.session_state[f"{prefix}confirmed"] = True
        st.session_state[f"{prefix}awaiting_confirmation"] = False
        st.session_state[f"{prefix}messages"].append(
            {"role": "assistant", "content": "✅ Rule confirmed! Here's your final rule:"}
        )
    else:
        st.session_state[f"{prefix}awaiting_confirmation"] = False
        st.session_state[f"{prefix}awaiting_modification"] = True
        st.session_state[f"{prefix}messages"].append(
            {"role": "assistant", "content": "What changes would you like to make to the rule?"}
        )

def generate_new_rule(tab_prefix: str, model_type: str):
    """Generate a new rule based on current state"""
    prefix = f"{tab_prefix}_"
    modification_request = None
    
    if st.session_state[f"{prefix}awaiting_modification"] and st.session_state[f"{prefix}messages"][-1]["role"] == "user":
        modification_request = st.session_state[f"{prefix}messages"][-1]["content"]
    
    with st.spinner("🧠 Generating rule..."):
        if model_type == "openai":
            new_rule = generate_rule_openai(
                st.session_state[f"{prefix}user_prompt"],
                modification_request
            )
        else:
            new_rule = generate_rule_llama(
                st.session_state[f"{prefix}user_prompt"],
                modification_request
            )
        
        if new_rule and new_rule.get("rules"):
            st.session_state[f"{prefix}current_rule"] = new_rule
            rule_preview = json.dumps(new_rule, indent=2)
            st.session_state[f"{prefix}messages"].append({
                "role": "assistant",
                "content": f"I've generated this rule:\n\n```json\n{rule_preview}\n```\n\nDoes this meet your requirements?"
            })
            st.session_state[f"{prefix}awaiting_confirmation"] = True
            st.session_state[f"{prefix}awaiting_modification"] = False
        else:
            st.session_state[f"{prefix}messages"].append({
                "role": "assistant",
                "content": "⚠️ Couldn't generate a valid rule. Please try rephrasing your request or providing more details."
            })

def render_tab(tab_prefix: str, model_type: str):
    """Render UI for a specific model tab"""
    prefix = f"{tab_prefix}_"
    initialize_session_state(tab_prefix)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        rule = st.session_state.get(f"{prefix}current_rule")
        if rule:
            display_rule_ui(rule)
            
            if st.session_state.get(f"{prefix}confirmed"):
                st.success("✅ Final Rule Confirmed")
                st.json(rule)
                
                json_str = json.dumps(rule, indent=2)
                st.download_button(
                    label="Download Rule JSON",
                    data=json_str,
                    file_name=f"mortgage_rule_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                if st.button("Create New Rule", key=f"new_rule_{tab_prefix}"):
                    st.session_state[f"{prefix}messages"] = [
                        {"role": "assistant", "content": "Let's create a new rule. What criteria would you like?"}
                    ]
                    st.session_state[f"{prefix}current_rule"] = None
                    st.session_state[f"{prefix}confirmed"] = False
                    st.session_state[f"{prefix}user_prompt"] = ""
                    st.rerun()
    
    with col2:
        st.subheader("Rule Assistant")
        
        for message in st.session_state[f"{prefix}messages"]:
            display_chat_message(message["role"], message["content"])
        
        if prompt := st.chat_input("Type your message here...", key=f"chat_{tab_prefix}"):
            cleaned_prompt = clean_user_input(prompt)
            st.session_state[f"{prefix}messages"].append({"role": "user", "content": cleaned_prompt})
            display_chat_message("user", cleaned_prompt)
            
            user_prompt = st.session_state.get(f"{prefix}user_prompt", "")
            
            if not user_prompt:
                st.session_state[f"{prefix}user_prompt"] = cleaned_prompt
                generate_new_rule(tab_prefix, model_type)
                st.rerun()
            
            elif st.session_state.get(f"{prefix}awaiting_confirmation", False):
                if "yes" in cleaned_prompt.lower() or "correct" in cleaned_prompt.lower():
                    handle_user_confirmation(tab_prefix, True)
                else:
                    handle_user_confirmation(tab_prefix, False)
                st.rerun()
            
            elif st.session_state.get(f"{prefix}awaiting_modification", False):
                generate_new_rule(tab_prefix, model_type)
                st.rerun()
            
            else:
                st.session_state[f"{prefix}user_prompt"] = cleaned_prompt
                st.session_state[f"{prefix}current_rule"] = None
                st.session_state[f"{prefix}confirmed"] = False
                generate_new_rule(tab_prefix, model_type)
                st.rerun()

def main():
    st.set_page_config(page_title="Mortgage Rule Generator", layout="wide")
    st.title("🏦 Smart Mortgage Rule Generator")
    
    # Custom CSS
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 25px;
            border-radius: 8px 8px 0 0;
            background-color: #f0f2f6;
            margin: 0 5px;
            transition: all 0.3s;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e0e5ec;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4a8cff;
            color: white !important;
        }
        .stChatFloatingInputContainer {
            bottom: 20px;
        }
        .stChatMessage {
            padding: 12px 15px;
            border-radius: 12px;
            margin-bottom: 15px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        [data-testid="stExpander"] .streamlit-expanderHeader {
            font-size: 16px;
            font-weight: 600;
            background-color: #f9fafb;
            border-bottom: 1px solid #eee;
        }
        .stButton>button {
            background-color: #4a8cff;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
        }
        .stDownloadButton>button {
            background-color: #28a745;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
        }
    </style>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["OpenAI GPT-4o", "Llama 3 70B"])
    
    with tab1:
        if not openai.api_key:
            st.warning("🔑 OpenAI API key is missing. Add it to your Streamlit secrets.")
        else:
            st.success("✅ OpenAI GPT-4o connected")
        render_tab("openai", "openai")
    
    with tab2:
        if not hasattr(groq_client, 'chat') or not st.secrets.get("GROQ_API_KEY"):
            st.warning("🔑 Groq API key is missing. Add it to your Streamlit secrets.")
        else:
            st.success("✅ Llama 3 70B connected")
        render_tab("llama", "llama")

if __name__ == "__main__":
    main()
