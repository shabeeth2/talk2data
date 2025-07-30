# -*- coding: utf-8 -*-
"""
AIWA - Talk 2 Data
Author: ravivarman.balaiyan
"""

import streamlit as st
import pandas as pd
import re
import os
import json 
import matplotlib.pyplot as plt
import plotly.express as px
from google import genai
from agents.coderAgent import get_code_response
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
import re
from datetime import datetime, timedelta
import networkx as nx
from wordcloud import WordCloud


st.set_page_config(
    page_title="AIWA - Talk 2 Data",
    page_icon="🏢",
    layout="wide"
)

# Initialize services
@st.cache_resource
def initialize_services():
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("Google API Key not found. Please check your .env file.")
            st.stop()
        
        try:
            client = genai.Client(api_key=google_api_key)
        except Exception as e:
            st.error(f"Failed to initialize Google AI client: {str(e)}")
            st.stop()
        try:
            db_path = os.path.abspath("./data/newSynthetic70k.db")
            if not os.path.exists(db_path):
                st.error(f"Database file not found at: {db_path}")
                st.stop()
            
            db = CustomDatabase(db_path)
            db_schema = db.get_schema_info()
            
            # Test the con
            test_result = db.run_query("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            if not test_result:
                raise Exception("Database appears empty or inaccessible")
                
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            st.stop()
        
        return client, db, db_schema
        
    except Exception as e:
        st.error(f"Service initialization failed: {str(e)}")
        return None, None, None

class CustomDatabase:
    
    def __init__(self, db_path):
        self.db_path = db_path
        
    def run_query(self, query):
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                conn.close()
                return str(results) if results else "No results found"
            else:
                conn.commit()
                conn.close()
                return "Query executed "
                
        except Exception as e:
            return f"Query error: {str(e)}"
    
    def run(self, query):
        return self.run_query(query)
        
    def get_schema_info(self):
        return """
Table 'addresses' has columns: address_id (TEXT), street (TEXT), city (TEXT), county (TEXT), postcode (TEXT), country (TEXT)
Table 'agents' has columns: agent_id (TEXT), agent_name (TEXT), email (TEXT), phone (TEXT), hire_date (TEXT), commission_rate (REAL)
Table 'claims' has columns: claim_id (TEXT), policy_id (TEXT), claim_date (TEXT), claim_amount (REAL), claim_status (TEXT), approved_amount (REAL)
Table 'commissions' has columns: commission_id (TEXT), agent_id (TEXT), policy_id (TEXT), commission_amount (REAL), paid_date (TEXT)
Table 'customers' has columns: cust_id (TEXT), name (TEXT), email (TEXT), phone (TEXT), address_id (TEXT), agent_id (TEXT), joined_date (TEXT), status (TEXT)
Table 'policies' has columns: policy_id (TEXT), customer_id (TEXT), policy_type (TEXT), start_date (TEXT), end_date (TEXT), premium_amount (REAL), status (TEXT)
Table 'prospects' has columns: prospect_id (TEXT), name (TEXT), email (TEXT), phone (TEXT), created_at (TEXT), status (TEXT)
Table 'quotes' has columns: quote_id (TEXT), prospect_id (TEXT), quote_date (TEXT), premium_amount (REAL), valid_till (TEXT), status (TEXT)
Table 'sales' has columns: sale_id (TEXT), policy_id (TEXT), agent_id (TEXT), customer_id (TEXT), sale_date (TEXT), premium_amount (REAL), commission_amount (REAL), region (TEXT)
Table 'customer_demographics' has columns: demo_id (TEXT), customer_id (TEXT), age (INTEGER), gender (TEXT), income_bracket (TEXT), occupation_category (TEXT), family_size (INTEGER), health_score (INTEGER)

Available views:
- monthly_sales_summary: Time series data for sales trends
- policy_type_performance: Life vs Health insurance comparison
- agent_performance_by_type: Agent performance metrics
- regional_performance: Geographic sales analysis
"""

FEW_SHOT_EXAMPLES = [
    {
        "input": "How many Customers are present?",
        "query": "SELECT COUNT(*) FROM customers;"
    },
    {
        "input": "List 10 claims with the highest claim amount",
        "query": "SELECT claim_id, claim_amount FROM claims ORDER BY claim_amount DESC LIMIT 10;"
    },
    {
        "input": "List the top 3 agents with highest commission",
        "query": "SELECT a.agent_name, c.commission_amount FROM commissions c LEFT JOIN agents a ON a.agent_id = c.agent_id ORDER BY c.commission_amount DESC LIMIT 3;"
    },
    {
        "input": "Monthly premium revenue trends",
        "query": "SELECT strftime('%Y-%m', sale_date) as month, SUM(premium_amount) as monthly_revenue FROM sales GROUP BY month ORDER BY month;"
    }
]

def make_plan(client, user_question, db_schema):
    """Simple planner: decides if SQL and chart are needed"""
    prompt = f"""
Analyze this user question about insurance data: "{user_question}"

Database has tables: customers, agents, policies, sales, claims, commissions, addresses, quotes

Decide:
1. sql_needed: true/false - Does this need database query?
2. chart_needed: true/false - Would a chart help visualize the answer?

Return only JSON: {{"sql_needed": boolean, "chart_needed": boolean}}
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        plan_text = response.text.strip() 
        if plan_text.startswith("```json"):
            plan_text = plan_text[7:-3]
        return json.loads(plan_text)
    except:
        return {"sql_needed": True, "chart_needed": False}

def generate_sql(client, user_question, db_schema):
    """Generate SQL query from user question"""
    examples_text = "\n".join([
        f"Question: {ex['input']}\nSQL: {ex['query']}\n"
        for ex in FEW_SHOT_EXAMPLES
    ])
    
    prompt = f"""
Database Schema: {db_schema}

Examples:
{examples_text}

Generate SQL for: "{user_question}"
IMPORTANT: Only use tables listed in the schema.
Return only the SQL query, no formatting.
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        sql_query = response.text.strip()
        sql_query = re.sub(r'```sql\n|```\n|```', '', sql_query)
        return sql_query.strip()
    except Exception as e:
        st.error(f"SQL generation error: {str(e)}")
        return None
# def generate_sql_query(client, prompt, user_question):
#     """Generate SQL query using Gemini AI"""
#     try:
#         response = client.models.generate_content(
#             model="gemini-2.5-flash-lite-preview-06-17",
#             contents=f"{prompt}\n\nUser Question: {user_question}"
#         )
        
#         sql_query = response.text.strip()
#         # Clean the SQL query
#         sql_query = re.sub(r'```sql\n|```\n|```', '', sql_query)
#         sql_query = re.sub(r'^sql\s*', '', sql_query, flags=re.IGNORECASE)
        
#         return sql_query.strip()
#     except Exception as e:
#         st.error(f"Error generating SQL query: {str(e)}")
#         return "I don't know"

# def run_query(db, sql_query):
#     """Execute SQL query with error handling"""
#     try:
#         if sql_query == "I don't know":
#             return "No result found in database"
        
#         result = db.run(sql_query)
#         return result if result else "No result found in database"
#     except Exception as e:
#         st.error(f"Query execution error: {str(e)}")
#         return "No result found in database"
@st.cache_data
def execute_sql(sql_query):
    """Execute SQL query with caching using custom database handler"""
    if not sql_query:
        return "No query to execute"
    
    try:
        db_path = os.path.abspath("./data/newSynthetic70k.db")
        db = CustomDatabase(db_path)
        result = db.run(sql_query)
        return result if result else "No results found"
    except Exception as e:
        st.error(f"Query execution error: {str(e)}")
        return "Query failed"

def parse_results_to_df(results_str):   
    try:
        import ast
        data = ast.literal_eval(results_str)
        df = pd.DataFrame(data)
        return df
    except (ValueError, SyntaxError, TypeError) as e:
        st.warning(f"Could not parse SQL results into a DataFrame: {e}")
        return pd.DataFrame() # Return empty DataFrame on failure

def create_chart(query_results, user_question):
    try:
        if isinstance(query_results, str):
            df = parse_results_to_df(query_results)
        else:
            df = pd.DataFrame(query_results)

        if df.empty:
            st.warning("The query returned no data to visualize.")
            return
        viz_code = get_code_response(df, user_question)
        
        # Import create_network_traces function from coderAgent
        from agents.coderAgent import create_network_traces
        
        local_vars = {
            'df': df, 'st': st, 'px': px, 'plt': plt, 'pd': pd, 'go': go,
            'ff': ff, 'np': np, 'nx': nx, 'WordCloud': WordCloud,
            'make_subplots': make_subplots, 'create_network_traces': create_network_traces
        }
        
        exec(viz_code, globals(), local_vars)
        
    except Exception as e:
        st.error(f"Enhanced visualization error: {str(e)}")
        st.code(viz_code, language='python')    
        # Smart fallback based on data characteristics
        # create_intelligent_fallback(df if 'df' in locals() else None, user_question)

def generate_response(client, query_results, user_question):
    """Generate final natural language response"""
    prompt = f"""
User asked: "{user_question}"
Database returned: {query_results}

Provide a clear, helpful answer. If no data, explain politely.
Be conversational and highlight key insights.
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except:
        return "I apologize, but I encountered an issue processing your request."

def main():
    """Main application"""
    # Init session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    client, db, db_schema = initialize_services()
    
    st.title("🤖 AIWA - Insurance Data Assistant")
    st.markdown("Ask questions about your insurance database in natural language!")
    
    with st.sidebar:
        st.title("🏢 Database Info")
        st.info("""
        **Available Tables:**
        - Customer, Agent, Policy
        - Sales, Claims, Commission
        - Address, Quote
        """)
        
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Show chart if available
            if message.get("chart_data") and message.get("show_chart"):
                create_chart(message["chart_data"], message.get("question", ""))
    
    # Chat input
    user_question = st.chat_input("Ask about policies, customers, agents, claims...")
    
    if user_question:

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.chat_message("user"):
            st.write(user_question)
        
        with st.chat_message("assistant"):
            # Step 1: Make plan
            with st.spinner("Thinking..."):
                plan = make_plan(client, user_question, db_schema)
                #t.write(f"📋 Plan: SQL needed: {plan['sql_needed']}, Chart needed: {plan['chart_needed']}")
            
            # Step 2: Execute SQL if needed
            query_results = None
            if plan["sql_needed"]:
                with st.spinner("Generating and executing SQL..."):
                    sql_query = generate_sql(client, user_question, db_schema)
                    if sql_query:
                        st.code(sql_query, language="sql")
                        query_results = execute_sql(sql_query)
            
            # Step 3: Create chart if needed
            if plan["chart_needed"] and query_results:
                with st.spinner("Creating visualization..."):
                    try:
                        create_chart(query_results, user_question)
                    except Exception as e:
                        st.error(f"Chart creation error: {str(e)}")

            
            # Step 4: Generate response
            with st.spinner("Generating response..."):
                if query_results:
                    response = generate_response(client, query_results, user_question)
                else:
                    response = "I can help you with questions about the insurance database. Please try asking about customers, policies, agents, or claims."
                
                st.write(response)
        
        # history
        assistant_msg = {
            "role": "assistant", 
            "content": response,
            "question": user_question
        }
        if query_results and plan["chart_needed"]:
            assistant_msg["chart_data"] = query_results
            assistant_msg["show_chart"] = True
        
        st.session_state.chat_history.append(assistant_msg)

if __name__ == "__main__":
    main()
