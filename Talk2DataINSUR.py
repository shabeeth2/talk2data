# -*- coding: utf-8 -*-
"""
Enhanced Talk2DataINSUR.py - Optimized for Performance and User Experience
Created on Sat Nov 16 20:08:52 2024
@author: ravivarman.balaiyan
"""

import streamlit as st
import pandas as pd
import re
import time
import os
import json # Added import
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from agents.coderAgent import get_code_response

# Load environment variables
load_dotenv()

# Configure page settings (must be first Streamlit command)
st.set_page_config(
    page_title="AIWA - Talk to Insurance Data",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Google AI client
@st.cache_resource
def initialize_genai_client():
    """Initialize and cache the Google AI client"""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API Key not found. Please check your .env file.")
        st.stop()
    return genai.Client(api_key=google_api_key)

# Initialize database connection with caching
@st.cache_resource
def initialize_db_connection():
    """Initialize and cache database connection"""
    try:
        # Use DATABASE_URI env var to avoid conflicts among multiple SQLite files
        db_uri = os.getenv("DATABASE_URI", "sqlite:///./data/newSynthetic70k.db")
        db = SQLDatabase.from_uri(db_uri)
        return db
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.stop()

@st.cache_data
def get_schema_db(_db):
    """Get and cache database schema"""
    return _db.get_table_info()

@st.cache_data
def build_few_shots_prompt(_db, chat_history=None):
    """Build and cache the few-shot prompt for SQL generation"""
    db_schema = get_schema_db(_db)
    
    few_shots = [
        {
            "input": "How many Customers are present?",
            "query": "SELECT COUNT(*) FROM customer;"
        },
        {
            "input": "List 10 claims with the highest claim amount",
            "query": "SELECT claim_id, claim_amount FROM claim ORDER BY claim_amount DESC LIMIT 10;"
        },
        {
            "input": "List the top 3 agents with highest commission",
            "query": "SELECT a.first_name, a.last_name, b.commission_amount FROM commission b LEFT JOIN agent a ON a.agent_id = b.agent_id ORDER BY b.commission_amount DESC LIMIT 3;"
        },
        {
            "input": "which agent sold highest number of policies yesterday",
            "query": "SELECT a.agent_id, a.first_name || ' ' || a.last_name AS agent_name, COUNT(s.sale_id) AS policies_sold FROM sales s JOIN agent a ON s.agent_id = a.agent_id WHERE DATE(s.sale_date) = DATE('now', '-1 day') GROUP BY a.agent_id, a.first_name, a.last_name ORDER BY policies_sold DESC LIMIT 1;"
        },
        {
            "input": "Show total premium collected by policy type",
            "query": "SELECT policy_type, SUM(premium_amount) as total_premium FROM policy GROUP BY policy_type;"
        },
        {
            "input": "Monthly premium revenue trends",
            "query": "SELECT strftime('%Y-%m', sale_date) as month, SUM(premium_amount) as monthly_revenue FROM sales s JOIN policy p ON s.policy_id = p.policy_id GROUP BY month ORDER BY month;"
        },
        {
            "input": "Top 5 customers by total premium paid",
            "query": "SELECT c.first_name || ' ' || c.last_name as customer_name, SUM(p.premium_amount) as total_premium FROM customer c JOIN policy p ON c.customer_id = p.customer_id GROUP BY c.customer_id ORDER BY total_premium DESC LIMIT 5;"
        },
        {
            "input": "Compare approved vs rejected claim amounts",
            "query": "SELECT claim_status, COUNT(*) as claim_count, SUM(claim_amount) as total_amount FROM claim GROUP BY claim_status;"
        }
    ]

    chat_context = f"\\nPrevious conversation:\\n{chat_history}" if chat_history else ""
    
    prompt_base = f"""
You are a SQL expert and intelligent assistant. Generate accurate, efficient SQL queries based on the database schema.

Database Schema: {db_schema}{chat_context}

Guidelines:
1. Analyze the request to determine exact data requirements
2. Use only provided schema (tables, columns, relationships)
3. Generate efficient queries with proper JOINs and indexing
4. Handle edge cases (null values, empty results)
5. Return ONLY executable SQL - no formatting, no explanations
6. Remove ``` and 'sql' keywords from output
7. ONE query only, no multiple queries
8. No DML statements (INSERT, UPDATE, DELETE, DROP)

Examples:
"""
    
    for example in few_shots:
        prompt_base += f"\\nInput: {example['input']}\\nSQL: {example['query']}\\n"
    
    return prompt_base

# Agent Definitions
def planner_agent(client, user_question, chat_history, db_schema):
    """
    Analyzes the user query, infers the analytical goal, and decides the steps.
    Returns a JSON string outlining the plan.
    """
    prompt = f"""
You are a planner agent for a multi-agent system that answers questions based on a SQL database.
Your goal is to analyze the user's query and the conversation history to determine the necessary steps.

Database Schema:
{db_schema}

User Question: "{user_question}"

Conversation History:
{chat_history}

Based on the user question, conversation history, and schema, decide:
1.  `sql_needed`: boolean - Is a SQL query required to answer this question?
2.  `sql_queries_count`: integer - How many SQL queries are likely needed? (e.g., 1 for a direct question, 2 or more for comparisons). If sql_needed is false, this should be 0.
3.  `chart_needed`: boolean - Is a chart likely to be helpful to visualize the answer? This should be true if the user asks for a chart, or if the data is suitable for visualization (e.g., trends, comparisons, distributions).

Output your decision as a JSON object with the keys "sql_needed", "sql_queries_count", and "chart_needed".
Example:
{{"sql_needed": true, "sql_queries_count": 1, "chart_needed": true}}
"""
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        # Ensure the response is a valid JSON string
        plan_str = response.text.strip()
        # Basic cleaning if the model wraps with markdown
        if plan_str.startswith("```json"):
            plan_str = plan_str[7:]
        if plan_str.endswith("```"):
            plan_str = plan_str[:-3]
        
        # Validate and parse JSON
        json.loads(plan_str) # Will raise error if not valid JSON
        return plan_str
    except Exception as e:
        st.error(f"Planner Agent Error: {str(e)}")
        # Fallback plan
        return json.dumps({"sql_needed": True, "sql_queries_count": 1, "chart_needed": False})


def sql_generator_agent(client, db, user_question, chat_history):
    """Generates and executes SQL query."""
    few_shots_prompt = build_few_shots_prompt(db, chat_history)
    sql_query = generate_sql_query(client, few_shots_prompt, user_question)
    
    if sql_query.strip().upper() == "I DON'T KNOW":
        return "I don't know", "No result found in database"

    st.code(sql_query, language="sql")
    query_results = run_query(db, sql_query)
    return sql_query, query_results

# visualization_generator_agent is the existing process_visualization function

def response_composer_agent(client, sql_query_results, user_question, plan):
    """Composes the final textual response."""
    # If SQL was not needed, the query_results might be a placeholder or direct answer
    if not plan.get("sql_needed") and isinstance(sql_query_results, str):
         # If no SQL was run, and we have a string, it might be a direct answer or an error.
         # For now, let's assume it's a message to be displayed.
         # A more sophisticated approach would be needed if the planner could decide to skip SQL
         # and directly compose an answer based on the query alone.
        if sql_query_results == "No result found in database" or sql_query_results == "I don't know": # Default from sql_generator_agent
            return "I couldn't find an answer to your question based on the available data."
        return sql_query_results


    return generate_final_answer(client, str(sql_query_results), user_question)


def generate_sql_query(client, prompt, user_question):
    """Generate SQL query using Gemini AI"""
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"{prompt}\n\nUser Question: {user_question}"
        )
        
        sql_query = response.text.strip()
        # Clean the SQL query
        sql_query = re.sub(r'```sql\n|```\n|```', '', sql_query)
        sql_query = re.sub(r'^sql\s*', '', sql_query, flags=re.IGNORECASE)
        
        return sql_query.strip()
    except Exception as e:
        st.error(f"Error generating SQL query: {str(e)}")
        return "I don't know"

def run_query(db, sql_query):
    """Execute SQL query with error handling"""
    try:
        if sql_query == "I don't know":
            return "No result found in database"
        
        result = db.run(sql_query)
        return result if result else "No result found in database"
    except Exception as e:
        st.error(f"Query execution error: {str(e)}")
        return "No result found in database"

def generate_final_answer(client, sql_response, user_question):
    """Generate final answer based on SQL results"""
    try:
        prompt = f"""
Based on the SQL response, provide a clear, user-friendly answer to: "{user_question}"

SQL Response: {sql_response}

Guidelines:
- Be conversational and informative
- Don't show error messages to user
- If empty results, provide polite message
- Highlight key insights from the data
- Be concise but comprehensive
"""
        
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return "I apologize, but I encountered an issue processing your request. Please try again."

def render_sidebar():
    """Render sidebar with controls and information"""
    with st.sidebar:
        st.title("🏢 Insurance Data Analytics")
        
        # Visualization toggle
        viz_enabled = st.toggle(
            "📊 Enable Visualizations",
            value=st.session_state.get('visualization_enabled', True),
            help="Toggle to enable/disable automatic chart generation"
        )
        st.session_state.visualization_enabled = viz_enabled
        
        # Database info
        st.subheader("📋 Database Information")
        st.info("""
        **Tables Available:**
        - Customer, Agent, Policy
        - Sales, Claims, Commission
        - Address, Quote
        
        **Sample Questions:**
        - "Show monthly premium trends"
        - "Top performing agents"
        - "Claims analysis by status"
        """)
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

def process_visualization(query_results, user_question):
    """Process and display visualization with improved data handling"""
    if not st.session_state.get('visualization_enabled', True):
        return "Visualization disabled."
    
    try:
        with st.spinner("🎨 Creating visualization..."):
            # Debug: Show what we received
            with st.expander("🔍 Debug: Query Results", expanded=False):
                st.write(f"Type: {type(query_results)}")
                st.write(f"Content: {query_results}")
            
            # Enhanced data conversion logic
            df = None
            
            # Handle different query result formats
            if query_results == "No result found in database" or not query_results:
                st.warning("⚠️ No data available for visualization")
                return
            
            # Try to convert to DataFrame based on data type
            if isinstance(query_results, str):
                # If it's a string that looks like data, try to parse it
                if query_results.startswith('[') or query_results.startswith('('):
                    try:
                        # Try to evaluate as Python literal
                        import ast
                        parsed_data = ast.literal_eval(query_results)
                        df = pd.DataFrame(parsed_data)
                    except:
                        st.warning("⚠️ Cannot parse string data for visualization")
                        return
                else:
                    st.warning("⚠️ Query result is text, not tabular data")
                    return
                    
            elif isinstance(query_results, (list, tuple)):
                # Handle list of tuples/lists (common SQL result format)
                if len(query_results) > 0:
                    if isinstance(query_results[0], (list, tuple)):
                        # List of rows - need to infer column names
                        df = pd.DataFrame(query_results)
                    else:
                        # List of values - single column
                        df = pd.DataFrame({'value': query_results})
                else:
                    st.warning("⚠️ Empty result set")
                    return
                    
            elif isinstance(query_results, dict):
                # Dictionary format
                df = pd.DataFrame(query_results)
                
            elif isinstance(query_results, pd.DataFrame):
                # Already a DataFrame
                df = query_results
                
            else:
                # Try generic conversion
                try:
                    df = pd.DataFrame(query_results)
                except Exception as e:
                    st.warning(f"⚠️ Cannot convert data to DataFrame: {str(e)}")
                    return
            
            # Validate DataFrame
            if df is None or df.empty:
                st.warning("⚠️ No tabular data available for visualization")
                return
            
            # Show DataFrame info for debugging
            with st.expander("📊 Data Preview", expanded=False):
                st.write(f"Shape: {df.shape}")
                st.write("Columns:", df.columns.tolist())
                st.dataframe(df.head())
            # Direct bar chart fallback for simple two-column data
            if df.shape[1] == 2:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], labels={df.columns[0]: df.columns[0], df.columns[1]: df.columns[1]})
                st.plotly_chart(fig, use_container_width=True)
                return  # Skip AI-generated code
            
            # Generate and execute visualization code
            visualization_code = get_code_response(query_results, user_question)
            
            # Create visualization container
            with st.container():
                # Execute visualization code safely with fallback for missing column names
                local_vars = {
                    'df': df,
                    'data_frame': df,  # Alias for visualization code expecting data_frame
                    'query_results': query_results,
                    'st': st,
                    'px': px,
                    'plt': plt,
                    'pd': pd,
                    'go': go if 'go' in globals() else None
                }
                try:
                    exec(visualization_code, globals(), local_vars)
                except Exception as e:
                    err = str(e).lower()
                    # Fallback generic bar chart for column name mismatches
                    if "not the name" in err or "keyerror" in err:
                        fig = px.bar(df, x=df.columns[0], y=df.columns[1], labels={df.columns[0]: df.columns[0], df.columns[1]: df.columns[1]})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        raise
                # Adjust matplotlib figure size if used
                try:
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.set_size_inches(10, 6)
                        plt.tight_layout()
                except:
                    pass
                
                return visualization_code
                        
    except Exception as e:
        st.error(f"❌ Visualization error: {str(e)}")
        with st.expander("🔍 Debug Information"):
            st.code(visualization_code if 'visualization_code' in locals() else "No code generated")
            st.write("Query Results Type:", type(query_results))
            st.write("Query Results:", str(query_results)[:500] + "..." if len(str(query_results)) > 500 else str(query_results))

    return None

def main():
    """Main application function"""
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "visualization_enabled" not in st.session_state:
        st.session_state.visualization_enabled = True

    # Initialize resources
    client = initialize_genai_client()
    db = initialize_db_connection()
    db_schema = get_schema_db(db)
    
    # Render sidebar
    render_sidebar()
    
    # Main header
    st.title("🤖 AIWA - Conversational Insurance Data Insights")
    st.markdown("Ask questions about your insurance database in natural language!")
    
    # Display chat history (only render chart for latest assistant message to improve performance)
    history = st.session_state.chat_history
    for idx, message in enumerate(history):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
                # Only re-render chart for the most recent assistant message
                if idx == len(history) - 1 and message.get("chart_data") and message.get("chart_question"):
                    process_visualization(message["chart_data"], message["chart_question"])
            else:
                st.write(message["content"])
    
    # Chat input
    user_question = st.chat_input(
        "💬 Ask about policies, customers, agents, claims, sales...",
        key="user_input"
    )
    
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)
        
        with st.chat_message("assistant"):
            assistant_response_content = ""
            query_results_for_viz = None # Store results for visualization
            viz_code = None # Store visualization code

            with st.spinner("🤔 Thinking..."):
                # 1. Planner Agent
                chat_context_for_planner = "\\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in st.session_state.chat_history[-5:]
                ])
                plan_str = planner_agent(client, user_question, chat_context_for_planner, db_schema)
                try:
                    plan = json.loads(plan_str)
                except json.JSONDecodeError:
                    st.error("Planner output was not valid JSON. Using default plan.")
                    plan = {"sql_needed": True, "sql_queries_count": 1, "chart_needed": False}
                
                with st.status("Thinking.", expanded=True):
                    st.write("📋 **Execution Plan:**")
                    st.json(plan)
                    status_sql.update(label="✅ Plan", state="complete", expanded=False)

                sql_query = "No SQL query generated."
                query_results = "No query run."

                if plan.get("sql_needed", False):
                    with st.status("⚙️ Processing SQL...", expanded=True) as status_sql:
                        # For simplicity, handling one SQL query if multiple are planned.
                        # A loop or more complex logic would be needed for `sql_queries_count > 1`.
                        sql_query, query_results = sql_generator_agent(client, db, user_question, chat_context_for_planner)
                        query_results_for_viz = query_results # Save for potential visualization
                        status_sql.update(label="✅ SQL Processing Complete", state="complete", expanded=False)
                else:
                    query_results_for_viz = "No data from SQL for visualization as SQL was not needed."


                # 3. Visualization Generator Agent (conditionally)
                if plan.get("chart_needed", False) and st.session_state.visualization_enabled:
                    if query_results_for_viz and query_results_for_viz not in ["No result found in database", "I don't know", "No data from SQL for visualization as SQL was not needed."]:
                        viz_code = process_visualization(query_results_for_viz, user_question)
                    elif query_results_for_viz == "No data from SQL for visualization as SQL was not needed.":
                         st.info("Chart was planned, but no SQL was executed to fetch data for it.")
                    else:
                        st.info("Chart was planned, but no data was returned from the database to visualize.")
                elif plan.get("chart_needed", False) and not st.session_state.visualization_enabled:
                    st.info("Chart generation is disabled in the sidebar.")

                # 4. Response Composer Agent
                with st.spinner("✍️ Composing response..."):
                    final_answer_text = response_composer_agent(client, query_results, user_question, plan)
                
                # Display final answer with typing effect
                response_placeholder = st.empty()
                full_response_text = ""
                for chunk in final_answer_text.split():
                    full_response_text += chunk + " "
                    time.sleep(0.03)
                    response_placeholder.markdown(full_response_text + "▌")
                response_placeholder.markdown(full_response_text)
                
                assistant_response_content = full_response_text
            
            # Append assistant message and persist chart data for history
            msg = {"role": "assistant", "content": assistant_response_content}
            # store chart context if generated
            if plan.get("chart_needed", False) and st.session_state.visualization_enabled and viz_code:
                msg["chart_data"] = query_results_for_viz
                msg["chart_question"] = user_question
            st.session_state.chat_history.append(msg)

if __name__ == "__main__":
    main()
