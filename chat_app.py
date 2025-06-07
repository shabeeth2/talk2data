"""
Minimalist Conversational Analytics Platform
"""
import streamlit as st
import json
import pandas as pd
from crewai import Agent, Task, Crew, LLM
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

load_dotenv()

# Set environment variables for CrewAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy-key")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = LLM(model="gemini/gemini-1.5-flash", api_key=google_api_key)

class SimplePlatform:
    """Minimalist analytics platform with chat interface"""
    
    def __init__(self):
        self.db_path = "./data/newSynthetic70k.db"
        self.agents = self._create_agents()
    
    def _create_agents(self):
        """Create minimal set of agents following exact specifications"""
        planner = Agent(
            role="Query Planner", 
            goal="Determine the minimal set of tools (query, visualization) and steps required to answer the user's query. Avoid speculative or fallback planning.",
            backstory="""Expert at identifying the EXACT tools needed:
            - If query needs data: specify SQL tool and required tables
            - If query needs visualization: specify chart type and data requirements  
            - If query is simple lookup: specify direct response
            
            Database Schema:
            - customers (cust_id, name, email, phone, address_id, agent_id, joined_date, status)
            - policies (policy_id, customer_id, policy_type, start_date, end_date, premium_amount, status)
            - claims (claim_id, policy_id, claim_date, claim_amount, claim_status, approved_amount)
            - agents (agent_id, agent_name, email, phone, hire_date, commission_rate)
            
            Return JSON with ONLY required tools: {"tools": ["sql", "chart"], "chart_type": "bar", "tables": ["customers"]}""",
            llm=llm,
            verbose=False
        )
        
        executor = Agent(
            role="Task Executor",
            goal="Perform the exact tasks identified by the planner using only the required tools. No parallel calls, retries, or speculative execution.",
            backstory="""Execute ONLY the specific tasks planned. Use exact database schema:
            - customers.cust_id = policies.customer_id
            - policies.policy_id = claims.policy_id
            
            Rules:
            1. Execute ONLY what planner specified
            2. Return ONLY SQL query, no explanations
            3. Use exact column names from schema
            4. No fallbacks or alternative approaches""",
            llm=llm,
            verbose=False
        )
        
        composer = Agent(
            role="Response Composer",
            goal="Create direct answers using actual database query results. Always use the specific numbers and data provided to answer the user's question exactly.",
            backstory="""Expert at converting database query results into clear, direct answers. 
            
            Rules:
            1. ALWAYS use the actual data values provided in the query results
            2. If given "Query result: 15247", answer "There are 15,247 claims"
            3. If given multiple records, summarize the key findings
            4. Be specific with numbers - no vague responses
            5. Answer the exact question asked using the real data
            6. No hedging, disclaimers, or suggestions to run queries""",
            llm=llm,
            verbose=False
        )
        
        return {"planner": planner, "executor": executor, "composer": composer}
    
    def _execute_sql(self, query: str) -> pd.DataFrame:
        """Execute SQL query against database - no fallbacks"""
        try:
            # Common column name fixes
            query = query.replace("customer_id", "cust_id").replace("customer_name", "name")
            query = query.replace("c.customer_id", "c.cust_id").replace("c.customer_name", "c.name")
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"SQL execution failed: {str(e)}")
            return pd.DataFrame()

    def _determine_chart_type(self, user_input: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Determine appropriate chart type based on query and data"""
        user_lower = user_input.lower()
        cols = len(df.columns)
        
        # Chart type detection logic
        if any(word in user_lower for word in ['trend', 'over time', 'monthly', 'yearly', 'by month', 'by year']):
            return {"type": "line", "title": "Trend Analysis"}
        elif any(word in user_lower for word in ['distribution', 'breakdown', 'percentage', 'share']):
            return {"type": "pie", "title": "Distribution Analysis"}
        elif any(word in user_lower for word in ['top', 'ranking', 'highest', 'lowest', 'best', 'worst']):
            return {"type": "bar", "title": "Ranking Analysis", "orientation": "h"}
        elif any(word in user_lower for word in ['correlation', 'relationship', 'vs', 'against']):
            return {"type": "scatter", "title": "Correlation Analysis"}
        elif cols >= 2:
            return {"type": "bar", "title": "Comparison Analysis"}
        else:
            return {"type": "metric", "title": "Key Metric"}
    
    def _generate_insights(self, df: pd.DataFrame, chart_config: Dict[str, Any], user_input: str) -> str:
        """Generate insights and analysis for the chart data"""
        if df.empty or len(df.columns) < 1:
            return ""
        
        insights = []
        chart_type = chart_config.get("type", "bar")
        
        # Debug output
        
        try:
            # Basic statistics
            if len(df.columns) >= 2:
                x_col, y_col = df.columns[0], df.columns[1]
                
                # Check if y_col is numeric
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    total = df[y_col].sum()
                    avg = df[y_col].mean()
                    max_val = df[y_col].max()
                    min_val = df[y_col].min()
                    
                    # Top performer
                    top_item = df.loc[df[y_col].idxmax(), x_col]
                    
                    insights.append(f"**📊 Key Statistics:**")
                    insights.append(f"• Total: {total:,.0f}")
                    insights.append(f"• Average: {avg:.0f}")
                    insights.append(f"• Range: {min_val:,.0f} - {max_val:,.0f}")
                    insights.append(f"• Top performer: {top_item} ({max_val:,.0f})")
                    
                    # Percentage analysis for top items
                    if len(df) > 1:
                        top_percentage = (max_val / total) * 100
                        insights.append(f"• Top item represents {top_percentage:.1f}% of total")
                        
                        # Find bottom performer
                        bottom_item = df.loc[df[y_col].idxmin(), x_col]
                        bottom_percentage = (min_val / total) * 100
                        insights.append(f"• Bottom item: {bottom_item} ({bottom_percentage:.1f}% of total)")
                    
                    # Distribution insights
                    if chart_type == "pie":
                        insights.append(f"\n**🥧 Distribution Analysis:**")
                        # Top 3 categories
                        top_3 = df.nlargest(3, y_col)
                        top_3_total = top_3[y_col].sum()
                        top_3_percentage = (top_3_total / total) * 100
                        insights.append(f"• Top 3 categories account for {top_3_percentage:.1f}% of total")
                        
                        # Check for concentration
                        if top_3_percentage > 70:
                            insights.append(f"• ⚠️ High concentration in top categories")
                        elif top_3_percentage < 50:
                            insights.append(f"• ✅ Well-distributed across categories")
                    
                    # Trend insights for line charts
                    elif chart_type == "line" and len(df) > 2:
                        insights.append(f"\n**📈 Trend Analysis:**")
                        # Calculate growth
                        first_val = df[y_col].iloc[0]
                        last_val = df[y_col].iloc[-1]
                        growth = ((last_val - first_val) / first_val) * 100 if first_val != 0 else 0
                        
                        if growth > 5:
                            insights.append(f"• 📈 Positive trend: {growth:.1f}% growth")
                        elif growth < -5:
                            insights.append(f"• 📉 Declining trend: {growth:.1f}% decrease")
                        else:
                            insights.append(f"• ➡️ Stable trend: {growth:.1f}% change")
                        
                        # Find peak and valley
                        peak_idx = df[y_col].idxmax()
                        valley_idx = df[y_col].idxmin()
                        insights.append(f"• Peak: {df.loc[peak_idx, x_col]} ({max_val:,.0f})")
                        insights.append(f"• Valley: {df.loc[valley_idx, x_col]} ({min_val:,.0f})")
                    
                    # Ranking insights for bar charts
                    elif chart_type == "bar":
                        insights.append(f"\n**🏆 Ranking Insights:**")
                        if len(df) >= 5:
                            # Top and bottom performers
                            top_5_total = df.nlargest(5, y_col)[y_col].sum()
                            top_5_percentage = (top_5_total / total) * 100
                            insights.append(f"• Top 5 items contribute {top_5_percentage:.1f}% of total")
                        
                        # Performance gaps
                        if len(df) > 1:
                            gap = max_val - min_val
                            gap_percentage = (gap / max_val) * 100
                            insights.append(f"• Performance gap: {gap_percentage:.1f}% between top and bottom")
            
            # Context-specific insights based on user query
            user_lower = user_input.lower()
            insights.append(f"\n**💡 Business Insights:**")
            
            if "premium" in user_lower:
                insights.append(f"• Focus on high-premium policy types for revenue growth")
                insights.append(f"• Consider promotional strategies for underperforming segments")
            elif "customer" in user_lower and "status" in user_lower:
                insights.append(f"• Monitor customer retention and activation strategies")
                insights.append(f"• Inactive customers may need re-engagement campaigns")
            elif "claims" in user_lower:
                insights.append(f"• Track claim approval rates and processing efficiency")
                insights.append(f"• High claim volumes may indicate risk assessment needs")
            elif "agent" in user_lower:
                insights.append(f"• Top performers can mentor others and share best practices")
                insights.append(f"• Consider incentive programs for underperforming agents")
            elif "policy" in user_lower:
                insights.append(f"• Popular policy types indicate market preferences")
                insights.append(f"• Diversify offerings based on demand patterns")
            
            # Data quality insights
            if len(df) > 0:
                insights.append(f"\n**📋 Data Summary:**")
                insights.append(f"• Dataset contains {len(df)} records")
                if len(df.columns) >= 2 and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
                    non_zero = (df[df.columns[1]] > 0).sum()
                    insights.append(f"• {non_zero} records with positive values ({(non_zero/len(df)*100):.1f}%)")
        
        except Exception as e:
            insights.append(f"• Analysis completed for {len(df)} records")
            st.write(f"🔍 **Debug Error**: {str(e)}")
        
        insight_text = "\n".join(insights)
        return insight_text

    def _create_chart(self, df: pd.DataFrame, chart_config: Dict[str, Any], user_input: str):
        """Create and display chart based on data and configuration with insights"""
        if df.empty or len(df.columns) < 1:
            return
        
        chart_type = chart_config.get("type", "bar")
        title = chart_config.get("title", "Data Analysis")
        
        try:
            if chart_type == "metric" and len(df.columns) == 1:
                # Display single metric
                value = df.iloc[0, 0] if len(df) > 0 else 0
                st.metric(label=title, value=f"{value:,.0f}" if isinstance(value, (int, float)) else str(value))
                
                # Add metric insights
                insights = self._generate_insights(df, chart_config, user_input)
                if insights:
                    st.markdown("---")
                    st.markdown(insights)
                return
            
            if len(df.columns) < 2:
                st.dataframe(df, use_container_width=True)
                return
            
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            if chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, title=title)
            elif chart_type == "pie":
                fig = px.pie(df, names=x_col, values=y_col, title=title)
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title=title)
            elif chart_type == "bar" and chart_config.get("orientation") == "h":
                fig = px.bar(df, x=y_col, y=x_col, orientation='h', title=title)
            else:
                fig = px.bar(df, x=x_col, y=y_col, title=title)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate and display insights below the chart
            insights = self._generate_insights(df, chart_config, user_input)
            if insights:
                st.markdown("---")
                with st.expander("📊 **Chart Insights & Analysis**", expanded=True):
                    st.markdown(insights)
            
        except Exception as e:
            st.error(f"Chart generation failed: {e}")
            st.dataframe(df, use_container_width=True)

    def _display_data_insights(self, df: pd.DataFrame, user_input: str):
        """Display comprehensive insights from the actual queried data"""
        if df.empty:
            return
        
        with st.expander("💡 **Data Insights & Analysis**", expanded=True):
            insights = []
            
            try:
                # Basic data overview
                insights.append(f"**📊 Data Overview:**")
                insights.append(f"• Retrieved {len(df)} records")
                insights.append(f"• Columns: {', '.join(df.columns)}")
                
                # Analyze numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    insights.append(f"\n**🔢 Numeric Analysis:**")
                    for col in numeric_cols:
                        total = df[col].sum()
                        avg = df[col].mean()
                        max_val = df[col].max()
                        min_val = df[col].min()
                        
                        insights.append(f"• **{col}:**")
                        insights.append(f"  - Total: {total:,.0f}")
                        insights.append(f"  - Average: {avg:.1f}")
                        insights.append(f"  - Range: {min_val:,.0f} to {max_val:,.0f}")
                        
                        # Find top performers
                        if len(df) > 1 and len(df.columns) >= 2:
                            non_numeric_col = [c for c in df.columns if c != col][0]
                            top_idx = df[col].idxmax()
                            bottom_idx = df[col].idxmin()
                            
                            insights.append(f"  - Highest: {df.loc[top_idx, non_numeric_col]} ({max_val:,.0f})")
                            insights.append(f"  - Lowest: {df.loc[bottom_idx, non_numeric_col]} ({min_val:,.0f})")
                
                # Analyze categorical data
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    insights.append(f"\n**📋 Categorical Analysis:**")
                    for col in categorical_cols:
                        unique_count = df[col].nunique()
                        most_common = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                        
                        insights.append(f"• **{col}:** {unique_count} unique values")
                        insights.append(f"  - Most common: {most_common}")
                        
                        # Show top categories if reasonable number
                        if unique_count <= 10:
                            value_counts = df[col].value_counts()
                            insights.append(f"  - Distribution: {dict(value_counts.head(3))}")
                
                # Performance analysis
                if len(df) > 1 and len(numeric_cols) > 0:
                    insights.append(f"\n**🏆 Performance Analysis:**")
                    
                    main_metric = numeric_cols[0]
                    if len(df.columns) >= 2:
                        category_col = [c for c in df.columns if c != main_metric][0]
                        
                        # Top performers
                        top_3 = df.nlargest(3, main_metric)
                        if len(top_3) > 0:
                            insights.append(f"• **Top 3 by {main_metric}:**")
                            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                                insights.append(f"  {i}. {row[category_col]}: {row[main_metric]:,.0f}")
                        
                        # Calculate percentages
                        total = df[main_metric].sum()
                        if total > 0:
                            top_performer_pct = (df[main_metric].max() / total) * 100
                            insights.append(f"• Top performer represents {top_performer_pct:.1f}% of total")
                            
                            # Bottom performers
                            bottom_3 = df.nsmallest(3, main_metric)
                            bottom_total = bottom_3[main_metric].sum()
                            bottom_pct = (bottom_total / total) * 100
                            insights.append(f"• Bottom 3 represent {bottom_pct:.1f}% of total")
                
                # Context-specific business insights
                user_lower = user_input.lower()
                insights.append(f"\n**💼 Business Recommendations:**")
                
                if "premium" in user_lower:
                    insights.append("• Focus marketing efforts on high-premium policy types")
                    insights.append("• Consider bundling strategies for lower-premium products")
                    insights.append("• Analyze customer segments for premium optimization")
                elif "customer" in user_lower and "status" in user_lower:
                    active_customers = df[df['status'] == 'Active']['count'].sum() if 'status' in df.columns and 'count' in df.columns else 0
                    insights.append("• Implement retention programs for at-risk customers")
                    insights.append("• Develop reactivation campaigns for inactive customers")
                    insights.append("• Track customer lifecycle metrics")
                elif "claims" in user_lower:
                    insights.append("• Monitor claim processing times and approval rates")
                    insights.append("• Identify patterns in high-value claims")
                    insights.append("• Implement fraud detection for unusual claim patterns")
                elif "agent" in user_lower:
                    insights.append("• Provide training programs for underperforming agents")
                    insights.append("• Create mentorship programs with top performers")
                    insights.append("• Review territory assignments and workload distribution")
                elif "policy" in user_lower:
                    insights.append("• Expand popular policy types in target markets")
                    insights.append("• Review pricing strategies for underperforming policies")
                    insights.append("• Consider cross-selling opportunities")
                
                # Data quality assessment
                insights.append(f"\n**✅ Data Quality:**")
                missing_data = df.isnull().sum().sum()
                insights.append(f"• Missing values: {missing_data}")
                insights.append(f"• Data completeness: {((df.size - missing_data) / df.size * 100):.1f}%")
                
                if len(numeric_cols) > 0:
                    zero_values = (df[numeric_cols] == 0).sum().sum()
                    insights.append(f"• Zero values in numeric columns: {zero_values}")
                
                # Display all insights
                st.markdown("\n".join(insights))
                
                # Show data sample
                st.markdown("**📋 Data Sample:**")
                st.dataframe(df.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
                st.dataframe(df, use_container_width=True)

    def _generate_chart_for_display(self, user_input: str):
        """Generate chart for display in chat history with insights"""
        try:
            # Simple SQL generation for chart display
            if "premium" in user_input.lower() and "policy type" in user_input.lower():
                query = "SELECT policy_type, SUM(premium_amount) as total_premium FROM policies GROUP BY policy_type ORDER BY total_premium DESC"
            elif "distribution" in user_input.lower() and "policy" in user_input.lower():
                query = "SELECT policy_type, COUNT(*) as count FROM policies GROUP BY policy_type"
            elif "customer" in user_input.lower() and "status" in user_input.lower():
                query = "SELECT status, COUNT(*) as count FROM customers GROUP BY status"
            elif "commission" in user_input.lower() and "agent" in user_input.lower():
                query = "SELECT a.agent_name, SUM(c.commission_amount) as total_commission FROM agents a JOIN commissions c ON a.agent_id = c.agent_id GROUP BY a.agent_name ORDER BY total_commission DESC LIMIT 10"
            elif "claims" in user_input.lower() and "status" in user_input.lower():
                query = "SELECT claim_status, COUNT(*) as count FROM claims GROUP BY claim_status"
            elif "top" in user_input.lower() and "agent" in user_input.lower():
                query = "SELECT agent_name, commission_rate FROM agents ORDER BY commission_rate DESC LIMIT 10"
            elif "trend" in user_input.lower() or "monthly" in user_input.lower():
                query = "SELECT strftime('%Y-%m', joined_date) as month, COUNT(*) as customers FROM customers GROUP BY month ORDER BY month"
            else:
                return  # No chart for this query
            
            df = self._execute_sql(query)
            if not df.empty:
                chart_config = self._determine_chart_type(user_input, df)
                # Use the enhanced _create_chart method that includes insights
                self._create_chart(df, chart_config, user_input)
        except Exception as e:
            pass  # Silently fail for chart generation

    def process_query(self, user_input: str) -> str:
        """Process query using strict Planner→Executor→SQL Execution→Final Response flow (robust, no JSON parsing)"""
        # Step 1: Run planner and executor agents
        plan_task = Task(
            description=f"Analyze query: '{user_input}' and determine tools needed.",
            agent=self.agents["planner"],
            expected_output="JSON with exact tools needed"
        )
        exec_task = Task(
            description=f"Generate SQL query for: '{user_input}' using the insurance schema.",
            agent=self.agents["executor"],
            expected_output="Clean SQL query"
        )
        Crew(agents=[self.agents["planner"], self.agents["executor"]], tasks=[plan_task, exec_task], verbose=False).kickoff()
        # Step 2: Extract SQL from executor output robustly
        sql_query = str(exec_task.output).strip()
        import re
        sql_match = re.search(r'(SELECT[\s\S]+?;)', sql_query, re.IGNORECASE)
        if sql_match:
            sql_query = sql_match.group(1)
        else:
            for line in sql_query.splitlines():
                if line.strip().upper().startswith('SELECT'):
                    sql_query = line.strip()
                    if not sql_query.endswith(';'):
                        sql_query += ';'
                    break
        # Step 3: Execute SQL and return result
        df = pd.DataFrame()
        try:
            if sql_query and sql_query.upper().startswith('SELECT'):
                df = self._execute_sql(sql_query)
        except Exception as e:
            return f"SQL execution error: {e}"
        if not df.empty and len(df) == 1 and len(df.columns) == 1:
            value = df.iloc[0, 0]
            if "policy" in user_input.lower() and "type" in user_input.lower():
                return f"There are {value} different types of policies in the database."
            elif "customer" in user_input.lower():
                return f"There are {value} customers in the database."
            elif "claim" in user_input.lower():
                return f"There are {value} claims in the database."
            else:
                return f"The answer is {value}."
        elif not df.empty:
            return f"Query returned {len(df)} records."
        else:
            return "No data found or unable to answer the question."
def main():
    """Enhanced chat interface with chart capabilities"""
    st.set_page_config(
        page_title="Insurance Chat Analytics",
        page_icon="💬",
        layout="wide"
    )
    
    # Sidebar with sample questions
    with st.sidebar:
        st.header("📊 Sample Chart Questions")
        st.markdown("*Click any question to try it:*")
        
        # Chart categories
        if st.button("📊 Show total premium by policy type"):
            st.session_state.sample_query = "Show total premium amount by policy type"
        
        if st.button("🥧 Distribution of policy types"):
            st.session_state.sample_query = "Distribution of policy types"
            
        if st.button("📈 Customer registration trend"):
            st.session_state.sample_query = "Customer registration trend by month"
            
        if st.button("🏆 Top 10 agents by commission"):
            st.session_state.sample_query = "Top 10 agents by commission"
            
        if st.button("📋 Count of customers by status"):
            st.session_state.sample_query = "Count of customers by status"
            
        if st.button("💰 Claims distribution by status"):
            st.session_state.sample_query = "Claims distribution by status"
    
    # Main content
    st.title("💬 Insurance Data Chat")
    st.markdown("Ask questions about your insurance data and get interactive charts!")
    
    # Initialize platform
    if 'platform' not in st.session_state:
        st.session_state.platform = SimplePlatform()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Handle sample query buttons
    if 'sample_query' in st.session_state:
        prompt = st.session_state.sample_query
        del st.session_state.sample_query
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process and get response
        with st.spinner("Analyzing data..."):
            response = st.session_state.platform.process_query(prompt)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # For assistant messages, check if we need to show a chart
            if message["role"] == "assistant" and i > 0:
                user_message = st.session_state.messages[i-1]["content"]
                # Check if this was a chart-worthy query
                if any(word in user_message.lower() for word in ['show', 'distribution', 'trend', 'top', 'count', 'total']):
                    # Always generate chart with insights for chat history
                    st.session_state.platform._generate_chart_for_display(user_message)
    
    # Chat input
    if prompt := st.chat_input("Ask about your insurance data... (e.g., 'Show total premium by policy type')"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process query and display response with chart
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                response = st.session_state.platform.process_query(prompt)
            st.write(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Force a rerun to update the display
        st.rerun()

if __name__ == "__main__":
    main()