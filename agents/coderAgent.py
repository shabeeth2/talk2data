import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import plotly.express as px
import plotly.graph_objects as go
from google import genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
import re
from datetime import datetime, timedelta
import networkx as nx
from wordcloud import WordCloud
from dotenv import load_dotenv

load_dotenv()

# Config - use lazy initialization
google_api_key = os.getenv(GOOGLE_API_KEY)
model_id = "gemini-1.5-flash"  

# Initialize client lazily (only when needed)
client = None

def get_client():
    """Get or create Google AI client with proper error handling"""
    global client
    if client is None:
        try:
            if google_api_key:
                client = genai.Client(api_key=google_api_key)
            else:
                print("Warning: GOOGLE_API_KEY not found in environment variables")
                return None
        except Exception as e:
            print(f"Warning: Failed to initialize Google AI client: {e}")
            return None
    return client

def gemini_response(prompt):
    """Enhanced error handling for Gemini API calls"""
    client = get_client()
    if client is None:
        return "Error: Google AI client not initialized. Please check your API key."
    
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# def chart_type(user_question, query_results):
#     """Enhanced chart type selection with more options"""
#     prompt_for_chart_design = f"""
# You are an expert data visualization assistant. Analyze the user question and query results to recommend the optimal chart type.

# ### Inputs:
# - User Question: {user_question}
# - Query Results: {query_results}

# ### Available Chart Types:
# | Use Case | Streamlit Function | Plotly Alternative |
# |----------|-------------------|-------------------|
# | Category comparison | `st.bar_chart()` | `px.bar()` |
# | Time series trends | `st.line_chart()` | `px.line()` |
# | Part-to-whole | `st.plotly_chart(px.pie())` | `px.pie()` |
# | Correlations | `st.scatter_chart()` | `px.scatter()` |
# | Distributions | `st.plotly_chart(px.histogram())` | `px.histogram()` |
# | Geographic data | `st.map()` | `px.scatter_mapbox()` |
# | Multiple metrics | `st.plotly_chart(px.bar())` | `px.bar()` |
# | KPI displays | `st.metric()` | Custom cards |
# | Heatmaps | `st.plotly_chart(px.imshow())` | `px.imshow()` |
# | Box plots | `st.plotly_chart(px.box())` | `px.box()` |

# ### Guidelines:
# - Prioritize Plotly for interactive charts
# - Use st.metric() for single values/KPIs
# - Consider data size and complexity
# - Ensure chart readability

# ### Output Format:
# CHART_TYPE: [specific function to use]
# LIBRARY: [plotly/streamlit/matplotlib]
# RATIONALE: [brief explanation]
# DATA_PREP: [required transformations]
# INSIGHTS: [key observations]
# """
    
#     response = gemini_response(prompt_for_chart_design)
#     return response

# def get_code_response(query_results, user_question):
#     """Enhanced code generation with better error handling and optimization"""
    
#     # First analyze chart requirements
#     task1_response = chart_type(user_question, query_results)
    
#     if "Error" in task1_response:
#         return f"# Error in chart analysis\nst.error('Chart analysis failed: {task1_response}')"

#     prompt_code = f"""
# You are an expert Python code generator for Streamlit data visualization.

# ### Chart Analysis:
# {task1_response}

# ### Data:
# Query Results: {query_results}
# User Question: {user_question}

# ### Requirements:
# Generate clean, executable Python code that:
# 1. Uses the provided df DataFrame (already created in the calling function)
# 2. Handles data cleaning/transformation if needed
# 3. Creates appropriate visualization
# 4. Displays insights using st.info() or st.success()
# 5. Uses error handling for robustness

# ### CRITICAL CONSTRAINT:
# - The code will be executed INSIDE a st.chat_message("assistant") context
# - DO NOT use st.chat_message() anywhere in the generated code
# - Use direct Streamlit functions like st.write(), st.bar_chart(), st.plotly_chart(), etc.
# - NO nested chat messages allowed

# ### Code Structure:
# ```python
# # Data is already available as 'df'
# try:
#     # Check if data is available
#     if df is None or df.empty:
#         st.write("⚠️ No data available for visualization")
#     else:
#         # Rename columns if they are numeric (0, 1, 2...)
#         if all(isinstance(col, (int, str)) and str(col).isdigit() for col in df.columns):
#             # Infer meaningful column names based on context
#             # [column renaming logic here]
        
#         # Data transformations here if needed
        
#         # Create visualization title
#         st.write("📊 [Chart Title]")
        
#         # Chart code here (NO st.chat_message!)
#         st.bar_chart(df)  # or appropriate chart
        
#         # Display insights
#         st.write("🔍 **Key Insights:**")
#         st.write("- [insights here]")
        
# except Exception as e:
#     st.error(f"❌ Visualization error: {{str(e)}}")
# ```

# ### Available Functions (NO chat_message):
# - st.write() for text
# - st.markdown() for formatted text
# - st.bar_chart(), st.line_chart(), st.area_chart(), st.scatter_chart()
# - st.plotly_chart() for Plotly charts
# - st.metric() for KPIs
# - st.info(), st.success(), st.warning(), st.error() for messages

# ### Constraints:
# - Use only: st, pd, px, plt, go (plotly.graph_objects)
# - NO imports or DataFrame creation
# - NO st.chat_message() calls
# - Handle empty/invalid data gracefully
# - Include meaningful titles and labels
# - Add interactive features where beneficial

# Generate ONLY the Python code block without markdown formatting.
# """
    
#     response = gemini_response(prompt_code)
    
#     # Extract and clean code
#     code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
#     if code_match:
#         code = code_match.group(1).strip()
#     else:
#         # If no code block found, use the entire response
#         code = response.strip()
#         # Remove any remaining markdown
#         code = re.sub(r'```python\n|```\n|```', '', code)
    
#     # Remove any st.chat_message calls that might have been generated
#     code = re.sub(r'st\.chat_message\([^)]*\)\.', 'st.', code)
#     code = re.sub(r'with st\.chat_message\([^)]*\):', '# Chat message removed', code)
    
#     # Add safety wrapper if not present
#     if "try:" not in code:
#         code = f"""try:
#     {code.replace(chr(10), chr(10) + '    ')}
# except Exception as e:
#     st.error(f"❌ Visualization error: {{str(e)}}")"""
    
#     return code


##################################################

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
import re
from datetime import datetime, timedelta
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def chart_type_analyzer(user_question, query_results):
    """Enhanced chart type selection with 31 chart types and dashboard elements"""
    
    data_str = str(query_results)[:1000] 
    
    prompt_for_chart_design = f"""
# MISSION
You are a world-class Principal Data Analyst. Your mission is to perform a rigorous analysis of a user's question and the provided data context. Your goal is to produce a detailed and precise **Visualization Blueprint** in YAML format. This blueprint will serve as the definitive instruction set for a downstream AI code generator. You must not generate the code itself, only the blueprint.

# INPUTS
1.  **`{user_question}`**: The user's specific question or goal. This is your primary directive.
2.  **`{query_results}`**: A rich summary of the dataset, including:
    *   `df.head(5).to_markdown()`: To understand the columns and sample values.
    *   `df.info()`: To understand data types, non-null counts, and memory usage.
    *   `df.describe().to_markdown()`: To understand the statistical distribution of numerical columns.

# REASONING FRAMEWORK
Follow these steps to construct your analysis:

1.  **Deconstruct User Intent:** Analyze the `{user_question}`. What is the core analytical task? (e.g., "compare performance across categories," "show a trend over time," "analyze the relationship between two variables," "understand the composition of a whole").

2.  **Analyze Data Properties:**
    *   Thoroughly examine the `{query_results}`.
    *   For each column, identify its data type (Categorical, Numerical-Continuous, Numerical-Discrete, Datetime, Geospatial, Text).
    *   Identify potential roles for each column: Is it a Dimension (something to group by), a Measure (something to aggregate), a Time Axis, a Geographic Identifier, etc.?

3.  **Synthesize and Select:** Based on the user's intent and the data's properties, select the single most effective `primary_chart` from the Chart Library. If the analysis is complex, recommend supplementary `dashboard_elements`.

4.  **Create the Blueprint:** Meticulously fill out the YAML output format. Be precise and explicit, especially in the `column_mapping` and `data_prep` sections. The goal is to create a perfect, machine-readable recipe.

# CHART LIBRARY (Reference)
 Basic Charts
Line Chart (px.line)

Bar Chart (px.bar, go.Bar)

Scatter Plot (px.scatter, go.Scatter)

Pie Chart (px.pie, go.Pie)

Histogram (px.histogram)

Box Plot (px.box)

Violin Plot (px.violin)

Area Chart (px.area)

Density Heatmap (px.density_heatmap)

Bubble Chart (px.scatter with size)

 Advanced Charts
Candlestick Chart (go.Candlestick)

OHLC Chart (go.Ohlc)

Sunburst Chart (px.sunburst)

Treemap (px.treemap)

Funnel Chart (px.funnel)

Waterfall Chart (go.Waterfall)

Radar / Spider Chart (go.Scatterpolar)

Parallel Coordinates (px.parallel_coordinates)

Parallel Categories (px.parallel_categories)

Sankey Diagram (go.Sankey)

Network Graph (networkx + plotly)

Gantt Chart (ff.create_gantt)

Word Cloud (WordCloud, rendered via matplotlib)

Correlation Matrix (ff.create_annotated_heatmap or sns.heatmap)


# OUTPUT FORMAT (Strict YAML)
Produce a single, valid YAML code block. Do not add any commentary before or after the YAML block.

```yaml
primary_chart:
  chart_type: [Specific type from Chart Library, e.g., LINE_CHART]
  title: [A clear, descriptive title for the chart, e.g., "Monthly Sales Trend Over the Last Year"]
  rationale: [Concise justification. Connect user intent, data types, and chart choice. e.g., "The user wants to see a trend, and the data contains a datetime column ('order_date') and a numerical measure ('sales'), making a LINE_CHART the ideal choice."]
  column_mapping:
    # --- CRITICAL: Map data columns to visual roles ---
    x_axis: [column_name_for_x_axis, e.g., 'order_date']
    y_axis: [column_name_for_y_axis, e.g., 'sales_amount']
    color: [Optional: column_name_for_color_encoding, e.g., 'product_category']
    size: [Optional: column_name_for_size_encoding (for BUBBLE_CHART), e.g., 'profit_margin']
    facet: [Optional: column_name_for faceting (small multiples), e.g., 'region']
    hover_data: [List of additional columns for tooltip context, e.g., ['customer_name', 'order_id']]
    # ... other relevant mappings for specific charts (e.g., lat, lon for maps)

data_prep:
  - "[Describe step 1 of data prep, e.g., Convert 'order_date' column to datetime objects using `pd.to_datetime`]"
  - "[Describe step 2 of data prep, e.g., Handle missing values in 'sales_amount' by filling with 0]"
  - "[Describe step 3 of data prep, e.g., Aggregate data to a monthly level: `df.resample('M', on='order_date')['sales_amount'].sum()`]"

dashboard_elements:
  - type: [e.g., KPI_CARDS]
    metrics:
      - label: "Total Revenue"
        value: "Sum of 'sales_amount'"
      - label: "Average Order Value"
        value: "Mean of 'sales_amount'"
  - type: [e.g., FILTERS]
    controls:
      - filter_on_column: 'region'
        control_type: 'st.multiselect'
        label: "Select Region(s)"

key_insights_to_highlight:
  - "Identify the overall trend (upward, downward, or stable)."
  - "Pinpoint the month with the highest and lowest sales."
  - "Check for any noticeable seasonality or cyclical patterns in the data."
"""
    
    response = gemini_response(prompt_for_chart_design)
    return response

FEW_SHOT_CHART_EXAMPLES = [
    {
        "user_question": "Show the distribution of insurance charges by region and smoking status.",
        "query_results": "[('southwest', 'yes', 16884.92), ('southeast', 'no', 1725.55), ('southeast', 'no', 4449.46), ('northwest', 'no', 21984.47), ('southeast', 'yes', 27808.72), ('southwest', 'no', 1826.84)]",
        "viz_code": """
# The DataFrame 'df' is created automatically in the calling function.
# We just need to name the columns and create the chart.
df.columns = ['region', 'smoker', 'charges']
fig = px.box(
    df,
    x='region',
    y='charges',
    color='smoker',
    title='Distribution of Charges by Region and Smoking Status',
    labels={
        "charges": "Insurance Charges ($)",
        "region": "Geographic Region",
        "smoker": "Smoker Status"
    },
    color_discrete_map={'yes': '#FF5733', 'no': '#33CFFF'}
)
fig.update_layout(legend_title_text='Smoker Status')
st.plotly_chart(fig, use_container_width=True)
"""
    },
    {
        "user_question": "What is the trend of monthly sales revenue?",
        "query_results": "[('2023-01', 150000), ('2023-02', 175000), ('2023-03', 160000), ('2023-04', 180000), ('2023-05', 210000), ('2023-06', 195000)]",
        "viz_code": """
df.columns = ['month', 'monthly_revenue']
df = df.sort_values('month')  # Sort by month for a correct time-series plot
fig = px.line(
    df, 
    x='month', 
    y='monthly_revenue', 
    title='Monthly Premium Revenue Trend',
    markers=True,
    labels={'month': 'Month', 'monthly_revenue': 'Premium Revenue ($)'}
)
fig.update_layout(xaxis_title="Month", yaxis_title="Revenue")
st.plotly_chart(fig, use_container_width=True)
"""
    },
    {
        "user_question": "List the top 5 agents by total commission amount.",
        "query_results": "[('John Doe', 5500.0), ('Jane Smith', 5200.0), ('Peter Jones', 4800.0), ('Mary Williams', 4750.0), ('David Brown', 4600.0)]",
        "viz_code": """
df.columns = ['agent_name', 'commission_amount']
fig = px.bar(
    df.sort_values('commission_amount', ascending=False), 
    x='agent_name', 
    y='commission_amount', 
    title='Top 5 Agents by Commission Amount',
    labels={'agent_name': 'Agent Name', 'commission_amount': 'Total Commission ($)'},
    color='commission_amount',  # Color by commission amount instead of agent name
    color_continuous_scale='viridis'
)
fig.update_layout(xaxis_title="Agent", yaxis_title="Commission Earned")
st.plotly_chart(fig, use_container_width=True)
"""
    },
    {
        "user_question": "Show me the breakdown of policies by type.",
        "query_results": "[('Auto Insurance', 1500), ('Health Insurance', 1200), ('Home Insurance', 950), ('Life Insurance', 800)]",
        "viz_code": """
df.columns = ['policy_type', 'count']
fig = px.pie(
    df, 
    names='policy_type', 
    values='count', 
    title='Distribution of Policy Types',
    hole=0.3
)
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig, use_container_width=True)
"""
    },
    {
        "user_question": "What is the relationship between customer age and policy premium amount?",
        "query_results": "[(25, 1200), (45, 2500), (33, 1800), (60, 3500), (22, 1100), (51, 2900)]",
        "viz_code": """
df.columns = ['age', 'premium_amount']
fig = px.scatter(
    df,
    x='age',
    y='premium_amount',
    title='Customer Age vs. Premium Amount',
    labels={'age': 'Custer Age', 'premium_amount': 'Premium Amount ($)'},
    trendline='ols'  # Add a regression line to show the trend
)
st.plotly_chart(fig, use_container_width=True)
"""
    }
]
examples_code = "\n".join([
        f"Question: {ex['user_question']}\nQuery Result: {ex['query_results']}\n Code : {ex['viz_code']}\n"
        for ex in FEW_SHOT_CHART_EXAMPLES ])

def get_code_response(query_results, user_question):
    """Enhanced code generation for chart types and dashboard elements"""

    chart_analysis = chart_type_analyzer(user_question, query_results)
        
    

    prompt_code = f"""
# # MISSION
You are an expert-level Data Visualization Engineer. Your mission is to translate data analysis into a Streamlit application. You will receive a DataFrame and a user question. Your task is to generate a single, clean, and executable Python script that creates the most effective visualization to answer that question.

# CONTEXT & INPUTS
You will be provided with:

1.  **`{chart_analysis}`**: A high-level summary of the data and the recommended visualization approach.
2.  **`{query_results}`**: A string representation of the pandas DataFrame's head.
3.  **`{user_question}`**: The original question the user is trying to answer.

# OUTPUT REQUIREMENTS
- Generate *only* the Python code. Do not include any surrounding text, explanations, or markdown formatting.
- The code must be clean, well-commented, and immediately usable in a Streamlit environment.
- The placeholder `df` is assumed to be an available pandas DataFrame.
- Follow the style of the few-shot examples provided below.

**Example codes for visualization**: 
(mimic the code structure below, but adapt to the specific chart type and data context)

{examples_code}

"""
    
    response = gemini_response(prompt_code)
    code = extract_and_clean_code(response)
    return code


def extract_and_clean_code(response):
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        code = response.strip()
        code = re.sub(r'```python\n|```\n|```', '', code)
    code = re.sub(r'st\.chat_message\([^)]*\)\.', 'st.', code)
    code = re.sub(r'with st\.chat_message\([^)]*\):', '# Chat message removed', code)
    
    if "try:" not in code and "except:" not in code:
        code = f"""try:
    # Handle DataFrames with numeric column indices
    if df is not None and not df.empty:
        # If columns are numeric (0, 1, 2...), assign meaningful names
        if all(isinstance(col, (int, np.integer)) for col in df.columns):
            if len(df.columns) == 1:
                df.columns = ['value']
            elif len(df.columns) == 2:
                df.columns = ['category', 'value']
            elif len(df.columns) == 3:
                df.columns = ['category', 'subcategory', 'value']
            else:
                df.columns = [f'column_{{i}}' for i in range(len(df.columns))]
    
    {code.replace(chr(10), chr(10) + '    ')}
except Exception as e:
    st.error(f"❌ Visualization error: {{str(e)}}")
    # Fallback visualization
    if not df.empty and len(df.columns) >= 2:
        # Ensure column names are strings for fallback
        if all(isinstance(col, (int, np.integer)) for col in df.columns):
            df.columns = [f'col_{{i}}' for i in range(len(df.columns))]
        col1, col2 = df.columns[0], df.columns[1]
        if df[col2].dtype in ['int64', 'float64']:
            fig = px.bar(df.head(10), x=col1, y=col2, title="Data Overview")
            st.plotly_chart(fig, use_container_width=True)"""
    
    return code


##################################################
#  NETWORK GRAPH
##################################################

def create_network_traces(G, pos):
    """Helper function to create network graph traces"""
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=0.5, color='#888'),
                           hoverinfo='none',
                           mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(x=node_x, y=node_y,
                           mode='markers',
                           hoverinfo='text',
                           marker=dict(size=10, color='lightblue',
                                     line=dict(width=2, color='black')))
    
    return edge_trace, node_trace

# Enhanced chart creation with all 31 types
# def create_advanced_visualization(df, chart_type, user_question):
#     ""Create visualization based on specific chart type""
    
#     chart_creators = {
#         'BAR_CHART': create_bar_chart,
#         'LINE_CHART': create_line_chart,
#         'PIE_CHART': create_pie_chart,
#         'SCATTER_PLOT': create_scatter_plot,
#         'HEATMAP': create_heatmap,
#         'BOX_PLOT': create_box_plot,
#         'VIOLIN_PLOT': create_violin_plot,
#         'TREEMAP': create_treemap,
#         'SUNBURST_CHART': create_sunburst,
#         'SANKEY_DIAGRAM': create_sankey,
#         'RADAR_CHART': create_radar_chart,
#         'WATERFALL_CHART': create_waterfall,
#         'GANTT_CHART': create_gantt,
#         'CANDLESTICK_CHART': create_candlestick,
#         'BUBBLE_CHART': create_bubble_chart,
#         'NETWORK_GRAPH': create_network_graph,
#         'PARALLEL_COORDINATES': create_parallel_coordinates,
#         'CHOROPLETH_MAP': create_choropleth,
#         'WORD_CLOUD': create_word_cloud,
#         'HISTOGRAM': create_histogram,
#         'AREA_CHART': create_area_chart,
#         'DOUGHNUT_CHART': create_doughnut_chart,
#         # Add more chart creators as needed
#     }
    
#     creator_func = chart_creators.get(chart_type, create_bar_chart)
#     return creator_func(df, user_question)

# Individual chart creator functions (samples)
def create_bar_chart(df, title):
    if len(df.columns) >= 2:
        col1, col2 = df.columns[0], df.columns[1]
        fig = px.bar(df, x=col1, y=col2, title=title)
        return fig
    return None

def create_line_chart(df, title):
    if len(df.columns) >= 2:
        col1, col2 = df.columns[0], df.columns[1]
        fig = px.line(df, x=col1, y=col2, title=title)
        return fig
    return None

def create_pie_chart(df, title):
    if len(df.columns) >= 2:
        col1, col2 = df.columns[0], df.columns[1]
        fig = px.pie(df, names=col1, values=col2, title=title)
        return fig
    return None




def create_intelligent_fallback(df, user_question):
    """Create fallback visualization based on data analysis"""
    if df is None or df.empty:
        st.info("No data available for visualization")
        return
    
    try:
        # Analyze data characteristics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            # Bar chart for mixed data
            fig = px.bar(df.head(20), x=categorical_cols[0], y=numeric_cols[0],
                        title="Data Overview")
            st.plotly_chart(fig, use_container_width=True)
        elif len(numeric_cols) >= 2:
            # Scatter plot for numeric data
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                           title="Correlation Analysis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple table fallback
            st.dataframe(df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Fallback visualization failed: {str(e)}")
        st.dataframe(df.head(10) if not df.empty else pd.DataFrame(), 
                    use_container_width=True)

def generate_insight_summary(query_results, user_question):
    """Generate actionable insights from data"""
    prompt_insights = f"""
Analyze the following data and provide 3-5 key business insights.

User Question: {user_question}
Data: {query_results}

Provide insights in this format:
1. **Key Finding**: [insight]
2. **Trend**: [pattern observed]
3. **Recommendation**: [actionable suggestion]

Keep insights concise and business-focused.
"""
    
    return gemini_response(prompt_insights)

def create_fallback_analysis(user_question, query_results):
    """Create a simple fallback analysis when the main analyzer fails"""
    # Determine chart type based on simple heuristics
    question_lower = user_question.lower()
    
    if any(word in question_lower for word in ['trend', 'time', 'month', 'year', 'over time']):
        chart_type = "LINE_CHART"
        rationale = "Time-based data detected, using line chart for trends"
    elif any(word in question_lower for word in ['compare', 'comparison', 'by', 'across']):
        chart_type = "BAR_CHART" 
        rationale = "Comparison detected, using bar chart"
    elif any(word in question_lower for word in ['distribution', 'breakdown', 'percentage', 'proportion']):
        chart_type = "PIE_CHART"
        rationale = "Distribution analysis detected, using pie chart"
    elif any(word in question_lower for word in ['relationship', 'correlation', 'vs', 'versus']):
        chart_type = "SCATTER_PLOT"
        rationale = "Relationship analysis detected, using scatter plot"
    else:
        chart_type = "BAR_CHART"
        rationale = "Default visualization using bar chart"
    
    return f"""
Chart Type: {chart_type}
Rationale: {rationale}
Data should be processed for {chart_type.lower().replace('_', ' ')} visualization.
Key insights should focus on the main patterns in the data.
"""
