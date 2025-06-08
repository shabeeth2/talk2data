import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.express as px
import plotly.graph_objects as go
from google import genai

# Config
google_api_key = os.getenv("GOOGLE_API_KEY")
model_id = "gemini-1.5-flash"  # Updated to use faster model
client = genai.Client(api_key=google_api_key)

def gemini_response(prompt):
    """Enhanced error handling for Gemini API calls"""
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def chart_type(user_question, query_results):
    """Enhanced chart type selection with more options"""
    prompt_for_chart_design = f"""
You are an expert data visualization assistant. Analyze the user question and query results to recommend the optimal chart type.

### Inputs:
- User Question: {user_question}
- Query Results: {query_results}

### Available Chart Types:
| Use Case | Streamlit Function | Plotly Alternative |
|----------|-------------------|-------------------|
| Category comparison | `st.bar_chart()` | `px.bar()` |
| Time series trends | `st.line_chart()` | `px.line()` |
| Part-to-whole | `st.plotly_chart(px.pie())` | `px.pie()` |
| Correlations | `st.scatter_chart()` | `px.scatter()` |
| Distributions | `st.plotly_chart(px.histogram())` | `px.histogram()` |
| Geographic data | `st.map()` | `px.scatter_mapbox()` |
| Multiple metrics | `st.plotly_chart(px.bar())` | `px.bar()` |
| KPI displays | `st.metric()` | Custom cards |
| Heatmaps | `st.plotly_chart(px.imshow())` | `px.imshow()` |
| Box plots | `st.plotly_chart(px.box())` | `px.box()` |

### Guidelines:
- Prioritize Plotly for interactive charts
- Use st.metric() for single values/KPIs
- Consider data size and complexity
- Ensure chart readability

### Output Format:
CHART_TYPE: [specific function to use]
LIBRARY: [plotly/streamlit/matplotlib]
RATIONALE: [brief explanation]
DATA_PREP: [required transformations]
INSIGHTS: [key observations]
"""
    
    response = gemini_response(prompt_for_chart_design)
    return response

def get_code_response(query_results, user_question):
    """Enhanced code generation with better error handling and optimization"""
    
    # First analyze chart requirements
    task1_response = chart_type(user_question, query_results)
    
    if "Error" in task1_response:
        return f"# Error in chart analysis\nst.error('Chart analysis failed: {task1_response}')"

    prompt_code = f"""
You are an expert Python code generator for Streamlit data visualization.

### Chart Analysis:
{task1_response}

### Data:
Query Results: {query_results}
User Question: {user_question}

### Requirements:
Generate clean, executable Python code that:
1. Uses the provided df DataFrame (already created in the calling function)
2. Handles data cleaning/transformation if needed
3. Creates appropriate visualization
4. Displays insights using st.info() or st.success()
5. Uses error handling for robustness

### CRITICAL CONSTRAINT:
- The code will be executed INSIDE a st.chat_message("assistant") context
- DO NOT use st.chat_message() anywhere in the generated code
- Use direct Streamlit functions like st.write(), st.bar_chart(), st.plotly_chart(), etc.
- NO nested chat messages allowed

### Code Structure:
```python
# Data is already available as 'df'
try:
    # Check if data is available
    if df is None or df.empty:
        st.write("⚠️ No data available for visualization")
    else:
        # Rename columns if they are numeric (0, 1, 2...)
        if all(isinstance(col, (int, str)) and str(col).isdigit() for col in df.columns):
            # Infer meaningful column names based on context
            # [column renaming logic here]
        
        # Data transformations here if needed
        
        # Create visualization title
        st.write("📊 [Chart Title]")
        
        # Chart code here (NO st.chat_message!)
        st.bar_chart(df)  # or appropriate chart
        
        # Display insights
        st.write("🔍 **Key Insights:**")
        st.write("- [insights here]")
        
except Exception as e:
    st.error(f"❌ Visualization error: {{str(e)}}")
```

### Available Functions (NO chat_message):
- st.write() for text
- st.markdown() for formatted text
- st.bar_chart(), st.line_chart(), st.area_chart(), st.scatter_chart()
- st.plotly_chart() for Plotly charts
- st.metric() for KPIs
- st.info(), st.success(), st.warning(), st.error() for messages

### Constraints:
- Use only: st, pd, px, plt, go (plotly.graph_objects)
- NO imports or DataFrame creation
- NO st.chat_message() calls
- Handle empty/invalid data gracefully
- Include meaningful titles and labels
- Add interactive features where beneficial

Generate ONLY the Python code block without markdown formatting.
"""
    
    response = gemini_response(prompt_code)
    
    # Extract and clean code
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # If no code block found, use the entire response
        code = response.strip()
        # Remove any remaining markdown
        code = re.sub(r'```python\n|```\n|```', '', code)
    
    # Remove any st.chat_message calls that might have been generated
    code = re.sub(r'st\.chat_message\([^)]*\)\.', 'st.', code)
    code = re.sub(r'with st\.chat_message\([^)]*\):', '# Chat message removed', code)
    
    # Add safety wrapper if not present
    if "try:" not in code:
        code = f"""try:
    {code.replace(chr(10), chr(10) + '    ')}
except Exception as e:
    st.error(f"Visualization error: {{str(e)}}")"""
    
    return code

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