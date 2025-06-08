from google.genai import types
import os
from google import genai
import pandas as pd
import matplotlib.pyplot as plt
import re
import plotly.express as px 
google_api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=google_api_key)
# model = genai.GenerativeModel(model_name="gemini-1.5-flash")
from google import genai
model_id = "gemini-2.0-flash"

client = genai.Client(api_key=google_api_key)
def gemini_response(prompt):
        response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                # config = types.GenerateContentConfig(
                #     tools=[types.Tool(
                #         code_execution=types.ToolCodeExecution
                #         )]
                #     )
                )
        return response.text
def chart_type(user_question,query_results):
      ## Task 1
      prompt_for_chart_design = f"""
    
You are a chart design assistant. Your task is to determine the most suitable chart type and required data transformations based on a user's question and a tabular query result.

### Inputs:

* `user_question`: A natural language question from the user.-- {user_question}
* `query_results`: A SQL query result represented as a Python dictionary (tabular data).-- {query_results}

### Objective:

Analyze both inputs to select the best visualization strategy using the following steps:

1. **Understand the data** in `query_results`: Identify column names, data types, and patterns (e.g., time series, categories, measures).
2. **Interpret the user’s intent** from `user_question`.
3. **Select the correct chart type** strictly based on the table below:

| Use Case                            | Chart Type                      |
| ----------------------------------- | ------------------------------- |
| Comparing multiple categories       | `st.bar_chart()`                |
| Tracking trends over time           | `st.line_chart()`               |
| Showing part-to-whole relationships | `st.altair_chart()` (pie chart) |
| Identifying correlations            | `st.scatter_chart()`            |
| Showing cumulative trends           | `st.altair_chart()`               |
| Plotting distributions              | `st.scatter_chart()`            |
| Showing geographic data             | `st.map()`                      |
| Complex/custom visualizations       | `st.altair_chart()`             |

4. **List required data transformations** to prepare the data for the selected chart type. This may include:

   * Convert string dates to datetime
   * Make sure Datetime is in the correct format for time series charts
   * Sort or index time-based data chronologically
   * Reshape (pivot/unpivot) tables
   * Aggregate, filter, or compute derived values
   * Round excessive float precision
   * Handle missing/null values (assume clean data, but note risk)


5. **Avoid common pitfalls**:

   * Do not use string dates for chronological indexing
   * Do not label a chart as “cumulative” unless a cumulative sum is applied
   * Do not allow floating-point artifacts to distort values visually
    

### Output Format:

Respond in the exact structure below:

```
CHART_TYPE: [selected Streamlit chart function]
RATIONALE: [why this chart was selected, referencing the table]
DATA_TRANSFORMATIONS:
- [list of specific data changes or calculations]
KEY_INSIGHTS:
- [brief, high-level data observations based on the result]
```

Your output will be used directly by a downstream system to generate the final charting code. Be concise, accurate, and deterministic.
"""
      
      response=gemini_response(prompt_for_chart_design)
      return response

def get_code_response(query_results, user_question):
    task1_response = chart_type(user_question,query_results)
    prompt_code1= f"""
You are a code generation assistant for data visualization using Streamlit.

You are provided with:
- A natural language user question (`user_question`)
- A SQL query result as a Python dictionary (`query_results`) representing tabular data

Your objective is to:
1. Analyze the user's question and the content of `query_results` to determine the most appropriate visualization.
2. Generate concise Python code using Streamlit's `st.chat_message()` API to:
   - **Curate the data** from `query_results` to ensure it's perfectly fit for the chart. This might involve data transformation, aggregation, or filtering.
   - Display an appropriate chart
   - Provide a brief written interpretation of the data
   - Surface key insights

**Constraints:**
- Output only valid, efficient Python code
- Do not include any import statements or Streamlit boilerplate
- Use only `st.chat_message("assistant")` to display all outputs (text and charts)
- Prioritize brevity and directness in the code
- Do not use a line chart unless the data involves a time series or continuous trend.
- Prefer bar charts for categorical comparisons, scatter charts for correlations, area charts for cumulative trends, and map for geospatial data as specified in the table below. 


**Visualization Strategy:**

Use the following chart type depending on the question and data:

| **Use Case** | **Chart Type** |
| --- | --- |
| Comparing multiple categories | `st.bar_chart()` |
| Tracking trends over time | `st.line_chart()` |
| Showing part-to-whole relationships | `` () |
| Identifying correlations | `st.scatter_chart()` |
| Plotting distributions | `st.scatter_chart()` |
| Showing geographic data | `st.map()` |
| Complex or custom visualizations | `st.altair_chart()` |

st.line_chart(): Ideal for visualizing trends over time, such as monthly revenues or daily user sign-ups. Dont use it unless the data is time series or continuous trend. 
st.bar_chart(): Best for comparing quantities across categories, like sales by region or product. 

st.altair_chart(): Useful for displaying cumulative data or emphasizing the magnitude of change over time.
st.scatter_chart(): Great for exploring relationships between two variables, such as age versus income.
st.map(): Designed for plotting geographic data points, like customer locations or event venues.
st.plotly_chart(): For complex visualizations, such as interactive charts or 3D plots.
st.altair_chart(): For creating declarative visualizations with Altair, especially for statistical graphics.(like PIE)

# note: Try to utilize parameters of the chart functions to enhance the visualizations, such as color, size, horizontal and labels.




DATA FOR CHARTS 
    -- decide the chart type based on the user question below here --
query_results = {query_results}
user_question = "{user_question}"

DECIDE THE CHART TYPE BASED ON THE USER QUESTION("{user_question}") AND DATA CHARACTERISTICS AMONG THE FOLLOWING:
scatter plot, bar chart, line chart, area chart, pie chart, map, plotly chart, altair chart, histogram, box plot, heatmap, etc.




**Instructions for Output:**

* Always Start by defining `query_results = {query_results}` at the top of the code block.
* Create a DataFrame `df = pd.DataFrame(query_results)`
* Use `st.chat_message("assistant").write("🔍Insights: ...")` for key insights.

** DOs & DON'T**
*- Every code you generate MUST have `query_results` as a DataFrame (df) as a first step.
*- Do not write the example data in the code, just write the logic to curate the query_results for chart....
*- Use `st.chat_message("assistant").write(-...)` to describe the chart purpose
*- Use `st.chat_message("assistant").write("🔍 Insights:\n: ...")` for key insights
*- create two charts for the same data if needed.(e.g. line and bar)
*- Use different colors for different charts and data points.
*- Do not add string literals like   "  ```python"  " or "```" in the code.

Instruction for Data Wrangling FOR THE CHART:
*-After deciding the chart type based on the user question you need to curate the data so it fits the chart type.
*-After defining `query_results`, create a DataFrame `df = pd.DataFrame(query_results)`.
*-Organize the data for the chart to fit the chart type.
*-write the logic to clean and organize the query_results carefully for chart.
*-Make sure no Error when df is given to the chart function.

**Examples :**
Caution: 
*-The examples below are for illustration only. Do not include them in your output.
*- Query results may vary in structure, so adapt the code accordingly.
*-Do not get influenced by the examples. Focus on the logic to curate the query_results and the user question to generate the chart.

Example 1:
user_question = "Compare claims by status."
query_results = {{'status': ['Active', 'Cancelled', 'Expired', 'Pending'], 'count': [240, 95, 60, 30] }}

Expected output:
# define query_results variable
query_results = {{ 'status': ['Active', 'Cancelled', 'Expired', 'Pending'], 'count': [240, 95, 60, 30] }}
## Curate the data for the chart....
df = pd.DataFrame(query_results)
st.chat_message("assistant").write("Claims by status.")
st.bar_chart(df.set_index('status'))
st.chat_message("assistant").write("Insights: \n Active policies have most claims.")


Example 3:
user_question = "Show relationship between premium amount and claim count per policy type across months."
query_results = [('2020-01', 'Health', 11776110.23, 216), ('2020-01', 'Life', 11029755.59, 238), ('2020-02', 'Health', 11197073.63, 240), ('2020-02', 'Life', 11998600.18, 248), ('2020-03', 'Health', 11826321.65, 245), ('2020-03', 'Life', 12921025.31, 243), ('2020-04', 'Health', 12797303.71, 246), ('2020-04', 'Life', 11546415.65, 212), ('2020-05', 'Health', 11918810.3, 250), ('2020-05', 'Life', 10809620.21, 214), ('2020-06', 'Health', 12889317.58, 249), ('2020-06', 'Life', 12223778.6, 254), ('2020-07', 'Health', 12732136.86, 261), ('2020-07', 'Life', 13292991.49, 234), ('2020-08', 'Health', 11484403.37, 225), ('2020-08', 'Life', 12566097.74, 234), ('2020-09', 'Health', 11790111.73, 263), ('2020-09', 'Life', 10846939.32, 208), ('2020-10', 'Health', 12401325.06, 256), ('2020-10', 'Life', 11451195.06, 199), ('2020-11', 'Health', 10815875.01, 205), ('2020-11', 'Life', 11637688.75, 232), ('2020-12', 'Health', 11774826.91, 235), ('2020-12', 'Life', 13460361.22, 242), ('2021-01', 'Health', 11797407.58, 234), ('2021-01', 'Life', 12684593.55, 254), ('2021-02', 'Health', 10015167.42, 197), ('2021-02', 'Life', 10569719.98, 214), ('2021-03', 'Health', 11902733.72, 254), ('2021-03', 'Life', 12894469.54, 252), ('2021-04', 'Health', 13438386.19, 258), ('2021-04', 'Life', 12611425.07, 269), ('2021-05', 'Health', 11786311.46, 225), ('2021-05', 'Life', 12463547.12, 259), ('2021-06', 'Health', 11673869.07, 238), ('2021-06', 'Life', 11001443.57, 200), ('2021-07', 'Health', 13144402.17, 238), ('2021-07', 'Life', 12717566.95, 270), ('2021-08', 'Health', 12736982.13, 248), ('2021-08', 'Life', 12544525.87, 278), ('2021-09', 'Health', 12526004.38, 267), ('2021-09', 'Life', 12201103.76, 233), ('2021-10', 'Health', 11657384.1, 224), ('2021-10', 'Life', 12549015.87, 230), ('2021-11', 'Health', 11638402.84, 249), ('2021-11', 'Life', 11058643.79, 213), ('2021-12', 'Health', 11996451.08, 250), ('2021-12', 'Life', 11879080.08, 231), ('2022-01', 'Health', 12265201.7, 240), ('2022-01', 'Life', 12529757.61, 237), ('2022-02', 'Health', 11779694.040000001, 233), ('2022-02', 'Life', 10621883.9, 210), ('2022-03', 'Health', 12445861.08, 270), ('2022-03', 'Life', 12143624.73, 242), ('2022-04', 'Health', 11959494.81, 234), ('2022-04', 'Life', 12472632.95, 247), ('2022-05', 'Health', 13068283.61, 263), ('2022-05', 'Life', 12696867.43, 254), ('2022-06', 'Health', 12793652.93, 269), ('2022-06', 'Life', 12828241.83, 260), ('2022-07', 'Health', 12316466.31, 241), ('2022-07', 'Life', 11377428.71, 215), ('2022-08', 'Health', 12183717.2, 267), ('2022-08', 'Life', 11993100.53, 233), ('2022-09', 'Health', 11887502.38, 257), ('2022-09', 'Life', 11917046.77, 226), ('2022-10', 'Health', 12343447.65, 250), ('2022-10', 'Life', 13866857.66, 287), ('2022-11', 'Health', 12478120.62, 249), ('2022-11', 'Life', 11612800.12, 225), ('2022-12', 'Health', 12185613.7, 239), ('2022-12', 'Life', 13328368.74, 258), ('2023-01', 'Health', 11978478.85, 233), ('2023-01', 'Life', 11113733.59, 209), ('2023-02', 'Health', 11419903.68, 224), ('2023-02', 'Life', 9767383.8, 216), ('2023-03', 'Health', 11839784.13, 218), ('2023-03', 'Life', 12491398.89, 252), ('2023-04', 'Health', 12455953.459999999, 245), ('2023-04', 'Life', 12099715.4, 255), ('2023-05', 'Health', 12496581.16, 266), ('2023-05', 'Life', 12963817.16, 248), ('2023-06', 'Health', 13483056.51, 242), ('2023-06', 'Life', 12363175.89, 205), ('2023-07', 'Health', 11640081.33, 214), ('2023-07', 'Life', 11723705.48, 221), ('2023-08', 'Health', 12686900.77, 252), ('2023-08', 'Life', 11238280.97, 222), ('2023-09', 'Health', 11306067.58, 219), ('2023-09', 'Life', 11942113.57, 243), ('2023-10', 'Health', 11751292.47, 229), ('2023-10', 'Life', 11592650.45, 216), ('2023-11', 'Health', 13051473.7, 265), ('2023-11', 'Life', 12939476.17, 223), ('2023-12', 'Health', 12624850.4, 243), ('2023-12', 'Life', 11788645.68, 234), ('2024-01', 'Health', 13069814.23, 230), ('2024-01', 'Life', 13010159.69, 256), ('2024-02', 'Health', 12281447.9, 226), ('2024-02', 'Life', 10912527.95, 221), ('2024-03', 'Health', 13933755.85, 296), ('2024-03', 'Life', 13066762.4, 260), ('2024-04', 'Health', 11052902.76, 212), ('2024-04', 'Life', 11739914.21, 247), ('2024-05', 'Health', 11589784.97, 222), ('2024-05', 'Life', 11404677.85, 213), ('2024-06', 'Health', 12692735.39, 235), ('2024-06', 'Life', 11415253.02, 217), ('2024-07', 'Health', 12983150.68, 251), ('2024-07', 'Life', 12299245.13, 232), ('2024-08', 'Health', 11972975.38, 255), ('2024-08', 'Life', 13006786.45, 273), ('2024-09', 'Health', 11429344.44, 226), ('2024-09', 'Life', 12121036.29, 200), ('2024-10', 'Health', 11291003.42, 242), ('2024-10', 'Life', 12821869.39, 235), ('2024-11', 'Health', 11848304.6, 256), ('2024-11', 'Life', 12505653.74, 239), ('2024-12', 'Health', 12100420.3, 251), ('2024-12', 'Life', 12525340.73, 250), ('2025-01', 'Health', 11792890.68, 250), ('2025-01', 'Life', 11928938.56, 275), ('2025-02', 'Health', 10766094.12, 214), ('2025-02', 'Life', 10630039.11, 218), ('2025-03', 'Health', 12683356.8, 254), ('2025-03', 'Life', 12389776.31, 253)]

Expected output:
import streamlit as st
import pandas as pd
df = pd.DataFrame(query_results, columns=['date', 'category', 'premium_amount', 'claim_count'])
df['date'] = pd.to_datetime(df['date'])
premium_pivot = df.pivot(index='date', columns='category', values='premium_amount')
claim_pivot = df.pivot(index='date', columns='category', values='claim_count')
st.chat_message("assistant").write("📊 Relationship between premium amount and claim count per policy type across months.")
st.chat_message("assistant").line_chart(premium_pivot, use_container_width=True,color= "#0000FF")
st.chat_message("assistant").write("🟦 Premium Amount Trends (Health vs Life)")
st.chat_message("assistant").line_chart(claim_pivot, use_container_width=True,color="#FF0000")
st.chat_message("assistant").write("🟥 Claim Count Trends (Health vs Life)")

st.chat_message("assistant").write(""🔍 **Insights**:
- \nThe first chart shows trends in premium amounts collected for Health and Life policies monthly.
- \nThe second chart shows trends in claim counts per month for each policy type.
- \nLook for parallel spikes or dips in both charts to identify periods of correlation between premium inflows and claim volume"")

Example 4:

user_question = "Monthly Premium Revenue (Jan 2020 – Mar 2025)."
query_result = [
    ["2020-01", 19862592.47],
    ["2020-02", 19840587.76],
    ["2020-03", 21713621.32],
    # ... (add all months up to 2025-03)
    ["2025-03", 22161314.20],
]
Expected output:
query_result = [
    ["2020-01", 19862592.47],
    ["2020-02", 19840587.76],
    ["2020-03", 21713621.32],
    # ... (add all months up to 2025-03)
    ["2025-03", 22161314.20],
]

df = pd.DataFrame(query_results, columns=["month", "total_premium"])

st.chat_message("assistant").write("📈 Monthly Premium Revenue (Jan 2020 – Mar 2025)")
st.chat_message("assistant").line_chart(df.set_index("month")["total_premium"])
st.chat_message("assistant").write(""🔍 **Insights**: 
- \nThe line chart shows the monthly premium revenue from January 2020 to March 2025.,....)

Example 5:
user_question = "Show the percentage of prospects in each status category."
query_results = [[('Converted', 7459, 33.378082069181545),
                 ('Interested', 7387, 33.055891171074414),
                 ('Not Interested', 7501, 33.566026759744034)]
Expected output:
import pandas as pd
import streamlit as st
import altair as alt

query_results = [('Converted', 7459, 33.378082069181545),
                 ('Interested', 7387, 33.055891171074414),
                 ('Not Interested', 7501, 33.566026759744034)]

df = pd.DataFrame(query_results, columns=['Status', 'Count', 'Percentage'])

st.chat_message("assistant").write("📊 This chart displays the **percentage of prospects** in each status category: Converted, Interested, and Not Interested.")

st.chat_message("assistant").bar_chart(df.set_index('Status')[['Percentage']], color="#008000", height=300)

st.chat_message("assistant").write("🔍 **Insights:**")
st.chat_message("assistant").write("- The 'Converted' and 'Interested' categories show comparable percentage of prospects.")
st.chat_message("assistant").write("- The 'Not Interested' category has a slightly higher percentage compared to the others.")

st.chat_message("assistant").write("🟠 Here's a pie chart representation for better visualization of the part-to-whole relationships of the prospect statuses:")

st.chat_message("assistant").altair_chart(
    alt.Chart(df).mark_arc().encode(
        theta=alt.Theta('Percentage:Q'),
        color=alt.Color('Status:N', legend=None),
        tooltip=['Status', 'Percentage']
    ).properties(height=300, width=300)
)
st.chat_message("assistant").write("🔍 **Insights:**.....")






"""



    

        # print(response.code_execution_result)
    prompt_code2 = f"""
 You are a code generation assistant for data visualization using Streamlit.You are provided with:

### Objective
Generate efficient Streamlit code to visualize the data based on the analysis from Task 1.

### Input from task1
{task1_response} 
- Above structured output is from Task 1 containing:
    Selected chart type
    Rationale for chart selection
    Required data transformations
    Key insights to highlight
- Original `query_results` and `user_question`
        {query_results} // {user_question}

### Process
1. Parse the Task 1 output to extract chart type, transformations, and insights
2. Generate concise Python code that:
   - Starts with query_results variable from Task 1 output
   - Creates a DataFrame with `df = pd.DataFrame(query_results)`
   - Applies the data transformations identified in Task 1
   - round up the values to integer if needed
   - Creates the visualization using the chart type from Task 1
   - Adds explanatory text and insights

3. Format the output using Streamlit's `st.chat_message()` API:
   - Use `st.chat_message("assistant").write("...")` for descriptive text
   - Use `st.chat_message("assistant").write(""""🔍 Insights:\n..."""")` for key insights from Task 1
   - Use appropriate chart functions inside the chat message blocks

### Constraints
- No import statements or Streamlit boilerplate
- Response should be valid Python code
- Prioritize code brevity and directness
- Apply appropriate colors and styling to enhance readability
- Create multiple charts only if specified in Task 1 output
- Ensure the code handles the data structure properly to avoid errors
- Do not stop without completing the code generation

### Output
Complete, ready-to-run Python code that:
- Implements the visualization strategy from Task 1
- Transforms the data appropriately that 
- Presents the visualization(s)
- Provides the meaningful insights identified in Task 1
- Avoids any unnecessary comments or explanations in the code
----
## Quick Data Transformations

```python
# Convert dates to datetime if needed
df['date'] = pd.to_datetime(df['date']) # Change format if necessary
df['date'] = df['date'].dt.strftime('%Y-%m')

# Group by category
df_grouped = df.groupby('category')['value'].sum().reset_index()
df

# Pivot for multi-series charts
pivot_df = df.pivot(index='date', columns='category', values='value')

# Sort for better visualization
df_sorted = df.sort_values('value', ascending=False)
```

## Chat Message API Pattern
# Standard pattern
st.chat_message("assistant").write("Chart Title")
st.line_chart(df)
st.chat_message("assistant").write(""""🔍 Insights:\n - Key point 1\n- Key point 2"""")
```

## Common Pie Chart (with Altair)

```python
pie_chart = alt.Chart(df).mark_arc().encode(
    theta=alt.Theta('value:Q'),
    color=alt.Color('category:N')
).properties(height=300, width=300)

st.chat_message("assistant").altair_chart(pie_chart)

## Output is directly fed into the Streamlit app, so no need to add any boilerplate code or imports.
```


Example 1:
user_question = "Compare claims by status."
query_results = {{'status': ['Active', 'Cancelled', 'Expired', 'Pending'], 'count': [240, 95, 60, 30] }}

Expected output:
```python
# define query_results variable
query_results = {{ 'status': ['Active', 'Cancelled', 'Expired', 'Pending'], 'count': [240, 95, 60, 30] }}
## Curate the data for the chart....
df = pd.DataFrame(query_results)
st.chat_message("assistant").write("Claims by status.")
st.bar_chart(df.set_index('status'))
st.chat_message("assistant").write(""""Insights: \n Active policies have most claims.""""")
## Streamlit Code Generation Checklist
```
sample code for area chart
``` python
# Convert your list of tuples to a dictionary first
data = {
    'Date': [item[0] for item in query_results],
    'Approved Claim Amount': [item[1] for item in query_results]
}

# Then create the DataFrame from the dictionary
df = pd.DataFrame(data)

df['Date'] = pd.to_datetime(df['Date'])
df['Cumulative Approved Claim Amount'] = df['Approved Claim Amount'].cumsum().round(2)

chart = alt.Chart(df).mark_line().encode(
    x=alt.X('Date', axis=alt.Axis(format="%Y-%m")),
    y=alt.Y('Cumulative Approved Claim Amount', title='Cumulative Approved Claim Amount')
).properties(
    title='Cumulative Approved Claim Amount Over Time'
)

st.chat_message("assistant").altair_chart(chart, use_container_width=True)
```
1. ✓ Start with `query_results = {...}` and `df = pd.DataFrame(query_results)`
2. ✓ Transform data to fit chart's expected format
3. ✓ Use appropriate chart function from streamlit
4. ✓ Provide title and insights with `.write()`
5. ✓ Handle common data issues (dates, missing values)
6. ✓ Add relevant chart parameters only from streamlit library
7. ✓ Create multiple charts if needed for complete analysis
"""
    prompt_code= prompt_code2

    text = gemini_response(prompt_code)
    code = re.search(r"```python\n(.*?)```", text, re.DOTALL).group(1).strip() if re.search(r"```python\n(.*?)```", text, re.DOTALL) else text.strip()
    # query_results=pd.DataFrame(query_results) 
    return code
        

#     else:
#         return None
#     You are given a user question and a Pandas DataFrame df that contains the data queried from a SQL database in response to that question. Your task is to write a Python code snippet using Streamlit's st.chat_message() elements to generate and display an appropriate data visualization based on the user's question and the content of df.

# Do not write the entire Streamlit app, only the visualization snippet that will be called within the existing main() function. Use Streamlit-compatible plotting libraries such as matplotlib, plotly, or altair. Make sure to include:

# A short explanatory message shown using st.chat_message("assistant").write(...)

# The actual chart displayed with st.chat_message("assistant").plotly_chart(...) or similar
# Sensible plot styling, labels, and titles derived from the user’s intent

# Example input:
# user_question = "Can you show me the monthly sales trends for the last year?"
# # df has columns: ['month', 'sales']
# Your output should be only the code snippet (no explanation), and must work when placed inside a Streamlit app.
# for now assume the data as care and sales
# """
# print(response.text)
df = pd.DataFrame(
    [('2020', 'Health', 125298864.44), ('2020', 'Life', 127550278.01), ('2021', 'Health', 126836255.89), ('2021', 'Life', 126930973.19), ('2022', 'Health', 127877472.15), ('2022', 'Life', 128584311.82), ('2023', 'Life', 125993964.28), ('2023', 'Health', 128624004.99), ('2024', 'Health', 127665506.29), ('2024', 'Life', 129780281.56), ('2025', 'Life', 30681079.7), ('2025', 'Health', 31468283.15)]


)
uq="How do cumulative premium collections grow over the years for each policy type?"

print(get_code_response(df, uq))
#print(chart_type(uq,df))















# prompt_code = f"""

# I am building a Streamlit-based data visualization assistant that takes in a user's natural language question and a SQL query result (converted to a dictionary, then to a Pandas DataFrame named `df`). The assistant should:

# 1. Analyze the user's question and infer the best type of chart to use (bar, line, area, pie, scatter, map, or complex plot).
# 2. Display the chart using Streamlit’s `st.chat_message("assistant")` API.
# 3. Write an interpretation of the chart and highlight insights directly after the visualization using `st.chat_message("assistant").write(...)`.
# 4. Always construct the DataFrame from `query_results` with `df = pd.DataFrame(query_results)`.
# 5. Only output valid Python code — no imports or app boilerplate — and restrict all UI output to Streamlit's chat message format.

# The assistant must choose chart types based on the following logic:
# - Use bar charts to compare multiple categories
# - Use line or area charts to show trends over time
# - Use pie charts for part-to-whole relationships
# - Use scatter plots for correlation/distribution
# - Use maps for geographic data
# - Use Plotly or Matplotlib for complex/custom visualizations

# Please ensure:
# - All methods use required arguments
# - All outputs are complete Streamlit-compatible Python code blocks
# - The logic is accurate and context-sensitive


# Here's an example input you will be working with:

# query_results = {{
#     'status': ['Active', 'Cancelled', 'Expired', 'Pending'],
#     'count': [240, 95, 60, 30]
# }}
# user_question = "What’s the current distribution of policy statuses?"

# Here's the input you will be working with:
# query_results = {query_results}
# user_question = "{user_question}"
    


# """


    
# You are a code generation assistant for data visualization using Streamlit.

# You are provided with:
# - A natural language user question (`user_question`)
# - A SQL query result as a Python dictionary (`query_results`) representing tabular data
# - A Pandas DataFrame `df` created from `query_results`

# Your objective is to:
# 1. Analyze the user's question and the content of `df` to determine the most appropriate visualization.
# 2. Generate Python code using Streamlit's `st.chat_message()` API to:
#    - Display an appropriate chart
#    - Provide a written interpretation of the data
#    - Surface as many meaningful insights as possible
   
# chart types:
# - Use `st.bar_chart()` for bar charts
# - Use `st.line_chart()` for line charts
# - Use `st.area_chart()` for area charts
# - Use `st.scatter_chart()` for scatter plots
# - Use `st.map()` for geographic data
# - Use `st.altair_chart()` for pie charts
# - Use `st.plotly_chart()` for complex visualizations

# **Constraints:**
# - Output only valid Python code
# - Do not include any import statements or Streamlit boilerplate
# - Only use `st.chat_message("assistant")` to display all outputs (text and charts)

# **Visualization Strategy:**
# - Use these chart types based on the user's question and data characteristics:

#   | Use Case                             | Chart Type                             |
#   |--------------------------------------|-----------------------------------------|
#   | Comparing multiple categories        | `st.bar_chart()` or horizontal bar with `matplotlib` |
#   | Tracking trends over time            | `st.line_chart()` or `st.area_chart()`  |
#   | Showing part-to-whole relationships  | Pie chart using `altair_chart()`        |
#   | Identifying correlations             | `st.scatter_chart()`                    |
#   | Plotting distributions               | `st.scatter_chart()`                    |
#   | Showing geographic data              | `st.map()`                              |
#   | Complex or custom visualizations     | Use `matplotlib` inside `st.chat_message("assistant").pyplot()` |

# **Instructions for Output:**
# - Start by defining `query_results = {query_results}``
# - Create a DataFrame `df` from `query_results` using `pd.DataFrame(query_results)`
# - Use `st.chat_message("assistant").write(...)` to describe the chart purpose
# - Render the chart with the most appropriate Streamlit chart function
# - Write multiple insights and interpretations inside `st.chat_message("assistant").write(🔍 Insights:\n-...)`
# - Ensure the response contains **only valid Python code**, no explanations or comments outside code blocks

# **Input data:**
# query_results = {query_results}
# user_question = "{user_question}"


# 5. Provide insights based on the chart and data, using `st.chat_message("assistant").write(...)`
# Restrictions:
# - Do NOT include any import statements
# - Do NOT include app boilerplate
# - Do NOT write explanations outside code
# - Output only valid Python code
# -   Every Code you generate MUST  HAVE query_results as dataframe (df)
#     like this:
#     query_results = {{
#         'status': ['Active', 'Cancelled', 'Expired', 'Pending'],
#         'count': [240, 95, 60, 30]
#     }}

# ---

# ### Examples (Format Only)

# import streamlit as st
# import pandas as pd
# import altair as alt
# import plotly.express as px

# st.title("Insurance Data Dashboard")

# # 1. Bar Chart - Number of claims per insurance type
# st.header(" Claims per Insurance Type (Bar Chart)")
# claims_data = pd.DataFrame({{
#     'policy_type': ['Health', 'Life'],
#     'claim_count': [15255, 14888]
# }})
# st.bar_chart(claims_data.set_index('policy_type'))
# st.chat_message("assistant").write("🔍 Insights:\n- The bar chart shows the number of claims for each insurance type. \n- Health insurance has a higher number of claims compared to Life insurance, indicating a potential area for further analysis.")

# # 2. Line Chart - Monthly Premium Collection
# st.header(" Monthly Premium Collection Over Time (Line Chart)")
# monthly_premiums = {{
#     "month": [
#         "2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10", 
#         "2020-11", "2020-12", "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07", "2021-08",
#         "2021-09", "2021-10", "2021-11", "2021-12", "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06", 
#         "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04", 
#         "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", 
#         "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", 
#         "2025-01", "2025-02", "2025-03"
#     ],
#     "total_premium": [
#         19862592.47, 19840587.76, 21713621.32, 21463310.85, 20257168.63, 21875875.15, 23339236.71, 21282050.42, 19767997.99, 21173823.47,
#         19755181.6, 22517696.08, 21722684.21, 18399713.89, 21871540.75, 22022253.99, 21516168.29, 20098832.94, 22641886.9, 22052139.85, 21328148.02,
#         21293802.01, 20195419.59, 20624638.64, 21878663.42, 19498193.11, 21348553.47, 21058076.52, 21810605.48, 22163549.57, 21049918.11, 21059231.39,
#         21182680.14, 22283118.22, 20536882.71, 22592311.83, 20655721.51, 18981241.39, 21288332.85, 21447230.32, 22046877.63, 22420367.07, 21165170.91,
#         21164363.77, 20621280.37, 20680974.21, 22799899.39, 21346509.85, 23026441.75, 20192100.37, 23716018.23, 20485983.26, 20407968.77, 21690029.55,
#         21847001.16, 21296705.17, 20990693.63, 21209758.13, 21117891.82, 21465196.01, 20847738.16, 19140310.49, 22161314.2
#     ]
# }}
# premium_df = pd.DataFrame(monthly_premiums)
# premium_df['month'] = pd.to_datetime(premium_df['month'])
# premium_df = premium_df.sort_values('month')
# st.line_chart(premium_df.set_index('month')['total_premium'])
# st.chat_message("assistant").write("🔍 Insights:\n- The line chart shows the monthly premium collection over time. \n- There are noticeable peaks in premium collection during certain months, indicating potential seasonal trends or events that may have influenced premium payments.")

# # 3. scatter plot -
# # Scatter plot - Create a scatter plot that compares the number of claims 

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# # Sample data based on SQL query result
# data = pd.DataFrame({{
#     'policy_type': [, 'Life', 'Health'],
#     'claim_count': [2, 2, 2, 1, 1],
#     'premium_amount': [5189.86, 39558.06, 5189.86, 10892.22, 39558.06],
#     'month': ['2020-01', '2020-01', '2021-01', '2020-01', '2021-02']
# }})

# st.chat_message("assistant").write("Scatter plot showing claim counts vs premium amounts:")
# fig, ax = plt.subplots()
# scatter = ax.scatter(data['premium_amount'], data['claim_count'], c=data['policy_type'].apply(lambda x: 0 if x == 'Health' else 1), cmap='viridis')
# ax.set_xlabel("Premium Amount")
# ax.set_ylabel("Claim Count")
# ax.set_title("Claims vs Premium Amount")
# fig.tight_layout()

# # Color legend for policy type
# legend1 = ax.legend(*scatter.legend_elements(), title="Policy Type")
# ax.add_artist(legend1)

# st.chat_message("assistant").pyplot(fig)

# # Insights
# st.chat_message("assistant").write("🔍 Insights:\n- Health insurance generally shows more claims than Life insurance.\n- Premium amount doesn’t seem to directly correlate with claim count in this dataset.")


# # 4. Map - Insurance Claim Locations
# st.header(" Insurance Claim Locations (Map)")
# locations = pd.DataFrame({{
#     "city": ["Markchester", "Payneberg", "Nicolestad", "Suttonbury", "Maureenhaven"],
#     "country": ["United Kingdom"] * 5,
#     "postcode": ["IP97 9BW", "TF4 5AJ", "B74 3HW", "G1 1SY", "B45 1DY"]
# }})
# st.write("City Locations of Claims")
# st.table(locations)

# # 5. Area Chart - Monthly Claims Volume
# st.header(" Monthly Claims Volume (Area Chart)")
# area_df = {{
#     "month": [
#         "2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10", 
#         "2020-11", "2020-12", "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07", "2021-08",
#         "2021-09", "2021-10", "2021-11", "2021-12", "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06", 
#         "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04", 
#         "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", 
#         "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", 
#         "2025-01", "2025-02", "2025-03"
#     ],
#     "claim_count": [
#         531, 478, 464, 486, 468, 466, 488, 472, 446, 482, 527, 470, 471, 411, 483, 453, 489, 486, 504, 497, 474, 471, 444, 504,
#         468, 435, 477, 503, 518, 488, 526, 483, 470, 524, 461, 464, 486, 444, 490, 508, 466, 467, 443, 495, 500, 519, 475, 466,
#         514, 495, 474, 483, 452, 475, 472, 479, 445, 468, 492, 502, 494, 385, 472
#     ]
# }}
# area_chart_df = pd.DataFrame(area_df)
# area_chart_df['month'] = pd.to_datetime(area_chart_df['month'])
# area_chart_df = area_chart_df.sort_values('month')

# area_chart = alt.Chart(area_chart_df).mark_area(opacity=0.6).encode(
#     x='month:T',
#     y='claim_count:Q'
# ).properties(width=700, height=400)
# st.altair_chart(area_chart, use_container_width=True)
# st.chat_message("assistant").write("🔍 Insights:\n- The area chart shows the monthly claims volume over time. \n- There are noticeable peaks in claims during certain months, indicating potential seasonal trends or events that may have influenced claim submissions.")

# # 6. Custom Chart - 
# user : Monthly vs Quarterly Claims Comparison
# response:        st.header("Claims: Monthly vs Quarterly Comparison")
#         monthly = pd.DataFrame({{
#             "month": [
#                 "2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10", 
#                 "2020-11", "2020-12", "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07", "2021-08", 
#                 "2021-09", "2021-10", "2021-11", "2021-12", "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06", 
#                 "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04", 
#                 "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", 
#                 "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08, "2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03"
#             ],
#             "claim_count": [
#                 531, 478, 464, 486, 468, 466, 488, 472, 446, 482, 527, 470, 471, 411, 483, 453, 489, 486, 504, 497, 474, 471, 444, 504,
#                 468, 435, 477, 503, 518, 488, 526, 483, 470, 524, 461, 464, 486, 444, 490, 508, 466, 467, 443, 495, 500, 519, 475, 466,
#                 514, 495, 474, 483, 452, 475, 472, 479, 445, 468, 492, 502, 494, 385, 472
#             ]
#         }})
#         monthly['month'] = pd.to_datetime(monthly['month'])
#         monthly = monthly.sort_values('month')

#         quarterly = pd.DataFrame({{
#             "quarter": [
#                 "2020-Q", "2021-Q", "2022-Q", "2023-Q", "2024-Q", "2025-Q"
#             ],
#             "claim_count": [
#                 5778, 5687, 5817, 5759, 5751, 1351
#             ]
#         }})
#         quarterly['quarter'] = pd.to_datetime(quarterly['quarter'], format='%Y-Q')
#         quarterly = quarterly.sort_values('quarter')

#         # Combine the two datasets for comparison
#         combined_df = pd.merge(monthly, quarterly, left_on=monthly['month'].dt.year, right_on=quarterly['quarter'].dt.year, suffixes=('_monthly', '_quarterly'))

#         # Plot Monthly vs Quarterly Claims Comparison using Plotly
#         fig = px.scatter(combined_df, x='month', y='claim_count_monthly', title="Monthly Claims vs Quarterly Claims Comparison",
#                         labels=({{'month': 'Month', 'claim_count_monthly': 'Monthly Claims'}}, color='claim_count_quarterly')
#         st.plotly_chart(fig)

# # 7. Pie Chart - Claims per Policy Type
# st.header(" Claims Distribution by Policy Type (Pie Chart)")
# claims_pie_data = {{
#     "policy_type": ["Health", "Life"],
#     "claim_count": [26175, 26232]
# }}
# claims_pie_df = pd.DataFrame(claims_pie_data)
# fig_pie = px.pie(claims_pie_df, names='policy_type', values='claim_count', title="Claims Distribution by Policy Type")
# st.plotly_chart(fig_pie)
# st.chat_message("assistant").write("🔍 Insights:\n- The pie chart shows the distribution of claims between Health and Life insurance policies. \n- Health insurance claims are slightly higher than Life insurance claims, indicating a potential area for further analysis.")



# """