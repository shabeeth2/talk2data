from google.genai import types
import os
from google import genai
import pandas as pd
import matplotlib.pyplot as plt
import re
google_api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=google_api_key)
# model = genai.GenerativeModel(model_name="gemini-1.5-flash")
from google import genai
model_id = "gemini-1.5-flash"

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


def get_code_response(query_results, user_question):

    prompt_code = f"""
You are a code generation assistant for data visualization using Streamlit.

You are provided with:
- A natural language user question (`user_question`)
- A SQL query result as a Python dictionary (`query_results`) representing tabular data
- A Pandas DataFrame `df` created from `query_results`

Your objective is to:
1. Analyze the user's question and the content of `df` to determine the most appropriate visualization.
2. Generate Python code using Streamlit's `st.chat_message()` API to:
   - Display an appropriate chart
   - Provide a written interpretation of the data
   - Surface as many meaningful insights as possible

**Constraints:**
- Output only valid Python code
- Do not include any import statements or Streamlit boilerplate
- Only use `st.chat_message("assistant")` to display all outputs (text and charts)

**Visualization Strategy:**
- Use these chart types based on the user's question and data characteristics:

  | Use Case                             | Chart Type                             |
  |--------------------------------------|-----------------------------------------|
  | Comparing multiple categories        | `st.bar_chart()` or horizontal bar with `matplotlib` |
  | Tracking trends over time            | `st.line_chart()` or `st.area_chart()`  |
  | Showing part-to-whole relationships  | Pie chart using `altair_chart()`        |
  | Identifying correlations             | `st.scatter_chart()`                    |
  | Plotting distributions               | `st.scatter_chart()`                    |
  | Showing geographic data              | `st.map()`                              |
  | Complex or custom visualizations     | Use `matplotlib` inside `st.chat_message("assistant").pyplot()` |

**Instructions for Output:**
- Start by defining `df = pd.DataFrame(query_results)`
- Use `st.chat_message("assistant").write(...)` to describe the chart purpose
- Render the chart with the most appropriate Streamlit chart function
- Write multiple insights and interpretations using `st.chat_message("assistant").write(...)`
- Ensure the response contains **only valid Python code**, no explanations or comments outside code blocks

**Input Example:**
query_results = {query_results}
user_question = "{user_question}"


5. Provide insights based on the chart and data, using `st.chat_message("assistant").write(...)`
Restrictions:
- Do NOT include any import statements
- Do NOT include app boilerplate
- Do NOT write explanations outside code
- Output only valid Python code
-   Every Code you generate MUST  HAVE query_results as dataframe (df)
    like this:
    query_results = {{
        'status': ['Active', 'Cancelled', 'Expired', 'Pending'],
        'count': [240, 95, 60, 30]
    }}

---

### Examples (Format Only)

import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

st.title("Insurance Data Dashboard")

# 1. Bar Chart - Number of claims per insurance type
st.header(" Claims per Insurance Type (Bar Chart)")
claims_data = pd.DataFrame({{
    'policy_type': ['Health', 'Life'],
    'claim_count': [15255, 14888]
}})
st.bar_chart(claims_data.set_index('policy_type'))
st.chat_message("assistant").write("🔍 Insights:\n- The bar chart shows the number of claims for each insurance type. \n- Health insurance has a higher number of claims compared to Life insurance, indicating a potential area for further analysis.")

# 2. Line Chart - Monthly Premium Collection
st.header(" Monthly Premium Collection Over Time (Line Chart)")
monthly_premiums = {{
    "month": [
        "2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10", 
        "2020-11", "2020-12", "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07", "2021-08",
        "2021-09", "2021-10", "2021-11", "2021-12", "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06", 
        "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04", 
        "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", 
        "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", 
        "2025-01", "2025-02", "2025-03"
    ],
    "total_premium": [
        19862592.47, 19840587.76, 21713621.32, 21463310.85, 20257168.63, 21875875.15, 23339236.71, 21282050.42, 19767997.99, 21173823.47,
        19755181.6, 22517696.08, 21722684.21, 18399713.89, 21871540.75, 22022253.99, 21516168.29, 20098832.94, 22641886.9, 22052139.85, 21328148.02,
        21293802.01, 20195419.59, 20624638.64, 21878663.42, 19498193.11, 21348553.47, 21058076.52, 21810605.48, 22163549.57, 21049918.11, 21059231.39,
        21182680.14, 22283118.22, 20536882.71, 22592311.83, 20655721.51, 18981241.39, 21288332.85, 21447230.32, 22046877.63, 22420367.07, 21165170.91,
        21164363.77, 20621280.37, 20680974.21, 22799899.39, 21346509.85, 23026441.75, 20192100.37, 23716018.23, 20485983.26, 20407968.77, 21690029.55,
        21847001.16, 21296705.17, 20990693.63, 21209758.13, 21117891.82, 21465196.01, 20847738.16, 19140310.49, 22161314.2
    ]
}}
premium_df = pd.DataFrame(monthly_premiums)
premium_df['month'] = pd.to_datetime(premium_df['month'])
premium_df = premium_df.sort_values('month')
st.line_chart(premium_df.set_index('month')['total_premium'])
st.chat_message("assistant").write("🔍 Insights:\n- The line chart shows the monthly premium collection over time. \n- There are noticeable peaks in premium collection during certain months, indicating potential seasonal trends or events that may have influenced premium payments.")

# 3. scatter plot -
# Scatter plot - Create a scatter plot that compares the number of claims 

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sample data based on SQL query result
data = pd.DataFrame({{
    'policy_type': ['Health', 'Health', 'Health', 'Life', 'Health'],
    'claim_count': [2, 2, 2, 1, 1],
    'premium_amount': [5189.86, 39558.06, 5189.86, 10892.22, 39558.06],
    'month': ['2020-01', '2020-01', '2021-01', '2020-01', '2021-02']
}})

st.chat_message("assistant").write("Scatter plot showing claim counts vs premium amounts:")
fig, ax = plt.subplots()
scatter = ax.scatter(data['premium_amount'], data['claim_count'], c=data['policy_type'].apply(lambda x: 0 if x == 'Health' else 1), cmap='viridis')
ax.set_xlabel("Premium Amount")
ax.set_ylabel("Claim Count")
ax.set_title("Claims vs Premium Amount")
fig.tight_layout()

# Color legend for policy type
legend1 = ax.legend(*scatter.legend_elements(), title="Policy Type")
ax.add_artist(legend1)

st.chat_message("assistant").pyplot(fig)

# Insights
st.chat_message("assistant").write("🔍 Insights:\n- Health insurance generally shows more claims than Life insurance.\n- Premium amount doesn’t seem to directly correlate with claim count in this dataset.")


# 4. Map - Insurance Claim Locations
st.header(" Insurance Claim Locations (Map)")
locations = pd.DataFrame({{
    "city": ["Markchester", "Payneberg", "Nicolestad", "Suttonbury", "Maureenhaven"],
    "country": ["United Kingdom"] * 5,
    "postcode": ["IP97 9BW", "TF4 5AJ", "B74 3HW", "G1 1SY", "B45 1DY"]
}})
st.write("City Locations of Claims")
st.table(locations)

# 5. Area Chart - Monthly Claims Volume
st.header(" Monthly Claims Volume (Area Chart)")
area_df = {{
    "month": [
        "2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10", 
        "2020-11", "2020-12", "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07", "2021-08",
        "2021-09", "2021-10", "2021-11", "2021-12", "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06", 
        "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04", 
        "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", 
        "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", 
        "2025-01", "2025-02", "2025-03"
    ],
    "claim_count": [
        531, 478, 464, 486, 468, 466, 488, 472, 446, 482, 527, 470, 471, 411, 483, 453, 489, 486, 504, 497, 474, 471, 444, 504,
        468, 435, 477, 503, 518, 488, 526, 483, 470, 524, 461, 464, 486, 444, 490, 508, 466, 467, 443, 495, 500, 519, 475, 466,
        514, 495, 474, 483, 452, 475, 472, 479, 445, 468, 492, 502, 494, 385, 472
    ]
}}
area_chart_df = pd.DataFrame(area_df)
area_chart_df['month'] = pd.to_datetime(area_chart_df['month'])
area_chart_df = area_chart_df.sort_values('month')

area_chart = alt.Chart(area_chart_df).mark_area(opacity=0.6).encode(
    x='month:T',
    y='claim_count:Q'
).properties(width=700, height=400)
st.altair_chart(area_chart, use_container_width=True)
st.chat_message("assistant").write("🔍 Insights:\n- The area chart shows the monthly claims volume over time. \n- There are noticeable peaks in claims during certain months, indicating potential seasonal trends or events that may have influenced claim submissions.")

# 6. Custom Chart - Monthly vs Quarterly Claims Comparison
st.header("Claims: Monthly vs Quarterly Comparison")
monthly = pd.DataFrame({{
    "month": [
        "2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10", 
        "2020-11", "2020-12", "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07", "2021-08", 
        "2021-09", "2021-10", "2021-11", "2021-12", "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06", 
        "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04", 
        "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", 
        "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08, "2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03"
    ],
    "claim_count": [
        531, 478, 464, 486, 468, 466, 488, 472, 446, 482, 527, 470, 471, 411, 483, 453, 489, 486, 504, 497, 474, 471, 444, 504,
        468, 435, 477, 503, 518, 488, 526, 483, 470, 524, 461, 464, 486, 444, 490, 508, 466, 467, 443, 495, 500, 519, 475, 466,
        514, 495, 474, 483, 452, 475, 472, 479, 445, 468, 492, 502, 494, 385, 472
    ]
}})
monthly['month'] = pd.to_datetime(monthly['month'])
monthly = monthly.sort_values('month')

quarterly = pd.DataFrame({{
    "quarter": [
        "2020-Q", "2021-Q", "2022-Q", "2023-Q", "2024-Q", "2025-Q"
    ],
    "claim_count": [
        5778, 5687, 5817, 5759, 5751, 1351
    ]
}})
quarterly['quarter'] = pd.to_datetime(quarterly['quarter'], format='%Y-Q')
quarterly = quarterly.sort_values('quarter')

# Combine the two datasets for comparison
combined_df = pd.merge(monthly, quarterly, left_on=monthly['month'].dt.year, right_on=quarterly['quarter'].dt.year, suffixes=('_monthly', '_quarterly'))

# Plot Monthly vs Quarterly Claims Comparison using Plotly
fig = px.scatter(combined_df, x='month', y='claim_count_monthly', title="Monthly Claims vs Quarterly Claims Comparison",
                 labels=({{'month': 'Month', 'claim_count_monthly': 'Monthly Claims'}}, color='claim_count_quarterly')
st.plotly_chart(fig)

# 7. Pie Chart - Claims per Policy Type
st.header(" Claims Distribution by Policy Type (Pie Chart)")
claims_pie_data = {{
    "policy_type": ["Health", "Life"],
    "claim_count": [26175, 26232]
}}
claims_pie_df = pd.DataFrame(claims_pie_data)
fig_pie = px.pie(claims_pie_df, names='policy_type', values='claim_count', title="Claims Distribution by Policy Type")
st.plotly_chart(fig_pie)
st.chat_message("assistant").write("🔍 Insights:\n- The pie chart shows the distribution of claims between Health and Life insurance policies. \n- Health insurance claims are slightly higher than Life insurance claims, indicating a potential area for further analysis.")



"""



    

        # print(response.code_execution_result)
    text = gemini_response(prompt_code)
    code = re.search(r"```python\n(.*?)```", text, re.DOTALL).group(1).strip() if re.search(r"```python\n(.*?)```", text, re.DOTALL) else text.strip()
        
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
df = pd.DataFrame({
    'status': ['Active', 'Cancelled', 'Expired', 'Pending'],
    'count': [240, 95, 60, 30]
})

print(get_code_response(df, "Show distribution of policy statuses?"))

