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
                config = types.GenerateContentConfig(
                    tools=[types.Tool(
                        code_execution=types.ToolCodeExecution
                        )]
                    )
                )
        return response.executable_code


def get_code_response(query_results, user_question):

    prompt_code = f"""
You are a code generation assistant.

You are given:
- A user question (natural language)
- A Pandas DataFrame `df`, containing data queried from a SQL database

here is the data:
query_results= {query_results}
user_questions={user_question}

Your task:
1. Analyze the user's question and `df`
2. Choose the best chart type using Streamlit chart elements:
   - st.bar_chart(df)
   - st.line_chart(df)
   - st.scatter_chart(df)
   - st.map(df)
   - OR use st.chat_message("assistant").pyplot(...) with matplotlib for complex cases

3. Format your response as valid Python code:
   - Define `df` using pd.DataFrame with data provided
   - Use `st.chat_message("assistant").write(...)` to describe the chart
   - Render the chart using the correct Streamlit chart element

Restrictions:
- Do NOT include any import statements
- Do NOT include app boilerplate
- Do NOT write explanations outside code
- Output only valid Python code

---

### Examples (Format Only)

1. Bar Chart:
```python
df = pd.DataFrame({{
    'insurance_type': ['Health', 'Auto', 'Home', 'Life'],
    'claims': [320, 450, 210, 150]
}})
st.chat_message("assistant").write("Here are the claims per insurance type:")
st.bar_chart(df.set_index("insurance_type"))

2. Line Chart:
df = pd.DataFrame({{
    'month': ['Jan', 'Feb', 'Mar', 'Apr'],
    'premium': [10000, 12000, 11000, 14000]
}})
st.chat_message("assistant").write("Here’s the monthly premium collection trend:")
st.line_chart(df.set_index("month"))


3. Scatter Plot:
df = pd.DataFrame({{
    'age': [25, 30, 35, 40, 45],
    'claim_amount': [1000, 1500, 1800, 1700, 1600]
}})
st.chat_message("assistant").write("Scatter plot of age vs claim amount:")
st.scatter_chart(df)

4. Map:
df = pd.DataFrame({{
    'lat': [37.7749, 34.0522, 40.7128],
    'lon': [-122.4194, -118.2437, -74.0060]
}})
st.chat_message("assistant").write("Here are the locations of insurance claims:")
st.map(df)

5. area_chart:
df = pd.DataFrame({{
    'month': ['Jan', 'Feb', 'Mar', 'Apr'],
    'claims': [1000, 1200, 1100, 1300]
}})
st.chat_message("assistant").write("Here’s the area chart for monthly claims:")
st.area_chart(df.set_index("month"))

6.custom chart:
df = pd.DataFrame({{
    'month': ['Jan', 'Feb', 'Mar', 'Apr'],
    'monthly_claims': [1000, 1500, 1800, 1700],
    'quarterly_claims': [3000, 3000, 3000, 3000]
}})
st.chat_message("assistant").write("Monthly vs Quarterly Claims Comparison:")

fig, ax = plt.subplots()
ax.plot(df['month'], df['monthly_claims'], marker='o', label='Monthly')
ax.plot(df['month'], df['quarterly_claims'], marker='s', label='Quarterly')
ax.set_title("Claims Comparison")
ax.set_xlabel("Month")
ax.set_ylabel("Claims")
ax.legend()
fig.tight_layout()

st.chat_message("assistant").pyplot(fig)

"""



    

        # print(response.code_execution_result)
    code = gemini_response(prompt_code)
    # code = re.search(r"```python\n(.*?)```", text, re.DOTALL).group(1).strip() if re.search(r"```python\n(.*?)```", text, re.DOTALL) else text.strip()
        
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
    'month': [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ],
    'sales': [12000, 13500, 12800, 15000, 16000, 17000,
              16500, 17500, 16000, 18000, 19000, 20000]
})
print(get_code_response(df, "Monthly vs Quarterly Claims Comparison?"))

