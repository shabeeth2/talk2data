from google import genai
import os
google_api_key = os.getenv("GOOGLE_API_KEY")




client = genai.Client(api_key=google_api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works",
)
# t=client.models.generate_content(model=,contents=)

print(response.text)





query_results = [
    ['2020', 'Health', 1.252989e+08],
    ['2020', 'Life', 1.275503e+08],
    ['2021', 'Health', 1.268363e+08],
    ['2021', 'Life', 1.269310e+08],
    ['2022', 'Health', 1.278775e+08],
    ['2022', 'Life', 1.285843e+08],
    ['2023', 'Life', 1.259940e+08],
    ['2023', 'Health', 1.286240e+08],
    ['2024', 'Health', 1.276655e+08],
    ['2024', 'Life', 1.297803e+08],
    ['2025', 'Life', 3.068108e+07],
    ['2025', 'Health', 3.146828e+07]
]

df = pd.DataFrame(query_results, columns=['Year', 'Policy Type', 'Premium'])
df['Year'] = df['Year'].astype(int)
df['Premium'] = df['Premium'].round(2)

df['Cumulative Premium'] = df.groupby('Policy Type')['Premium'].cumsum()

chart = alt.Chart(df).mark_line().encode(
    x=alt.X('Year:O', title='Year'),
    y=alt.Y('Cumulative Premium:Q', title='Cumulative Premium Collections'),
    color='Policy Type:N'
).properties(
    title='Cumulative Premium Collections Over the Years for Each Policy Type'
)

st.chat_message("assistant").altair_chart(chart, use_container_width=True)

st.chat_message("assistant").write("🔍 Insights:")
st.chat_message("assistant").write("- Shows the cumulative growth of premium collections for each policy type over the years.")
PS D:\work\talk2data>  