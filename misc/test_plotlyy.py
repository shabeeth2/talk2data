import plotly.express as px
import pandas as pd
df = pd.DataFrame({
    'column_x': [1, 2, 3],
    'column_y': [4, 5, 6],
    'category': ['A', 'B', 'C'] 
})
user_question = "What is the distribution of customers across cities or counties??" 
if df is None or df.empty:
        print("⚠️ No data available for visualization")
else:
        # Data cleaning/transformation
        df_agg = df.groupby('category').size().reset_index(name='customer_count')
        df_agg.rename(columns={'category': 'city_county'}, inplace=True)

        print("📊 Customer Distribution by City/County")
        # Plotly chart generation and display using st.plotly_chart()
        fig = px.bar(df_agg, x='city_county', y='customer_count', title='Customer Distribution by City/County', labels={'city_county': 'City/County', 'customer_count': 'Number of Customers'})
        fig.show()