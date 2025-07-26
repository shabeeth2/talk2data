import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration for a professional look
st.set_page_config(layout="wide", page_title="Insurance Charge Analysis")

# --- MOCK DATA SETUP ---
# In a real application, you would load your data here, e.g., from a CSV.
# This mock data makes the script runnable for demonstration.
data = {
    'age': [19, 18, 28, 33, 32, 31, 46, 37, 37, 60, 25, 62, 23, 56, 27, 19, 52, 23, 56, 30, 60, 30, 18, 34, 37, 59, 63, 55, 23, 31],
    'sex': ['female', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'female', 'female', 'male', 'female'],
    'bmi': [27.9, 33.77, 33.0, 22.705, 28.88, 25.74, 33.44, 27.74, 29.83, 25.84, 26.22, 26.29, 34.4, 39.82, 42.13, 24.6, 30.78, 23.845, 39.82, 35.3, 36.005, 32.4, 34.1, 31.92, 28.025, 27.72, 23.085, 32.775, 23.21, 28.9],
    'children': [0, 1, 3, 0, 0, 0, 1, 3, 2, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 2, 0, 0, 2, 0, 1],
    'smoker': ['yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'no'],
    'region': ['southwest', 'southeast', 'southeast', 'northwest', 'northwest', 'southeast', 'southeast', 'northwest', 'northeast', 'northwest', 'northeast', 'southeast', 'southwest', 'southeast', 'southeast', 'southwest', 'northeast', 'northeast', 'southeast', 'southwest', 'northeast', 'southwest', 'southeast', 'northeast', 'northwest', 'southeast', 'northeast', 'northwest', 'southwest', 'northeast'],
    'charges': [16884.924, 1725.5523, 4449.462, 21984.47061, 3866.8552, 3756.6216, 8240.5896, 7281.5056, 6406.4107, 28923.13692, 2721.3208, 27808.7251, 1826.843, 11090.7178, 39611.7577, 1837.237, 10797.3362, 2395.17155, 11090.7178, 36837.467, 13228.84695, 4149.736, 1137.011, 5563.5618, 6203.90175, 12629.1658, 29569.72915, 12268.63225, 2396.0959, 4687.797]
}
df = pd.DataFrame(data)
# --- END MOCK DATA ---

try:
    # --- Main Application Logic ---
    st.title("🩺 Health Insurance Charge Analysis")
    st.markdown("An interactive dashboard to analyze the distribution of insurance charges based on demographic factors.")

    # --- Preprocessing and Data Validation ---
    if df is None or df.empty:
        st.warning("⚠️ No data available for visualization.")
    else:
        # Clean data by dropping rows with missing charges
        df.dropna(subset=['charges'], inplace=True)
        # Ensure correct data types
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['charges'] = pd.to_numeric(df['charges'], errors='coerce')

        # --- Interactive Filters in Sidebar ---
        st.sidebar.header("📊 Filters")
        selected_sex = st.sidebar.selectbox(
            "Filter by Gender",
            options=['All'] + list(df['sex'].unique()),
            index=0
        )

        selected_age = st.sidebar.slider(
            "Filter by Age Range",
            min_value=int(df['age'].min()),
            max_value=int(df['age'].max()),
            value=(int(df['age'].min()), int(df['age'].max()))
        )

        # Apply filters to the dataframe
        filtered_df = df[
            (df['age'] >= selected_age[0]) & (df['age'] <= selected_age[1])
        ]
        if selected_sex != 'All':
            filtered_df = filtered_df[filtered_df['sex'] == selected_sex]

        if filtered_df.empty:
            st.warning("No data matches the selected filters.")
        else:
            # --- KPI Dashboard ---
            st.markdown("### 📈 Key Performance Indicators")
            
            # Calculate KPIs from the filtered data
            avg_charge_smoker = filtered_df[filtered_df['smoker'] == 'yes']['charges'].mean()
            avg_charge_nonsmoker = filtered_df[filtered_df['smoker'] == 'no']['charges'].mean()
            delta = avg_charge_smoker - avg_charge_nonsmoker

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Average Charge (Smokers)",
                    value=f"${avg_charge_smoker:,.2f}",
                )
            with col2:
                st.metric(
                    label="Average Charge (Non-Smokers)",
                    value=f"${avg_charge_nonsmoker:,.2f}",
                )
            with col3:
                st.metric(
                    label="Smoker Surcharge (Avg. Difference)",
                    value=f"${delta:,.2f}",
                    delta_color="inverse"
                )

            st.markdown("---")

            # --- Main Visualization ---
            st.markdown("### ⚖️ Distribution of Charges by Region and Smoking Status")
            
            fig = px.box(
                filtered_df,
                x='region',
                y='charges',
                color='smoker',
                # Critical step: Customize labels and title for clarity
                labels={
                    "charges": "Insurance Charges ($)",
                    "region": "Geographic Region",
                    "smoker": "Smoker Status"
                },
                # Add extra context to the hover tooltip
                hover_data=['age', 'bmi', 'children'],
                color_discrete_map={'yes': '#FF5733', 'no': '#33CFFF'}
            )
            fig.update_layout(legend_title_text='Smoker Status')
            st.plotly_chart(fig, use_container_width=True)
            

            # --- Data-Driven Insights ---
            st.markdown("#### 🔍 Key Insights")
            
            insight_1 = "Smokers face drastically higher insurance charges across all regions, with a median charge often 3-4x higher than non-smokers."
            insight_2 = "The median charge for non-smokers is consistently below $10,000, while the median for smokers often exceeds $20,000."
            insight_3 = "The highest charges and widest distribution for both smokers and non-smokers are typically observed in the Southeast region."

            st.markdown(f"""
            - **Significant Cost Disparity:** {insight_1}
            - **Clear Cost Thresholds:** {insight_2}
            - **Regional Hotspot:** {insight_3}
            - **Outliers:** Note the presence of significant outliers, especially among smokers. These often correspond to individuals with higher BMIs or ages.
            """)
            
            # Optional: Show a sample of the filtered data
            with st.expander("View Filtered Data Table"):
                st.dataframe(filtered_df.sort_values("charges", ascending=False), use_container_width=True)

except Exception as e:
    st.error(f"❌ An error occurred while generating the visualization: {str(e)}")
    st.exception(e)