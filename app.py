import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="E-commerce Customer Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
)

# 2. Load Dataset with Caching (as required)
@st.cache_data
def load_data():
  return pd.read_csv("ecommerce_customers.csv")


df = load_data()

# 3. App Header & Description
st.title("🛍️ E-Commerce Customer Behavior & Purchase Patterns")
st.markdown(
    "This interactive dashboard analyzes customer engagement across mobile apps"
    " and websites, exploring what drives yearly spending."
)

# 4. Sidebar Filters (At least 2 filters required)
st.sidebar.header("🔍 Filter Dashboard")

# Filter 1: Slider for Length of Membership
min_mem = float(df["Length of Membership"].min())
max_mem = float(df["Length of Membership"].max())
selected_membership = st.sidebar.slider(
    "Select Minimum Membership Years",
    min_value=min_mem,
    max_value=max_mem,
    value=min_mem,
)

# Filter 2: Slider for Yearly Amount Spent
min_spend = float(df["Yearly Amount Spent"].min())
max_spend = float(df["Yearly Amount Spent"].max())
selected_spend = st.sidebar.slider(
    "Select Minimum Yearly Spent ($)",
    min_value=min_spend,
    max_value=max_spend,
    value=min_spend,
)

# Apply filters to dataframe
filtered_df = df[
    (df["Length of Membership"] >= selected_membership)
    & (df["Yearly Amount Spent"] >= selected_spend)
]

# 5. KPI Metrics (4 Key numbers on top)
st.markdown("### 📊 Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

with col1:
  st.metric(
      label="Total Customers", value=f"{filtered_df.shape[0]:,}"
  )
with col2:
  avg_spend = (
      filtered_df["Yearly Amount Spent"].mean()
      if not filtered_df.empty
      else 0
  )
  st.metric(label="Average Spend", value=f"${avg_spend:,.2f}")
with col3:
  avg_mem = (
      filtered_df["Length of Membership"].mean()
      if not filtered_df.empty
      else 0
  )
  st.metric(label="Avg Membership", value=f"{avg_mem:.1f} years")
with col4:
  max_spend_val = (
      filtered_df["Yearly Amount Spent"].max()
      if not filtered_df.empty
      else 0
  )
  st.metric(label="Highest Spend", value=f"${max_spend_val:,.2f}")

st.markdown("---")

# 6. Visualizations (At least 5 graphs including univariate, bivariate, and heatmap)
st.markdown("### 📈 Data Visualizations & Insights")

if filtered_df.empty:
  st.warning(
      "No data matches your filter criteria. Please adjust the sidebar"
      " sliders."
  )
else:
  # Graph 1: Univariate - Distribution of Yearly Amount Spent
  st.subheader("1. Univariate Analysis: Distribution of Yearly Amount Spent")
  fig1 = px.histogram(
      filtered_df,
      x="Yearly Amount Spent",
      nbins=30,
      title="Histogram of Yearly Amount Spent",
      color_discrete_sequence=["#636EFA"],
  )
  st.plotly_chart(fig1, use_container_width=True)

  # Graph 2: Bivariate - Time on App vs Yearly Amount Spent
  st.subheader("2. Bivariate Analysis: Time on App vs Yearly Amount Spent")
  fig2 = px.scatter(
      filtered_df,
      x="Time on App",
      y="Yearly Amount Spent",
      trendline="ols",
      title="Time on Mobile App vs. Yearly Spending",
      color_discrete_sequence=["#EF553B"],
  )
  st.plotly_chart(fig2, use_container_width=True)

  # Graph 3: Bivariate - Time on Website vs Yearly Amount Spent
  st.subheader("3. Bivariate Analysis: Time on Website vs Yearly Amount Spent")
  fig3 = px.scatter(
      filtered_df,
      x="Time on Website",
      y="Yearly Amount Spent",
      trendline="ols",
      title="Time on Website vs. Yearly Spending",
      color_discrete_sequence=["#00CC96"],
  )
  st.plotly_chart(fig3, use_container_width=True)

  # Graph 4: Custom Comparison - Length of Membership vs Yearly Amount Spent
  st.subheader("4. Customer Segment: Length of Membership vs Yearly Spending")
  fig4 = px.scatter(
      filtered_df,
      x="Length of Membership",
      y="Yearly Amount Spent",
      color="Yearly Amount Spent",
      title="Impact of Membership Tenure on Spending",
      color_continuous_scale="Viridis",
  )
  st.plotly_chart(fig4, use_container_width=True)

  # Graph 5: Correlation Heatmap
  st.subheader("5. Correlation Heatmap of Numeric Features")
  numeric_df = filtered_df[
      [
          "Avg. Session Length",
          "Time on App",
          "Time on Website",
          "Length of Membership",
          "Yearly Amount Spent",
      ]
  ]
  corr_matrix = numeric_df.corr().reset_index()
  corr_melted = corr_matrix.melt(id_vars="index")
  fig5 = px.imshow(
      numeric_df.corr(),
      text_auto=True,
      color_continuous_scale="RdBu_r",
      title="Correlation Matrix Heatmap",
  )
  st.plotly_chart(fig5, use_container_width=True)

# 7. Raw Data Checkbox
st.markdown("---")
if st.checkbox("Show Raw Dataset Table"):
  st.subheader("Raw Data Preview")
  st.dataframe(filtered_df)

# 8. Main Findings Section
st.markdown("---")
st.markdown("### 📝 Key Findings & Strategic Summary")
st.markdown("""
- **Primary Spending Driver:** *Length of Membership* shows the strongest positive relationship with yearly revenue. Long-term customers consistently spend significantly more.
- **Platform Performance:** *Time on App* exhibits a much stronger upward correlation with yearly spending than *Time on Website*, indicating that mobile app users are more valuable and engaged.
- **Actionable Suggestion:** The store should heavily invest in optimizing the mobile app experience, introducing loyalty tiers, and launching push-notification retention campaigns to nurture long-term membership value.
""")