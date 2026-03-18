import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import urllib.parse

load_dotenv()

st.set_page_config(page_title="Market Sizing App", layout="wide")

st.title("Market Sizing & Rankings")
st.write("Analyze estimated spending on digital products across US regions and metro markets.")

# 1. Database Connection Cache
@st.cache_resource
def get_db_connection():
    user = os.getenv('MYSQL_USER')
    password = os.getenv('MYSQL_PASSWORD')
    host = os.getenv('MYSQL_HOST')
    port = os.getenv('MYSQL_PORT')
    database = os.getenv('MYSQL_DB')

    safe_password = urllib.parse.quote_plus(password)

    db_url = f"mysql+pymysql://{user}:{safe_password}@{host}:{port}/{database}"
    engine = create_engine(db_url)
    return engine

# 2. Data Retrieval Cache
@st.cache_data
def query_data(query):
    try:
        engine = get_db_connection()
        df_result = pd.read_sql(query, con=engine)
        return df_result
    except Exception as e:
        st.error(f"Error in querying data: {e}")
        return None

# --- Main Execution ---
data_query = """
SELECT * FROM household_digital_products_expenditure
"""

df = query_data(data_query)

# Safety check: Stop the app gracefully if the database connection fails
if df is None:
    st.warning("Could not load data. Please check your database connection.")
    st.stop()

# --- THE FIX: Calculate the missing column ---
df["estimated_spend"] = df["cost"] * df["calibration_weight"]

# Sidebar filters
st.sidebar.header("Filters")

year_options = sorted(df["reference_year"].dropna().unique())
selected_year = st.sidebar.selectbox("Select Year", year_options)

category_options = sorted(df["product_category"].dropna().unique())
selected_category = st.sidebar.selectbox("Select Product Category", category_options)

region_options = sorted(df["region"].dropna().unique())
selected_regions = st.sidebar.multiselect("Select Region(s)", region_options, default=region_options)

filtered_df = df[
    (df["reference_year"] == selected_year) &
    (df["product_category"] == selected_category) &
    (df["region"].isin(selected_regions))
].copy()

# Top KPIs
st.subheader("Overview")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", len(filtered_df))
k2.metric("Unique CUs", filtered_df["newid"].nunique())
k3.metric("Raw Spend", f"${filtered_df['cost'].sum():,.0f}")
k4.metric("Estimated Spend", f"${filtered_df['estimated_spend'].sum():,.0f}")

# Region summary
st.subheader("Regional Market Sizing")

region_summary = (
    filtered_df.groupby("region", as_index=False)
    .agg(
        total_estimated_spend=("estimated_spend", "sum"),
        total_raw_spend=("cost", "sum"),
        unique_cus=("newid", "nunique")
    )
)

region_summary["spend_per_cu"] = (
        region_summary["total_estimated_spend"] / region_summary["unique_cus"]
)

region_summary = region_summary.sort_values("total_estimated_spend", ascending=False)

st.dataframe(region_summary, use_container_width=True)
st.bar_chart(region_summary.set_index("region")["total_estimated_spend"])

# Metro summary
st.subheader("Metro / PSU Rankings")

metro_summary = (
    filtered_df.groupby("psu", as_index=False)
    .agg(
        total_estimated_spend=("estimated_spend", "sum"),
        total_raw_spend=("cost", "sum"),
        unique_cus=("newid", "nunique")
    )
)

metro_summary["spend_per_cu"] = (
        metro_summary["total_estimated_spend"] / metro_summary["unique_cus"]
)

ranking_metric = st.radio(
    "Rank metros by:",
    ["total_estimated_spend", "spend_per_cu"],
    horizontal=True
)

top_n = st.selectbox("Top N markets", [5, 10, 25], index=1)

metro_ranked = metro_summary.sort_values(ranking_metric, ascending=False).reset_index(drop=True)
top_markets = metro_ranked.head(top_n)

st.dataframe(top_markets, use_container_width=True)
st.bar_chart(top_markets.set_index("psu")[ranking_metric])

# Concentration
st.subheader("Market Concentration")

def concentration_share(dataframe, n):
    total_spend = dataframe["total_estimated_spend"].sum()
    top_spend = dataframe.sort_values("total_estimated_spend", ascending=False).head(n)["total_estimated_spend"].sum()

    if total_spend == 0:
        return 0

    return top_spend / total_spend

top_5_share = concentration_share(metro_summary, 5)
top_10_share = concentration_share(metro_summary, 10)
top_25_share = concentration_share(metro_summary, 25)

c1, c2, c3 = st.columns(3)
c1.metric("Top 5 Share", f"{top_5_share:.1%}")
c2.metric("Top 10 Share", f"{top_10_share:.1%}")
c3.metric("Top 25 Share", f"{top_25_share:.1%}")

# Optional raw preview
with st.expander("Show filtered raw data"):
    st.dataframe(filtered_df, use_container_width=True)