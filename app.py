import pandas as pd
import numpy as np
import streamlit as st
import joblib

cluster_centers = joblib.load("centers.pkl")
ohe = joblib.load("ohe.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("scaler.pkl")

cluster_names = {
    0: "Low-Spending Family Customers",
    1: "High-Value Partner Customers",
    2: "Low-Activity Solo Customers",
    3: "High-Value Solo Customers"
}

cluster_descriptions = {
    0: "Customers with relatively lower income and spending, mostly living with a partner.",
    
    1: "High-income and high-spending customers, mostly living with a partner and purchasing through multiple channels.",
    
    2: "Lower-income and low-spending customers, mostly living alone with relatively low purchase activity.",
    
    3: "High-income and high-spending customers, mostly living alone with strong purchase activity."
}


st.title("🛒 Smartcart Customer Segmentation")

st.write(
    "Enter customer details below to identify the customer segmant."
)

st.divider()


st.subheader("👤 Customer details")

col1,col2,col3 = st.columns(3)

with col1:
    education = st.selectbox(
        "Education",
        ["Graduate", "Postgraduate", "Undergraduate"]
    )

with col2:
    living_with = st.selectbox(
        "Living With",
        ["Alone","Partner"]
    )

with col3:
    age = st.number_input(
        "Age",
        min_value =0,
        value  = 30
    )

st.subheader("🛍️ Purchase Details")
col1 ,col2 ,col3 = st.columns(3)

with col1:
    income = st.number_input(
        "Income",
        min_value=0.0,
    
        value=50000.0
    )

with col2:
    recency = st.number_input(
        "Recency (Days)",
        min_value=0,
        value=30
    )

with col3:
    total_spending = st.number_input(
        "Total Spending",
        min_value=0.0,
        value=1000.0
    )


col1, col2, col3 = st.columns(3)

with col1:
    num_deals = st.number_input(
        "Deals Purchases",
        min_value=0,
        value=2
    )

with col2:
    num_web = st.number_input(
        "Web Purchases",
        min_value=0,
        value=3
    )

with col3:
    num_catalog = st.number_input(
        "Catalog Purchases",
        min_value=0,
        value=2
    )


col1, col2, col3 = st.columns(3)

with col1:
    num_store = st.number_input(
        "Store Purchases",
        min_value=0,
        value=5
    )

with col2:
    num_web_visits = st.number_input(
        "Web Visits / Month",
        min_value=0,
        value=5
    )

with col3:
    total_children = st.number_input(
        "Total Children",
        min_value=0,
        value=0
    )

st.subheader("📊 Customer Activity")

col1, col2, col3 = st.columns(3)

with col1:
    complain = st.selectbox(
        "Complaint",
        [0, 1]
    )

with col2:
    response = st.selectbox(
        "Response",
        [0, 1]
    )

with col3:
    tenure = st.number_input(
        "Customer Tenure (Days)",
        min_value=0,
        value=365
    )


st.divider()

input_data = pd.DataFrame({
    "Education":[education],
    "Income":[income],
    "Recency":[recency],
    "NumDealsPurchases":[num_deals],
    "NumWebPurchases":[num_web],
    "NumCatalogPurchases":[num_catalog],
    "NumStorePurchases":[num_store],
    "NumWebVisitsMonth":[num_web_visits],
    "Complain":[complain],
    "Response":[response],
    "Age":[age],
    "Customer_Tenure_Days":[tenure],
    "Total_spending":[total_spending],
    "Total_Children":[total_children],
    "Living_With":[living_with]


})

predict_button = st.button(
    "🔮 Predict Customer Segment",
    use_container_width=True # mtlb jitne space available hai poore le lo

)

if predict_button:

    cat_cols = ["Education", "Living_With"]

    enc_cols = ohe.transform(
        input_data[cat_cols]
    )

    enc_df = pd.DataFrame(
        enc_cols.toarray(),
        columns=ohe.get_feature_names_out(cat_cols),
        index=input_data.index
    )

    X = pd.concat(
        [
            input_data.drop(columns=cat_cols),
            enc_df
        ],
        axis=1
    )

    scaled_data = scaler.transform(X)

    new_pca = pca.transform(scaled_data)

    distances = {}

    for cluster, center in cluster_centers.items():

        distance = np.linalg.norm(
            new_pca[0] - center
        )

        distances[cluster] = distance

    predicted_cluster = min(
        distances,
        key=distances.get
    )

    segment_name = cluster_names[predicted_cluster]

    segment_description = cluster_descriptions[predicted_cluster]


    st.success(
        f"🎯 Customer Segment: {segment_name}"
    )

    st.info(
        segment_description

    )

    st.write(
        f"**Cluster Number:** {predicted_cluster}"
    )

