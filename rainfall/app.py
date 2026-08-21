import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Smart Crop Yield Predictor", page_icon="🌾", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("crop_yield_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("yield_df.csv")

model = load_model()
df = load_data()

st.title("🌾 Smart Crop Yield Predictor")
st.write("Predict crop yield using the trained Random Forest model.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    area = st.selectbox("Country / Area", sorted(df["Area"].dropna().unique()))
    item = st.selectbox("Crop", sorted(df["Item"].dropna().unique()))
    year = st.number_input("Year", min_value=int(df["Year"].min()), max_value=int(df["Year"].max()), value=int(df["Year"].max()), step=1)
with col2:
    rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0, value=float(df["average_rain_fall_mm_per_year"].median()))
    pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, value=float(df["pesticides_tonnes"].median()))
    avg_temp = st.number_input("Average Temperature (°C)", min_value=float(df["avg_temp"].min()), max_value=float(df["avg_temp"].max()), value=float(df["avg_temp"].median()))

if st.button("Predict Crop Yield", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        "Area": [area], "Item": [item], "Year": [year],
        "average_rain_fall_mm_per_year": [rainfall],
        "pesticides_tonnes": [pesticides], "avg_temp": [avg_temp]
    })
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Crop Yield: {prediction:,.2f} hg/ha")
    st.subheader("Input Summary")
    st.dataframe(input_data, use_container_width=True, hide_index=True)

st.divider()
st.caption("Machine Learning Bonus — Random Forest Crop Yield Prediction")
