import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

# 1. SET UP THE PAGE
st.set_page_config(page_title="AI Smart Farm", layout="centered")
st.title("🌱 AI-Driven Crop Recommendation System")
st.write("This system uses the SwiFT/TabNet model from the research paper to suggest the best crop for your soil.")

# 2. LOAD THE BRAIN (The files you downloaded from Colab)
@st.cache_resource
def load_models():
    # Load the trained AI
    model = TabNetClassifier()
    model.load_model('crop_recommendation_model.zip')
    
    # Load the preprocessing tools
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    
    return model, scaler, label_encoder

model, scaler, label_encoder = load_models()

# 3. USER INPUTS (Sliders for the farmer)
st.sidebar.header("Enter Soil & Climate Details")
n = st.sidebar.slider("Nitrogen (N)", 0, 140, 40)
p = st.sidebar.slider("Phosphorus (P)", 5, 145, 40)
k = st.sidebar.slider("Potassium (K)", 5, 205, 40)
temp = st.sidebar.number_input("Temperature (°C)", 10.0, 50.0, 25.0)
hum = st.sidebar.number_input("Humidity (%)", 10.0, 100.0, 70.0)
ph = st.sidebar.slider("Soil pH", 3.0, 10.0, 6.5)
rain = st.sidebar.number_input("Rainfall (mm)", 20.0, 300.0, 100.0)

# 4. PREDICTION LOGIC
if st.button("Recommend Best Crop"):
    # Format the input for the AI
    input_data = np.array([[n, p, k, temp, hum, ph, rain]])
    
    # IMPORTANT: Scale the input just like we did in training!
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    crop_name = label_encoder.inverse_transform(prediction)[0]
    
    # Show Result
    st.success(f"### The best crop for your soil is: **{crop_name.upper()}**")
    st.balloons()

st.info("Note: This recommendation is based on Explainable AI (XAI) models.")