import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="AI Smart Farm", page_icon="🌱", layout="wide")

# --- 2. LOGIC FUNCTIONS ---

@st.cache_resource
def load_assets():
    """Loads the AI model and the preprocessing tools."""
    # Load the TabNet model
    model = TabNetClassifier()
    model.load_model('crop_recommendation_model.zip')
    
    # Load the scaler and label encoder
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    
    return model, scaler, label_encoder

def get_weather(city_name, api_key):
    """Fetches real-time weather for irrigation advice."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
        res = requests.get(url).json()
        temp = res['main']['temp']
        hum = res['main']['humidity']
        desc = res['weather'][0]['description']
        return temp, hum, desc
    except:
        return None, None, None

def suggest_fertilizer(n, p, k):
    """Expert system for fertilizer based on NPK gaps."""
    advice = []
    # Using 0.3 as a threshold because data is normalized 0-1
    if n < 0.3: advice.append("Low Nitrogen: Apply Urea.")
    if p < 0.3: advice.append("Low Phosphorus: Apply DAP.")
    if k < 0.3: advice.append("Low Potassium: Apply MOP.")
    
    if not advice:
        return "✅ Soil nutrients are optimal for the recommended crop!"
    return " ".join(advice)

# --- 3. LOAD MODELS ---
try:
    model, scaler, label_encoder = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}. Ensure .zip and .joblib files are in the repository.")

# --- 4. SIDEBAR INPUTS ---
st.sidebar.header("📍 Soil & Environmental Data")
st.sidebar.write("Adjust the values to match your soil test report.")

n = st.sidebar.slider("Nitrogen (N)", 0.0, 1.0, 0.5)
p = st.sidebar.slider("Phosphorus (P)", 0.0, 1.0, 0.5)
k = st.sidebar.slider("Potassium (K)", 0.0, 1.0, 0.5)
temp_input = st.sidebar.slider("Temperature (°C)", 0.0, 1.0, 0.5)
hum_input = st.sidebar.slider("Humidity (%)", 0.0, 1.0, 0.5)
ph_input = st.sidebar.slider("Soil pH", 0.0, 1.0, 0.5)
rain_input = st.sidebar.slider("Rainfall (mm)", 0.0, 1.0, 0.5)

# --- 5. MAIN DASHBOARD ---
st.title("🌱 Intelligent Agriculture Decision Support System")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Crop Recommendation")
    if st.button("Predict Best Crop"):
        # Format input for prediction
        input_data = np.array([[n, p, k, temp_input, hum_input, ph_input, rain_input]])
        
        # Predict
        prediction = model.predict(input_data)
        crop_name = label_encoder.inverse_transform(prediction)[0]
        
        st.success(f"### Recommended Crop: **{crop_name.upper()}**")
        
        # Fertilizer Advice
        st.subheader("Fertilizer Advice")
        f_advice = suggest_fertilizer(n, p, k)
        st.info(f_advice)
        st.balloons()

with col2:
    st.header("2. Real-Time Irrigation")
    city = st.text_input("Enter your City Name", "London")
    
    api_key = ff693136f1d9a97f195bf7008361aaa1
    
    if st.button("Check Weather & Pump Status"):
        curr_temp, curr_hum, curr_desc = get_weather(city, api_key)
        
        if curr_temp is not None:
            st.write(f"**Current Weather in {city}:** {curr_desc.capitalize()}")
            c_temp_col, c_hum_col = st.columns(2)
            c_temp_col.metric("Temp", f"{curr_temp}°C")
            c_hum_col.metric("Humidity", f"{curr_hum}%")
            
            # Integrated Irrigation Logic
            st.subheader("Automated Pump Status")
            if "rain" in curr_desc.lower():
                st.warning("PUMP: OFF (Rainfall detected/predicted)")
            elif curr_temp > 30:
                st.error("PUMP: ON (High Temperature detected)")
            else:
                st.success("PUMP: OFF (Conditions are stable)")
        else:
            st.error("Invalid City or API Key. Please check settings.")

st.markdown("---")
st.caption("Powered by TabNet (Explainable AI) & OpenWeather API. Based on the Integrated Crop Recommendation Framework.")

