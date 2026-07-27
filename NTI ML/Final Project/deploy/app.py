
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders
gbc_model = joblib.load('gbc_model.pkl')
encoders = joblib.load('encoders.pkl')

# Define the order of AQI Buckets for consistency
aqi_order = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]

st.set_page_config(layout="wide")
st.title("India Air Quality Prediction")
st.markdown("This app predicts the AQI Bucket based on various environmental and time-series features.")

# Create input widgets for each feature
st.sidebar.header("Input Features")

def user_input_features():
    state = st.sidebar.selectbox('State', encoders['state'].classes_)
    pollutant_id = st.sidebar.selectbox('Pollutant ID', encoders['pollutant_id'].classes_)
    pollutant_min = st.sidebar.slider('Pollutant Min', 0.0, 500.0, 50.0)
    pollutant_max = st.sidebar.slider('Pollutant Max', 0.0, 500.0, 150.0)
    pollutant_avg = st.sidebar.slider('Pollutant Avg', 0.0, 500.0, 100.0)
    temperature_c = st.sidebar.slider('Temperature (°C)', 0.0, 50.0, 25.0)
    humidity_percent = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 70.0)
    wind_speed_kmh = st.sidebar.slider('Wind Speed (km/h)', 0.0, 50.0, 10.0)
    year = st.sidebar.slider('Year', 2025, 2030, 2027)
    month = st.sidebar.slider('Month', 1, 12, 1)
    day = st.sidebar.slider('Day', 1, 31, 15)
    hour = st.sidebar.slider('Hour', 0, 23, 12)
    day_of_week = st.sidebar.slider('Day of Week (0=Monday, 6=Sunday)', 0, 6, 3)
    season = st.sidebar.selectbox('Season', encoders['Season'].classes_)

    data = {
        'state': state,
        'pollutant_id': pollutant_id,
        'pollutant_min': pollutant_min,
        'pollutant_max': pollutant_max,
        'pollutant_avg': pollutant_avg,
        'Temperature_C': temperature_c,
        'Humidity_%': humidity_percent,
        'Wind_Speed_kmh': wind_speed_kmh,
        'Year': year,
        'Month': month,
        'Day': day,
        'Hour': hour,
        'Day_of_Week': day_of_week,
        'Season': season
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# Apply Label Encoding to user input
for col in ["state", "pollutant_id", "Season"]:
    if col in input_df.columns:
        # Ensure the input value is in the classes known by the encoder
        # If not, handle it (e.g., use a default, or raise an error)
        # For simplicity, we'll assume valid inputs based on selectbox options.
        encoded_value = encoders[col].transform([input_df[col].iloc[0]])[0]
        input_df[col] = encoded_value

# Make prediction
if st.sidebar.button('Predict AQI Bucket'):
    prediction = gbc_model.predict(input_df)
    prediction_proba = gbc_model.predict_proba(input_df)

    st.subheader('Prediction')
    predicted_bucket = prediction[0]
    st.success(f"The Predicted AQI Bucket is: **{predicted_bucket}**")

    st.subheader('Prediction Probability')
    proba_df = pd.DataFrame(prediction_proba, columns=gbc_model.classes_)
    proba_df = proba_df[aqi_order] # Order columns for better display
    st.bar_chart(proba_df.iloc[0])
    st.write(proba_df)

