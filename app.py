import streamlit as st
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

# Define target columns
target_columns = ['bearings', 'exvalve', 'radiator', 'wpump']

# Load CatBoost models
catboost_models = {}
for target in target_columns:
    try:
        model = CatBoostClassifier()
        model.load_model(f'catboost_model_{target}.cbm')
        catboost_models[target] = model
    except Exception as e:
        st.error(f"Error loading model for {target}: {e}")

# Load Label Encoders
label_encoders = {}
for target in target_columns:
    try:
        with open(f'l_encoder_{target}.json', 'r') as f:
            data = json.load(f)
            le = LabelEncoder()
            le.classes_ = np.array(data['classes'])
            label_encoders[target] = le
    except Exception as e:
        st.error(f"Error loading label encoder for {target}: {e}")

# Define feature columns
feature_columns = ['noise_db', 'water_outlet_temp', 'water_flow']

# Function to make predictions
def predict(features):
    input_df = pd.DataFrame([features], columns=feature_columns)
    predictions = {}
    for target in target_columns:
        try:
            model = catboost_models[target]
            le = label_encoders[target]
            prediction_encoded = model.predict(input_df)
            prediction = le.inverse_transform(prediction_encoded)
            predictions[target] = prediction[0]
        except Exception as e:
            predictions[target] = f"Error predicting {target}: {e}"
    return predictions

# Streamlit UI
st.title('Predictive Maintenance Of Compressor (AC)')
image_path = "AC.jpg"
st.image(image_path, width=500)

st.write("Enter the features to get predictions:")

# Input fields with limits
water_outlet_temp = st.number_input('Water Outlet Temperature', min_value=76.9, max_value=173.0, format="%.1f")
noise_db = st.number_input('Noise DB', min_value=1400, max_value=19500, format="%d")
water_flow = st.number_input('Water Flow', min_value=13.2, max_value=93.5, format="%.1f")

# Button to make predictions
if st.button('Predict'):
    features = [water_outlet_temp, noise_db, water_flow]
    predictions = predict(features)
    for target, prediction in predictions.items():
        st.write(f"Prediction for {target}: {prediction}")
