import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from io import StringIO

# Load the trained model
model = joblib.load('LR_model.pkl')

# Set page configuration
st.set_page_config(page_title="Linear Regression Prediction App", layout="wide")

# Title
st.title("Linear Regression Prediction App")

# Input section
st.header("Input Data")
input_data = st.text_area(
    "Enter your data in CSV format:",
    placeholder="e.g., 1.23, 4.56, 7.89",
    height=100
)

# Prediction section
if input_data:
    try:
        # Convert input data to a DataFrame
        data = pd.read_csv(StringIO(input_data), header=None)
        
        # Preprocess the data
        scaler = StandardScaler()
        data.iloc[:, 1:4] = scaler.fit_transform(data.iloc[:, 1:4])
        
        # Make predictions
        predictions = model.predict(data)
        
        st.success("Prediction successful!")
        st.header("Predictions")
        st.write(predictions)
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("""
    **Note:** This app uses a pre-trained linear regression model to make predictions based on the input data.
""")