import joblib
import os
import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="ML Prediction App",
    page_icon=":factory:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load model
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'model.pkl')
model = joblib.load(model_path)

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the options below to interact with the app:")

# Main title and description
st.title("Machine Learning Prediction App")
st.markdown("""
    Welcome to the ML Prediction App. This tool allows you to input data and receive predictions based on our trained machine learning model.
    Please enter your data in CSV format below.
""")

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
        data = [float(x) for x in input_data.split(",")]
        predictions = model.predict([data])
        st.success("Prediction successful!")
        st.header("Predictions")
        st.write(predictions)
    except ValueError:
        st.error("Invalid input format. Please ensure all inputs are numeric and separated by commas.")

# Footer with contact information
st.markdown("---")
st.markdown("""
    **Contact Information**

    For further inquiries or support, please contact our team at [balivadatarun@gmail.com](mailto:balivadatarun@gmail.com).
""")
