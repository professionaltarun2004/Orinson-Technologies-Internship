import joblib
import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="ML Prediction App",
    page_icon=":factory:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load model
model_path = 'model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
else:
    st.error(f"Model file not found at {model_path}. Please upload the model file.")

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the options below to interact with the app:")

# Main title and description
st.title("Machine Learning Prediction App")
st.markdown("""
    Welcome to the ML Prediction App. This tool allows you to input data and receive predictions based on our trained machine learning model.
    Please enter your data in CSV format below.
""")

# Input section with a slider for better interaction
st.header("Input Data")
input_data = st.text_area(
    "Enter your data in CSV format (comma separated values):",
    placeholder="e.g., 1.23, 4.56, 7.89",
    height=100
)

# Prediction section with a loading spinner
if input_data:
    with st.spinner('Predicting...'):
        try:
            data = [float(x) for x in input_data.split(",")]
            predictions = model.predict([data])
            st.success("Prediction successful!")
            st.header("Predictions")
            st.write(predictions)
        except ValueError:
            st.error("Invalid input format. Please ensure all inputs are numeric and separated by commas.")

# Real-time data visualization with multiple plot options
st.header("Real-Time Data Visualization")
if input_data:
    data_dict = {f"Feature {i+1}": [float(x)] for i, x in enumerate(input_data.split(","))}
    df = pd.DataFrame(data_dict)

    # Interactive Plot Type Selector
    st.sidebar.header("Choose Plot Type")
    plot_type = st.sidebar.selectbox("Select Visualization Type", ["Bar", "Line", "Scatter"])

    # Improved Bar Chart
    if plot_type == "Bar":
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df.columns, df.iloc[0], color='skyblue', edgecolor='black')
        ax.set_title("Feature Values (Bar Chart)", fontsize=16, color='darkblue')
        ax.set_xlabel("Features", fontsize=12)
        ax.set_ylabel("Values", fontsize=12)
        st.pyplot(fig)

    # Improved Line Plot
    elif plot_type == "Line":
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df.columns, df.iloc[0], color='purple', marker='o', linestyle='-', linewidth=2, markersize=8)
        ax.set_title("Feature Values (Line Plot)", fontsize=16, color='darkred')
        ax.set_xlabel("Features", fontsize=12)
        ax.set_ylabel("Values", fontsize=12)
        ax.grid(True)
        st.pyplot(fig)

    # Improved Scatter Plot
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df.columns, df.iloc[0], color='orange', s=100, edgecolors='black')
        ax.set_title("Feature Values (Scatter Plot)", fontsize=16, color='darkgreen')
        ax.set_xlabel("Features", fontsize=12)
        ax.set_ylabel("Values", fontsize=12)
        st.pyplot(fig)

    # Adding a dynamic chart to show the data distribution
    st.header("Data Distribution")
    st.bar_chart(df.transpose())

# Footer with contact information and a call to action
st.markdown("---")
st.markdown("""
    **Contact Information**
    
    For further inquiries or support, please contact our team at [balivadatarun@gmail.com](mailto:balivadatarun@gmail.com).
""")

