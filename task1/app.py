import joblib
import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="ML Prediction App",
    page_icon=":factory:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for background and styling
st.markdown("""
    <style>
        body {
            background-image: url('"C:\Users\DELL\Downloads\aiimg.jpg"');
            background-size: cover;
            color: white;
        }
        .main {
            background: rgba(0, 0, 0, 0.6);
            padding: 10px;
            border-radius: 10px;
        }
        .sidebar .sidebar-content {
            background: rgba(0, 0, 0, 0.8);
        }
    </style>
""", unsafe_allow_html=True)

# Load model
model_path = 'model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
else:
    st.error(f"Model file not found at {model_path}. Please upload the model file.")

# Sidebar for navigation with widgets
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the options below to interact with the app:")

# Add interactivity for plot color choices
plot_color = st.sidebar.color_picker("Pick a color for plots", '#1f77b4')

# Add widgets for custom input
st.sidebar.subheader("Adjust Visualization Settings")
feature_count = st.sidebar.slider("Select Number of Features", min_value=2, max_value=10, value=5)

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

    # Display 6 types of plots
    st.subheader("Multiple Visualizations")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Bar Chart
    axes[0, 0].bar(df.columns, df.iloc[0], color=plot_color, edgecolor='black')
    axes[0, 0].set_title("Bar Chart", fontsize=14)
    axes[0, 0].set_xlabel("Features", fontsize=12)
    axes[0, 0].set_ylabel("Values", fontsize=12)

    # Line Plot
    axes[0, 1].plot(df.columns, df.iloc[0], color=plot_color, marker='o', linestyle='-', linewidth=2, markersize=8)
    axes[0, 1].set_title("Line Plot", fontsize=14)
    axes[0, 1].set_xlabel("Features", fontsize=12)
    axes[0, 1].set_ylabel("Values", fontsize=12)
    axes[0, 1].grid(True)

    # Scatter Plot
    axes[0, 2].scatter(df.columns, df.iloc[0], color=plot_color, s=100, edgecolors='black')
    axes[0, 2].set_title("Scatter Plot", fontsize=14)
    axes[0, 2].set_xlabel("Features", fontsize=12)
    axes[0, 2].set_ylabel("Values", fontsize=12)

    # Histogram
    axes[1, 0].hist(df.iloc[0], bins=5, color=plot_color, edgecolor='black')
    axes[1, 0].set_title("Histogram", fontsize=14)
    axes[1, 0].set_xlabel("Values", fontsize=12)
    axes[1, 0].set_ylabel("Frequency", fontsize=12)

    # Pie Chart
    axes[1, 1].pie(df.iloc[0], labels=df.columns, autopct='%1.1f%%', colors=[plot_color] * len(df.columns), startangle=90)
    axes[1, 1].set_title("Pie Chart", fontsize=14)

    # Box Plot
    axes[1, 2].boxplot(df.iloc[0], vert=False, patch_artist=True, boxprops=dict(facecolor=plot_color))
    axes[1, 2].set_title("Box Plot", fontsize=14)
    axes[1, 2].set_xlabel("Values", fontsize=12)

    # Adjust layout and show
    plt.tight_layout()
    st.pyplot(fig)

    # Data distribution chart
    st.header("Data Distribution")
    st.bar_chart(df.transpose())

# Footer with contact information and a call to action
st.markdown("---")
st.markdown("""
    **Contact Information**
    
    For further inquiries or support, please contact our team at [balivadatarun@gmail.com](mailto:balivadatarun@gmail.com).
""")
