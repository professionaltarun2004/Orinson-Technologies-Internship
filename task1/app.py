import joblib
import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

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

# Real-time data visualization with multiple plot options
if input_data:
    data_dict = {f"Feature {i+1}": [float(x)] for i, x in enumerate(input_data.split(","))}
    df = pd.DataFrame(data_dict)

    # Interactive Plot Type Selector
    st.sidebar.header("Choose Plot Type")
    plot_type = st.sidebar.selectbox("Select Visualization Type", ["Bar", "Line", "Scatter"])

    # Plotting according to selected type
    fig, ax = plt.subplots(figsize=(8, 5))
    if plot_type == "Bar":
        ax.bar(df.columns, df.iloc[0], color='skyblue', edgecolor='black')
        ax.set_title("Feature Values (Bar Chart)", fontsize=16, color='darkblue')
    elif plot_type == "Line":
        ax.plot(df.columns, df.iloc[0], color='purple', marker='o', linestyle='-', linewidth=2, markersize=8)
        ax.set_title("Feature Values (Line Plot)", fontsize=16, color='darkred')
    else:
        ax.scatter(df.columns, df.iloc[0], color='orange', s=100, edgecolors='black')
        ax.set_title("Feature Values (Scatter Plot)", fontsize=16, color='darkgreen')
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Values", fontsize=12)
    ax.grid(True)
    st.pyplot(fig)

# Prediction section with a loading spinner and displaying metrics
if input_data:
    with st.spinner('Predicting...'):
        try:
            data = [float(x) for x in input_data.split(",")]
            predictions = model.predict([data])
            st.success("Prediction successful!")
            st.header("Predictions")
            st.write(predictions)

            # Assuming that the model's labels are numeric (if not, modify for classification)
            true_labels = np.array([1])  # Just an example of ground truth, modify as per your dataset

            # Calculate Metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='binary', zero_division=1)
            recall = recall_score(true_labels, predictions, average='binary', zero_division=1)
            f1 = f1_score(true_labels, predictions, average='binary', zero_division=1)

            # Displaying the Metrics
            st.subheader("Model Evaluation Metrics")
            st.write(f"Accuracy: {accuracy:.4f}")
            st.write(f"Precision: {precision:.4f}")
            st.write(f"Recall: {recall:.4f}")
            st.write(f"F1 Score: {f1:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(true_labels, predictions)
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
            st.subheader("Confusion Matrix")
            cm_display.plot(cmap='Blues', ax=plt.gca())
            st.pyplot(fig)

        except ValueError:
            st.error("Invalid input format. Please ensure all inputs are numeric and separated by commas.")

# Footer with contact information and a call to action
st.markdown("---")
st.markdown("""
    **Contact Information**
    
    For further inquiries or support, please contact our team at [balivadatarun@gmail.com](mailto:balivadatarun@gmail.com).
""")
