import joblib
import os
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import qrcode
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="ML Prediction App", layout="wide")

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

# Add widgets to the sidebar
st.sidebar.subheader("Customization")
plot_color = st.sidebar.color_picker("Pick a color for plots", '#1f77b4')
feature_count = st.sidebar.slider("Select Number of Features", min_value=2, max_value=10, value=5)
data_source = st.sidebar.selectbox("Choose Data Source", ["Upload CSV", "Enter Data Manually"])

# Scan QR code widget
st.sidebar.subheader("QR Code Generator")
website_url = "https://task1-ml-app.streamlit.app/#multiple-visualizations"
qr_button = st.sidebar.button("Generate QR Code")
if qr_button and website_url:
    qr = qrcode.make(website_url)
    buf = BytesIO()
    qr.save(buf)
    st.sidebar.image(buf)

# Removed QR Code section
# Developer Resume section
st.sidebar.subheader("Developer Information")
st.sidebar.markdown("""
    **Developer Resume**
    You can view the developer's resume in PDF format [here](https://drive.google.com/file/d/1e4vE9YccL0knn_tzh3U3LlXYo30nApnT/view?usp=drive_link).
""")

# Main title and description
st.title("Machine Learning Prediction App")
st.markdown("""
    Welcome to the ML Prediction App. This tool allows you to input data and receive predictions based on our trained machine learning model.
    Please enter your data in CSV format below.
""")

# Input section with a slider for better interaction
st.header("Input Data")
input_data = ""
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        input_data = uploaded_file.read().decode("utf-8")
elif data_source == "Enter Data Manually":
    input_data = st.text_area(
        "Enter your data in CSV format (comma separated values):",
        placeholder="e.g., 1.23, 4.56, 7.89",
        height=100
    )

# Prediction section with a loading spinner
if input_data:
    with st.spinner('Predicting...'):
        try:
            if data_source == "Upload CSV":
                df = pd.read_csv(StringIO(input_data))
                data = df.iloc[0].values
            else:
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
    if data_source == "Upload CSV":
        df = pd.read_csv(StringIO(input_data))
    else:
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

    # Add interactive operations for visualizations
    st.subheader("Perform Operations on Data")
    operation = st.selectbox(
        "Choose operation",
        ["Sum", "Max", "Min", "Mean", "Median", "Standard Deviation", "Variance", "Range", "Skewness", "Kurtosis"]
    )

    # Perform the selected operation
    if operation == "Sum":
        result = df.sum().iloc[0]
    elif operation == "Max":
        result = df.max().iloc[0]
    elif operation == "Min":
        result = df.min().iloc[0]
    elif operation == "Mean":
        result = df.mean().iloc[0]
    elif operation == "Median":
        result = df.median().iloc[0]
    elif operation == "Standard Deviation":
        result = df.std().iloc[0]
    elif operation == "Variance":
        result = df.var().iloc[0]
    elif operation == "Range":
        result = df.max().iloc[0] - df.min().iloc[0]
    elif operation == "Skewness":
        result = df.skew().iloc[0]
    elif operation == "Kurtosis":
        result = df.kurtosis().iloc[0]

    # Display the result
    st.write(f"Result of {operation}: {result}")

# Footer with contact information and developer resume link
st.markdown("---")
st.markdown("""
    **Contact Information**
    
    For further inquiries or support, please contact our team at [balivadatarun@gmail.com](mailto:balivadatarun@gmail.com).
    
    **Developer Resume**
    You can view the developer's resume in PDF format [here](path_to_resume.pdf).
""")
