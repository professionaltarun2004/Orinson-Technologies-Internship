import streamlit as st
import pandas as pd
import joblib
import os
import folium
import plotly.express as px
from streamlit_folium import st_folium
from folium.plugins import HeatMap

# Page Configuration
st.set_page_config(
    page_title="🏠 High-End Real Estate Platform",
    layout="wide",
    page_icon="🏡",
)

# Custom CSS for a Premium Look
st.markdown("""
    <style>
        /* General Styling */
        body {
            background: linear-gradient(120deg, #e0eafc, #cfdef3);
            font-family: 'Poppins', sans-serif;
            color: #2c3e50;
        }
        h1, h2, h3 {
            font-weight: bold;
            color: #2c3e50;
        }
        h1 {
            font-size: 2.5em;
            text-align: center;
            margin-bottom: 20px;
        }
        h2 {
            color: #34495e;
            font-size: 1.8em;
        }
        h3 {
            color: #16a085;
            font-size: 1.5em;
        }
        .stButton>button {
            background: linear-gradient(45deg, #1abc9c, #16a085);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #16a085, #1abc9c);
            transform: translateY(-2px);
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        .stSidebar > div:first-child {
            background: linear-gradient(45deg, #1abc9c, #16a085);
            color: white;
            padding: 20px;
            border-radius: 10px;
            font-size: 18px;
            text-align: center;
        }
        .stMetric {
            background: #bdc3d9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            font-size: 20px;
        }
        /* Dataset Table Styling */
        .dataframe {
            margin: auto;
            border-collapse: collapse;
            font-size: 16px;
        }
        .dataframe th {
            background: #34495e;
            color: white;
            padding: 10px;
        }
        .dataframe td {
            background: #f7f9fc;
            color: #2c3e50;
            padding: 8px;
            border: 1px solid #ddd;
        }
        /* Footer Styling */
        footer {
            font-size: 14px;
            text-align: center;
            margin-top: 50px;
            color: #7f8c8d;
        }
        footer a {
            color: #16a085;
            text-decoration: none;
        }
        footer a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🏡 High-End Real Estate Price Prediction")
st.markdown("""
Welcome to the **Premium Real Estate Platform**. Use this tool to explore real estate trends in Hyderabad and predict house prices with advanced machine learning.
""")

# Load the Model
model_path = "house_price_model.pkl"
if os.path.exists(model_path):
    model_pipeline = joblib.load(model_path)
    st.success("✔️ Model loaded successfully!")
else:
    st.error(f"❌ Model file not found at {model_path}. Please upload the model file.")

# Sidebar Inputs
st.sidebar.header("🏠 Property Features")
def get_user_input():
    area = st.sidebar.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1500)
    location = st.sidebar.selectbox("Location", ["Nizampet", "Hitech City", "Manikonda", "Gachibowli", "Kukatpally"])
    bedrooms = st.sidebar.slider("Bedrooms", 1, 5, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
    parking = st.sidebar.slider("Parking Spaces", 0, 3, 1)
    age_of_property = st.sidebar.slider("Property Age (years)", 0, 50, 5)
    furnishing = st.sidebar.radio("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
    return pd.DataFrame({
        'Area': [area],
        'Location': [location],
        'No. of Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'CarParking': [parking],
        'Age_of_Property': [age_of_property],
        'Furnishing_Status': [furnishing]
    })

user_input = get_user_input()
st.subheader("📋 Input Features")
st.write(user_input)

# Align Input Features for Prediction
if 'Location' in user_input.columns:
    # One-hot encoding for categorical features
    user_input_encoded = pd.get_dummies(user_input, columns=['Location', 'Furnishing_Status'], drop_first=True)
    
    # Align with model's training features
    required_columns = model_pipeline.feature_names_in_
    for col in required_columns:
        if col not in user_input_encoded.columns:
            user_input_encoded[col] = 0
    user_input_encoded = user_input_encoded[required_columns]

    # Prediction
    try:
        prediction = model_pipeline.predict(user_input_encoded)[0]
        st.subheader("💰 Predicted House Price")
        st.metric(label="Estimated Price (₹)", value=f"{prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.error("Input data is not formatted correctly for prediction!")

# Load Dataset
csv_path = "Hyderabad.csv"
if os.path.exists(csv_path):
    dataset = pd.read_csv(csv_path)

    # Display Dataset
    st.subheader("📊 Real Estate Dataset")
    st.dataframe(dataset.head(10))

    # Heatmap Visualization
    st.subheader("🌏 Hyderabad Real Estate Heatmap")
    if 'Latitude' in dataset.columns and 'Longitude' in dataset.columns:
        map = folium.Map(location=[17.3850, 78.4867], zoom_start=11)
        heat_data = [[row['Latitude'], row['Longitude'], row['Price']] for index, row in dataset.iterrows()]
        HeatMap(heat_data, radius=15).add_to(map)
        st_folium(map, width=800, height=500)

    # Price Distribution
    st.subheader("📈 Price Distribution Across Locations")
    fig = px.box(
        dataset,
        x="Location",
        y="Price",
        color="Location",
        title="Interactive Price Distribution",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatterplot: Area vs. Price
    st.subheader("📊 Property Area vs Price with Detailed Hover Information")
    fig = px.scatter(
        dataset,
        x="Area",
        y="Price",
        color="Location",
        size="Price",
        hover_name="Location",
        hover_data=["Price", "Area", "No. of Bedrooms"],
        title="Interactive Scatterplot of Property Features",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Amenities Analysis
    st.subheader("🏢 Amenities Analysis")
    amenities = [
        "Gymnasium", "SwimmingPool", "JoggingTrack", "ClubHouse", "SportsFacility", 
        "24X7Security", "PowerBackup", "LiftAvailable"
    ]
    amenity_counts = {amenity: dataset[amenity].sum() for amenity in amenities if amenity in dataset.columns}
    amenity_df = pd.DataFrame(list(amenity_counts.items()), columns=["Amenity", "Count"])
    fig = px.bar(
        amenity_df,
        x="Amenity",
        y="Count",
        title="Count of Properties with Key Amenities",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error(f"Dataset not found at {csv_path}. Please upload the file.")

# Footer
st.markdown("""
    ---
    <footer>
        Developed with ❤️ by the High-End Real Estate Team. For inquiries, reach us at 
        <a href="mailto:balivadatarun@gmail.com">balivadatarun@gmail.com</a>.
    </footer>
""", unsafe_allow_html=True)
