import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
import folium
from streamlit_folium import st_folium

# Set page configuration
st.set_page_config(page_title="House Price Prediction", layout="wide")

# Load model
model_path = 'house_price_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
else:
    st.error(f"Model file not found at {model_path}. Please upload the model file.")

st.title("House Price Prediction App")
st.markdown("""
    Welcome to the House Price Prediction App. This tool allows you to input data and receive predictions based on our trained machine learning model.
    Please enter your data below.
""")

# Sidebar for input features
st.sidebar.header("Input Features")
st.sidebar.markdown("Please input the following features to get a prediction or select a location on the map.")

def get_user_input():
    area = st.sidebar.number_input("Area (in sq ft)", min_value=0, value=1000)
    location = st.sidebar.selectbox("Location", ["Nizampet", "Hitech City", "Manikonda", "Gachibowli", "Kukatpally"])
    bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
    parking = st.sidebar.number_input("Number of Parking Spaces", min_value=0, max_value=10, value=1)
    
    user_data = {
        'Area': area,
        'Location': location,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Parking': parking
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_input = get_user_input()
st.subheader("User Input Features")
st.write(user_input)

# Add a map of Hyderabad with click functionality
st.subheader("Hyderabad Map")
m = folium.Map(location=[17.3850, 78.4867], zoom_start=12)

# Add a click handler to the map
clicks = folium.LatLngPopup()
m.add_child(clicks)

# Display the map
map_data = st_folium(m, width=700, height=500)

# Update location based on map click
if map_data and map_data['last_clicked']:
    lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
    st.sidebar.write(f"Selected Location: Latitude {lat}, Longitude {lon}")
    # Here you can add logic to convert lat/lon to a specific location name if needed

# Convert categorical data to dummy variables
user_input_with_location = user_input.copy()  # Keep a copy with the original Location column
user_input = pd.get_dummies(user_input, columns=['Location'], drop_first=True)

# Ensure all expected columns are present
expected_columns = [
    'Area', 'Bedrooms', 'Bathrooms', 'Parking', 'Location_Hitech City', 'Location_Manikonda', 
    'Location_Gachibowli', 'Location_Kukatpally', '24X7Security', 'JoggingTrack', 'Wifi', 
    'LandscapedGardens', 'ClubHouse', 'ShoppingMall', 'SwimmingPool', 'Resale', 'Refrigerator', 
    'Gasconnection', 'MultipurposeRoom', 'Hospital', 'Wardrobe', 'VaastuCompliant', 'DiningTable', 
    'Sofa', 'MaintenanceStaff', 'School', 'AC', 'Gymnasium', 'Intercom', 'GolfCourse', 
    'SportsFacility', 'TV', 'Cafeteria', 'RainWaterHarvesting', 'No. of Bedrooms', 'LiftAvailable', 
    'BED', "Children'splayarea", 'StaffQuarter', 'IndoorGames', 'WashingMachine', 'CarParking', 
    'PowerBackup', 'ATM', 'Microwave', 'Location'
]
for col in expected_columns:
    if col not in user_input.columns:
        user_input[col] = 0

# Add the original Location column back
user_input['Location'] = user_input_with_location['Location']

# Predict house price
predictions = model.predict(user_input)
st.subheader("Predicted House Price")
st.write(f"₹ {predictions[0]:,.2f}")

# Visualize the input data
st.subheader("Input Data Visualization")
fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size for better readability
# Exclude the 'Location' column from the bar plot
user_input_for_plot = user_input.drop(columns=['Location'])
bars = ax.bar(user_input_for_plot.columns, user_input_for_plot.iloc[0], color=plt.cm.tab20.colors, width=0.6)
ax.set_title("Input Features", fontsize=18, fontweight='bold')
ax.set_ylabel("Values", fontsize=14)
ax.set_xlabel("Features", fontsize=14)
ax.set_xticklabels(user_input_for_plot.columns, rotation=45, ha='right', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=10)

# Add a legend
ax.legend(bars, user_input_for_plot.columns, title="Features", bbox_to_anchor=(1.05, 1), loc='upper left')

st.pyplot(fig)

# View CSV file
csv_path = "Hyderabad.csv"
if os.path.exists(csv_path):
    dataset = pd.read_csv(csv_path)
    st.subheader("House Prices Dataset")
    st.dataframe(dataset)
else:
    st.error(f"Dataset not found at {csv_path}. Please upload the file.")

# Footer with contact information
st.markdown("---")
st.markdown("""
    **Contact Information**
    
    For further inquiries or support, please contact our team at [balivadatarun@gmail.com](mailto:balivadatarun@gmail.com).
""")
