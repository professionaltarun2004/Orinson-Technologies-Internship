import joblib
model = joblib.load('model.pkl')
def predict(input_data):
    # Assuming `input_data` is a list of features
    prediction = model.predict([input_data])
    return prediction


import streamlit as st
st.title("ML prediction app")
input_data=st.text_input("Enter your data in csv format")
if input_data:
    data=[float(x) for x in input_data.split(",")]
    predictions=model.predict([data])
    st.write("predictions : ",predictions)  