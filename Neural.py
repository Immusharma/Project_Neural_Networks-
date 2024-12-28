import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import base64

# Function to set a background image
def add_background_image(image_file):
    try:
        with open(image_file, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{base64_image}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error("Background image file not found. Please check the path.")

# Add your background image
add_background_image('D:\\GUVI\\IITMDSA MDT34, MDT35,& MDT36\\python\\churnn.jpg')

# Rest of your code follows...
st.title("CHURN PREDICTION")

# Input number fields instead of sliders
p1 = st.number_input("**Enter how many days has the customer been buying books:**", min_value=0, max_value=1200, value=0)
p2 = st.number_input("**Enter how many days has the customer not returned to the shop:**", min_value=0, max_value=1200, value=0)

# Date input
selected_date = st.date_input("Select an order date")
if selected_date:
    p3 = selected_date.day
    p4 = selected_date.month
    p5 = selected_date.year

# Additional inputs with number input
p6 = st.number_input("**Enter the price of the book bought:**", min_value=0, max_value=500, value=0)
p7 = st.number_input("**Enter how many times the customer has ordered from your shop:**", min_value=0, max_value=500, value=8)

# Load the pre-trained model
try:
    model = load_model('D:\GUVI\IITMDSA MDT34, MDT35,& MDT36\python\model.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

# Prepare input data
input_data = np.array([[float(p2), float(p6), float(p1), float(p7), int(p3), int(p4), int(p5)]])

# Prediction button
if st.button('Predict') and model:
    pred = model.predict(input_data)
    predicted_class = np.argmax(pred, axis=1)[0]
    if predicted_class == 0:
        st.success("**The customer is still with us not Churn.**")
    elif predicted_class == 1:
        st.warning("**The customer has churned; let's try giving them a discount to bring them back.**")
