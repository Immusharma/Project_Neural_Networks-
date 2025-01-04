import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import base64

# Function for background image
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

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Customer Retention Analysis"])

# Dashboard
if page == 'Dashboard':
    st.markdown("<h1 style='color:#1877F2; font-weight:bold;'>SMART BOOKSTORE - CUSTOMER CHURN PREDICTION 📈📖</h1>", unsafe_allow_html=True)
    st.subheader("Publishing industry data analyzed using advanced ANN Deep Learning techniques")
    st.markdown("<p style='color:darkred; font-weight:bold;'>*Dive into customer churn predictions ahead* 😎</p>", unsafe_allow_html=True)
    st.image("iStock.jpg")  

    # below the image text
    st.subheader("Publishing Industry Data Transformed into Insights")
    st.markdown("""
    - This app uses AI (ANN) to predict if customers might stop buying..
    - It helps understand customer habits and how to keep them coming back.
    - Go to the next page to find out which customers may leave.""")
    
# Retention Analysis Page
elif page == 'Customer Retention Analysis':
    
    # background image
    add_background_image("churnn.jpg")

    st.markdown('<h1 style="color:darkred; font-style: italic;">CHURN PREDICTION</h1>',unsafe_allow_html=True)
    # Inputs of sliders
    p1 = st.number_input("**Enter the total number of days the customer has been with your shop:**", min_value=0, max_value=1200, value=0)
    p2 = st.number_input("**Enter the number of days since the customer last visited:**", min_value=0, max_value=1200, value=0)
    selected_date = st.date_input("Select the date of the customer's last order")
    if selected_date:
        p3 = selected_date.day
        p4 = selected_date.month
        p5 = selected_date.year
    p6 = st.number_input("**Enter the price of the last book purchased:**", min_value=0, max_value=500, value=0)
    p7 = st.number_input("**Enter the total number of times the customer has placed an order:**", min_value=0, max_value=500, value=8)

    # Load pre-trained model
    try:
        model = load_model("model.h5")
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