import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
MODEL_PATH = "lumpy_skin_cattle_detector_transfer.h5"
model = load_model(MODEL_PATH)

# Function to preprocess and predict an image
def predict_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make a prediction
    prediction = model.predict(img)[0][0]
    
    # Interpret the result
    if prediction < 0.5:
        return "❌ Lumpy Skin Disease Detected"
    else:
        return "✅ Healthy Cattle (Normal Skin)"

# Streamlit UI
st.title("🐄 Lumpy Skin Disease Detector")
st.write("Upload an image of cattle, and the model will predict if it has Lumpy Skin Disease.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    if st.button("Predict"):
        prediction = predict_image(image)
        st.write(f"**Result:** {prediction}")

# User Feedback for Model Improvement
st.subheader("Help Improve the Model")
st.write("If the model is incorrect, you can provide the correct label.")

label = st.radio("What is the correct label?", ["Healthy", "Lumpy Skin Disease"])
if st.button("Submit Label"):
    feedback_dir = "feedback/"
    os.makedirs(feedback_dir, exist_ok=True)
    
    # Save the labeled image for future training
    image_path = os.path.join(feedback_dir, f"{label}_{uploaded_file.name}")
    image.save(image_path)
    
    st.success("Thank you! Your image has been saved for future training.")
