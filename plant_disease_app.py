import streamlit as st
import numpy as np
import pickle
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ✅ Load model and label binarizer
model = load_model("cnn_model.h5")
label_binarizer = pickle.load(open("label_transform.pkl", "rb"))

# ✅ App UI
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection from Leaf Image")
st.write("Upload a leaf image to predict the disease.")

# ✅ File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ✅ Preprocessing function
def preprocess_image(image_data):
    image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# ✅ Prediction and display
if uploaded_file is not None:
    image_data = preprocess_image(uploaded_file)
    prediction = model.predict(image_data)
    predicted_label = label_binarizer.classes_[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.image(uploaded_file, caption='🖼️ Uploaded Leaf Image', use_column_width=True)
    st.success(f"✅ Predicted Disease: **{predicted_label}**")
    st.info(f"🔍 Confidence: {confidence:.2f}%")
