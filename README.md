# Plant-Disease-Detection-with-keras

This project detects plant leaf diseases using a custom-trained Convolutional Neural Network (CNN). The model is trained on the PlantVillage dataset and is deployed via a simple Streamlit web application.

---

## 📌 Features

- 🔍 Classifies 15 types of plant leaf diseases
- 🧠 Trained CNN model using Keras/TensorFlow
- 📊 Evaluates using Accuracy, Precision, Recall, F1 Score
- 🖼️ Visualization of training curves, confusion matrix, misclassifications
- 💻 Streamlit-powered web interface for real-time predictions
- 🌍 Ready to deploy on Streamlit Cloud

---

## 🧠 Model Performance

| Metric        | Value   |
|---------------|---------|
| ✅ Training Accuracy  | 91.5%  |
| ✅ Validation Accuracy | 80.5%  |
| ✅ Test Accuracy       | 53.1%  |
| ✅ Precision           | 0.68   |
| ✅ Recall              | 0.53   |
| ✅ F1 Score            | 0.49   |

📋 **Sample Classification Report Output:**
Tomato__Target_Spot: Precision 0.62 | Recall 0.78 | F1-score 0.70
Tomato_healthy: Precision 0.98 | Recall 0.85 | F1-score 0.91
Tomato_YellowLeaf_Curl: Precision 1.00 | Recall 0.44 | F1-score 0.61

yaml
Copy
Edit

---

## 📊 Visual Outputs

### 📈 Accuracy & Loss Curves

![image](https://github.com/user-attachments/assets/1649a1b5-cf5f-4370-ae32-11ff03ccd643)

### 🔲 Confusion Matrix

![image](https://github.com/user-attachments/assets/a4bc44fe-9915-492b-bab5-387458209fd4)

---

## 💻 Streamlit Web App

The app allows users to upload a `.jpg/.png` leaf image and returns the predicted disease label.

### 🔧 Core UI Code (Snippet)
```python
model = load_model("cnn_model.h5")
label_binarizer = pickle.load(open("label_transform.pkl", "rb"))

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])
if uploaded_file:
    image_data = preprocess_image(uploaded_file)
    prediction = model.predict(image_data)
    predicted_label = label_binarizer.classes_[np.argmax(prediction)]
    st.success(f"Predicted Disease: {predicted_label}")
# 🖼️ Streamlit GUI Demo

# https://drive.google.com/file/d/1qy0lZJ9z6RZpomexkSLO2eHPWhFpYQoT/view?usp=sharing

🚀 How to Run Locally
bash
Copy
Edit
# 1. Activate your Anaconda environment
conda activate myenv

# 2. Navigate to project folder
cd "D:/ML PROJECTS/Plant Disease Detection with keras"


📁 Project Structure
bash
Copy
Edit
📦 Plant Disease Detection
├── plant_disease_app.py        # Streamlit frontend
├── cnn_model.h5
               # Trained Keras model
├── label_transform.pkl         # Encoded labels
├── plant_disease_training.ipynb  # (optional) model training notebook
├── README.md                   # Project description
✅ To-Do / Future Scope
 Improve model generalization on unseen data

 Add webcam-based prediction option

 Add multi-language support for local farmers

 Mobile deployment using TensorFlow Lite

👨‍💻 Developed By
V. Yuvan Krishnan
B.Tech, SRM Institute of Science & Technology
Project Duration: 2 Weeks
