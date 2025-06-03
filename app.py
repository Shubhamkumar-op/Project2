import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #4CAF50;
        margin-bottom: 10px;
    }
    .subtext {
        text-align: center;
        font-size: 1.2em;
        color: #555;
        margin-bottom: 30px;
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 30px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.06);
    }
    .prediction-title {
        font-weight: bold;
        font-size: 1.2em;
        color: #333333;
        margin-top: 10px;
    }
    .confidence {
        color: #2196F3;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌿 Xception Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Upload one or more plant images to classify their species.</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model\my_model_3.h5")

model = load_model()

def load_class_names(txt_path="name_of_spesies.txt"):
    class_names = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(')')
            if len(parts) == 2:
                class_names.append(parts[1].strip())
    return class_names

class_names = load_class_names()

# Upload files
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def preprocess_image(image):
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = preprocess_image(image)
        prediction = model.predict(img_array, verbose=0)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width=True)

        st.markdown(f'<div class="prediction-title">🔍 Prediction:</div>', unsafe_allow_html=True)
        st.write(f"**Predicted Species:** `{predicted_class}`")
        st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)

        st.markdown("### 🔢 Class Probabilities")
        prob_chart = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
        st.bar_chart(prob_chart)

        st.markdown('</div>', unsafe_allow_html=True)
