import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

st.set_page_config(page_title="Fruit Freshness Detector", page_icon="🍎", layout="centered")
st.title("🍎 Fresh vs. Rotten Fruit Detector")
st.write("Fruit ki tasveer upload karein ya camera se photo khinch kar check karein ke woh fresh hai ya kharab.")

# Model aur Classes auto-load karna
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('fruit_freshness_model.h5')
    with open('classes.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, CLASS_NAMES = load_assets()

def predict_fruit(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)
    return predicted_class, confidence

tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Use Camera"])
image_to_predict = None

with tab1:
    uploaded_file = st.file_uploader("Gallery se image choose karein...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption="Uploaded Image", use_container_width=True)

with tab2:
    camera_file = st.camera_input("Camera se fruit ki photo lein")
    if camera_file is not None:
        image_to_predict = Image.open(camera_file)
        st.image(image_to_predict, caption="Captured Image", use_container_width=True)

if image_to_predict is not None:
    if st.button("Check Freshness 🔍"):
        with st.spinner("Analyze ho raha hai..."):
            label, confidence = predict_fruit(image_to_predict)
            
            st.subheader("Result:")
            if "fresh" in label.lower():
                st.success(f"✅ Yeh **FRESH** hai! ({label}) — Confidence: **{confidence}%**")
            else:
                st.error(f"❌ Yeh **ROTTEN (Kharab)** hai! ({label}) — Confidence: **{confidence}%**")
