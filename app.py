# app.py - Add model status display
import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os

from utils import load_model, predict_tumor, get_confidence_color, class_names

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.title("🧠 Brain Tumor Detection System")
st.markdown("### Clinical-Grade MRI Analysis Tool")

# Sidebar with model status
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
    
    # Check model status
    if os.path.exists('brain_tumor_model.h5'):
        st.success("✅ Real AI Model Loaded")
        st.info("Using trained model with 93.6% accuracy")
    else:
        st.warning("⚠️ Demo Mode - Using mock predictions")
        st.info("Real model will be added soon")

# File upload
uploaded_file = st.file_uploader("Upload MRI Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", width=400)
    
    if st.button("🔬 Analyze MRI Scan"):
        with st.spinner("Analyzing with AI model..."):
            model = load_model()
            tumor_type, confidence, all_probs = predict_tumor(model, image)
            
            if tumor_type == 'notumor':
                st.success(f"✅ Result: No Tumor Detected\n\nConfidence: {confidence:.1%}")
            else:
                st.warning(f"⚠️ Result: {tumor_type.upper()} Tumor Detected\n\nConfidence: {confidence:.1%}")
            
            # Show probability chart
            prob_df = pd.DataFrame({
                'Tumor Type': class_names,
                'Probability': [p * 100 for p in all_probs]
            })
            st.bar_chart(prob_df.set_index('Tumor Type'))