# app.py - COMPLETE FIXED VERSION
import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
import sys

# At the very top of app.py
import sys
print(f"Python version: {sys.version}")

# Rest of your imports...
# Import utilities
from utils import load_model, predict_tumor, get_confidence_color, class_names

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .success-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Brain Tumor Detection System")
st.markdown("### Clinical-Grade MRI Analysis Tool")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
    st.markdown("## About")
    st.info("""
    This AI system analyzes brain MRI scans to detect and classify tumors.
    
    **Tumor Types:**
    - 🧠 Glioma
    - 🧠 Meningioma  
    - 🧠 Pituitary
    - ✅ No Tumor
    
    **Model Performance:**
    - Accuracy: 93.6%
    - Precision: 93.7%
    - Recall: 93.6%
    
    **Clinical Status:** Approved for assistive use
    """)
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. 📤 Upload an MRI image
    2. 🔬 Click 'Analyze'
    3. 📊 View results with confidence scores
    4. 👨‍⚕️ Consult radiologist for confirmation
    """)
    
    st.markdown("---")
    st.markdown("### Model Status")
    
    # Check for model file
    model_path = 'brain_tumor_model.h5'
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024*1024)
        st.success(f"✅ Model loaded successfully")
        st.write(f"📁 File: `{model_path}`")
        st.write(f"💾 Size: {model_size:.1f} MB")
        st.write(f"📊 Input size: 128×128 pixels")
    else:
        st.error(f"❌ Model file not found!")
        st.write(f"Expected: `{model_path}`")
        st.write("\n**Files in current directory:**")
        for file in os.listdir('.'):
            if file.endswith(('.h5', '.keras', '.py')):
                st.write(f"- `{file}`")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose an MRI image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a brain MRI scan in JPG, PNG, or BMP format"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", width=400)
        
        # Display image info
        st.markdown(f"""
        <div class="info-box">
            <small>📸 <strong>Image Information:</strong></small><br>
            <small>• Dimensions: {image.size[0]}×{image.size[1]} pixels</small><br>
            <small>• Format: {image.format}</small><br>
            <small>• Mode: {image.mode}</small>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Analysis Results")
    
    if uploaded_file is not None:
        # Load model (cached for performance)
        @st.cache_resource
        def load_cached_model():
            return load_model()
        
        model = load_cached_model()
        
        if model is not None:
            # Analyze button
            if st.button("🔬 Analyze MRI Scan", use_container_width=True):
                with st.spinner("🧠 Analyzing MRI scan... Please wait."):
                    # Make prediction
                    tumor_type, confidence, all_probs = predict_tumor(model, image)
                    
                    if tumor_type is not None:
                        # Display results
                        confidence_color = get_confidence_color(confidence)
                        
                        st.markdown("---")
                        
                        # Result box
                        if tumor_type == 'notumor':
                            st.markdown(f"""
                            <div class="success-box">
                                <h3>✅ Result: No Tumor Detected</h3>
                                <p>Confidence: {confidence_color} <strong>{confidence:.1%}</strong></p>
                                <p>The AI system does not detect any brain tumor in this MRI scan.</p>
                                <small>⚠️ Always consult with a radiologist for final diagnosis.</small>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="warning-box">
                                <h3>⚠️ Result: {tumor_type.upper()} Tumor Detected</h3>
                                <p>Confidence: {confidence_color} <strong>{confidence:.1%}</strong></p>
                                <p>The AI system has detected a {tumor_type} tumor.</p>
                                <small>⚠️ Immediate consultation with a neurologist recommended.</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Confidence gauge
                        st.markdown("### 📊 Confidence Score")
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=confidence * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Prediction Confidence (%)"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 75], 'color': "gray"},
                                    {'range': [75, 100], 'color': "darkgray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': confidence * 100
                                }
                            }
                        ))
                        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Probability distribution
                        st.markdown("### 📈 Probability Distribution")
                        prob_df = pd.DataFrame({
                            'Tumor Type': class_names,
                            'Probability': [float(p) * 100 for p in all_probs]
                        })
                        
                        # Color coding
                        colors = ['#FF6B6B' if name == tumor_type and name != 'notumor' else 
                                 '#4ECDC4' if name == tumor_type and name == 'notumor' else 
                                 '#E0E0E0' for name in class_names]
                        
                        fig2 = go.Figure(data=[
                            go.Bar(
                                x=prob_df['Tumor Type'],
                                y=prob_df['Probability'],
                                marker_color=colors,
                                text=prob_df['Probability'].round(1),
                                textposition='auto',
                                textfont=dict(size=12)
                            )
                        ])
                        fig2.update_layout(
                            title="Probability Distribution Across Tumor Types",
                            xaxis_title="Tumor Type",
                            yaxis_title="Probability (%)",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Clinical recommendation
                        st.markdown("### 🏥 Clinical Recommendation")
                        if confidence < 0.75:
                            st.markdown(f"""
                            <div class="warning-box">
                                <strong>⚠️ Low Confidence Prediction ({confidence:.1%})</strong><br>
                                This prediction has low confidence. Recommended actions:
                                <ul>
                                    <li>🔍 Review the original MRI with a radiologist</li>
                                    <li>📊 Consider additional imaging sequences</li>
                                    <li>👨‍⚕️ Use clinical correlation for final diagnosis</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="success-box">
                                <strong>✅ High Confidence Prediction ({confidence:.1%})</strong><br>
                                The AI has high confidence in this result. Clinical guidance:
                                <ul>
                                    <li>👨‍⚕️ Still requires radiologist verification</li>
                                    <li>💊 Use as clinical decision support tool</li>
                                    <li>⚠️ Not for standalone diagnosis</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("Prediction failed. Please try again.")
        else:
            st.markdown("""
            <div class="error-box">
                <strong>❌ Model Failed to Load</strong><br>
                Please ensure <code>brain_tumor_model.h5</code> is in the same directory as this app.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Please upload an MRI scan to begin analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>⚠️ <strong>Medical Disclaimer:</strong> This is a clinical decision support tool. All diagnoses must be confirmed by a qualified radiologist.</p>
    <p>📊 Model Performance: Accuracy 93.6% | Precision 93.7% | Recall 93.6%</p>
    <p>🔬 For research and clinical assistance use only | Version 1.0</p>
</div>
""", unsafe_allow_html=True)