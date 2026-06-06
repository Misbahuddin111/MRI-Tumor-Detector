import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load your model
model_path = os.path.join(current_dir, 'brain_tumor_model.h5')

if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print(f"❌ Model not found at: {model_path}")
    model = None

# Class names (in order your model expects)
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']
CLASS_NAMES_DISPLAY = ['🧠 Glioma', '🧠 Meningioma', '🧠 Pituitary', '✅ No Tumor']

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype('uint8'))
    else:
        img = image
    
    # Resize to 128x128 (your model's input size)
    img = img.resize((128, 128))
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(image):
    """Make prediction on uploaded image"""
    if image is None:
        return "Please upload an image", None
    
    if model is None:
        return "❌ Model not loaded. Please check logs.", None
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        predicted_class = CLASS_NAMES[predicted_idx]
        
        # Create result dictionary for Gradio Label output
        result = {CLASS_NAMES_DISPLAY[i]: float(predictions[i]) for i in range(4)}
        
        # Format result message
        if predicted_class == 'notumor':
            message = f"✅ **Result: No Tumor Detected**\n\n**Confidence:** {confidence:.1%}\n\n**Recommendation:** Routine follow-up"
        else:
            message = f"⚠️ **Result: {predicted_class.upper()} Tumor Detected**\n\n**Confidence:** {confidence:.1%}\n\n**Recommendation:** Immediate consultation with neurologist"
        
        return message, result
        
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}", None

# Create the Gradio interface
with gr.Blocks(title="Brain Tumor Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Brain Tumor Detection System
    
    **Clinical-Grade AI for MRI Analysis | 93.6% Accuracy**
    
    Upload an MRI scan to detect and classify brain tumors.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="📤 Upload MRI Scan", type="numpy")
            submit_btn = gr.Button("🔬 Analyze MRI Scan", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_text = gr.Markdown(label="📊 Result", value="*Waiting for upload...*")
            output_labels = gr.Label(label="📈 Probability Distribution", num_top_classes=4)
    
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_text, output_labels]
    )
    
    gr.Markdown("""
    ---
    ### 📊 Model Performance Metrics
    
    | Tumor Type | Precision | Recall | F1-Score |
    |------------|----------|--------|----------|
    | **Glioma** | 91.82% | 96.19% | 94.0% |
    | **Meningioma** | 93.58% | 94.29% | 88.0% |
    | **Pituitary** | 91.23% | 99.05% | 95.0% |
    | **No Tumor** | 98.05% | 95.71% | 97.0% |
    
    **Overall Accuracy:** 93.57% | **Precision:** 93.67% | **Recall:** 93.57%
    
    ---
    ⚠️ **Medical Disclaimer:** This is a clinical decision support tool. All diagnoses must be confirmed by a qualified radiologist.
    
    🔬 Model trained on 5,600 MRI scans | EfficientNetB0 Architecture
    """)

# Launch the app
demo.launch()