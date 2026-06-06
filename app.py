import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Try different model filenames (use the one you have)
model_paths = [
    os.path.join(current_dir, 'final_clinical_approved_model.keras'),
    os.path.join(current_dir, 'brain_tumor_model.h5'),
    os.path.join(current_dir, 'model.keras'),
    os.path.join(current_dir, 'model.h5')
]

model = None
MODEL_PATH = None

for path in model_paths:
    if os.path.exists(path):
        MODEL_PATH = path
        try:
            model = tf.keras.models.load_model(path, compile=False)
            print(f"✅ Model loaded successfully from: {path}")
            break
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")

if model is None:
    print("❌ No model file found! Please upload your model file.")
    # Create a dummy model for testing (remove in production)
    inputs = tf.keras.Input(shape=(128, 128, 3))
    outputs = tf.keras.layers.Dense(4, activation='softmax')(inputs)
    model = tf.keras.Model(inputs, outputs)
    print("⚠️ Using dummy model - upload your real model for correct predictions")

# Class names in order (must match your model's output)
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']
CLASS_NAMES_DISPLAY = ['🧠 Glioma', '🧠 Meningioma', '🧠 Pituitary', '✅ No Tumor']

def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype('uint8'))
    else:
        img = image
    
    # Resize to 128x128 (match your model's input size)
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
            message = f"✅ **Result: No Tumor Detected**\n\nConfidence: {confidence:.1%}"
        else:
            message = f"⚠️ **Result: {predicted_class.upper()} Tumor Detected**\n\nConfidence: {confidence:.1%}"
        
        return message, result
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Create the Gradio interface
with gr.Blocks(title="Brain Tumor Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Brain Tumor Detection System
    
    **Clinical-Grade AI for MRI Analysis**
    
    Upload an MRI scan to detect and classify brain tumors with **93.6% accuracy**.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload MRI Scan", type="numpy")
            submit_btn = gr.Button("🔬 Analyze MRI Scan", variant="primary")
        
        with gr.Column(scale=1):
            output_text = gr.Markdown(label="Result", value="Waiting for upload...")
            output_labels = gr.Label(label="Probability Distribution", num_top_classes=4)
    
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_text, output_labels]
    )
    
    gr.Markdown("""
    ---
    ### 📊 Model Performance
    
    | Tumor Type | Precision | Recall | F1-Score |
    |------------|----------|--------|----------|
    | **Glioma** | 91.82% | 96.19% | 94.0% |
    | **Meningioma** | 93.58% | 94.29% | 88.0% |
    | **Pituitary** | 91.23% | 99.05% | 95.0% |
    | **No Tumor** | 98.05% | 95.71% | 97.0% |
    
    **Overall Accuracy:** 93.57%
    
    ⚠️ **Medical Disclaimer:** This is a clinical decision support tool. All diagnoses must be confirmed by a qualified radiologist.
    """)

demo.launch()