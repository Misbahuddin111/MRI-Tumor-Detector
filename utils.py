# utils.py - COMPLETE FIXED VERSION
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']

def load_model():
    """Load the trained model from brain_tumor_model.h5"""
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible paths
    possible_paths = [
        os.path.join(current_dir, 'brain_tumor_model.h5'),  # Your actual model file
        os.path.join(current_dir, 'model.h5'),
        'brain_tumor_model.h5',
        'model.h5',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            try:
                model = tf.keras.models.load_model(path)
                print("Model loaded successfully!")
                return model
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
    
    # If we get here, no model was found
    print("\n" + "="*50)
    print("ERROR: Model file not found!")
    print(f"Current directory: {current_dir}")
    print("\nFiles in directory:")
    for file in os.listdir(current_dir):
        print(f"  - {file}")
    print("="*50)
    
    return None

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for prediction"""
    # Convert to RGB if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Handle different color modes
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 1:  # Single channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_tumor(model, image):
    """Make prediction"""
    if model is None:
        return None, 0.0, None
    
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    return class_names[predicted_class], confidence, predictions[0]

def get_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence >= 0.90:
        return "🟢"
    elif confidence >= 0.75:
        return "🟡"
    else:
        return "🔴"