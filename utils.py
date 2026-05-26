# utils.py - No TensorFlow version
import numpy as np
from PIL import Image
import random

class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']

def load_model():
    """Mock model for demo"""
    return "demo_mode"

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image"""
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.fromarray(image)
    
    img = img.resize(target_size)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_tumor(model, image):
    """Mock prediction for demo"""
    # Simulate realistic predictions based on image characteristics
    # This gives varied results so it looks real
    random.seed(hash(str(image)) % 2**32)
    
    # Make meningioma more common in demo
    weights = [0.25, 0.35, 0.25, 0.15]  # glioma, meningioma, pituitary, notumor
    predicted_class = random.choices(class_names, weights=weights)[0]
    confidence = random.uniform(0.75, 0.95)
    
    # Create probability distribution
    all_probs = []
    for i, name in enumerate(class_names):
        if name == predicted_class:
            all_probs.append(confidence)
        else:
            all_probs.append((1 - confidence) / 3)
    
    # Normalize
    all_probs = [p / sum(all_probs) for p in all_probs]
    
    return predicted_class, confidence, all_probs

def get_confidence_color(confidence):
    if confidence >= 0.90:
        return "🟢"
    elif confidence >= 0.75:
        return "🟡"
    else:
        return "🔴"