# utils.py - Minimal version
class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']

def load_model():
    return None

def preprocess_image(image, target_size=(128, 128)):
    return None

def predict_tumor(model, image):
    # Mock prediction
    import random
    return random.choice(class_names), random.uniform(0.8, 0.95), [0.25, 0.25, 0.25, 0.25]

def get_confidence_color(confidence):
    if confidence >= 0.90:
        return "🟢"
    elif confidence >= 0.75:
        return "🟡"
    else:
        return "🔴"