"""
AgriCure - Flask API Server
Serves the plant disease detection ML model

Install: pip install flask flask-cors torch torchvision pillow numpy
Run:     python server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import base64
import json
import io
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════
# DISEASE DATABASE
# (Treatments & Info for each class)
# ═══════════════════════════════
DISEASE_INFO = {
    "healthy": {
        "severity": 0,
        "description": "Your crop appears healthy with no visible signs of disease or pest damage.",
        "treatments": ["Continue regular monitoring", "Maintain proper irrigation", "Apply balanced fertilizer"],
        "prevention": "Keep up good farming practices — proper spacing, irrigation, and regular inspection."
    },
    "default": {
        "severity": 60,
        "description": "Disease detected on the crop. Immediate attention recommended.",
        "treatments": ["Consult local agricultural extension officer", "Apply appropriate fungicide/pesticide", "Remove and destroy infected plant parts"],
        "prevention": "Regular crop monitoring and maintaining proper plant hygiene can prevent spread."
    },
    # Specific disease info
    "Apple___Apple_scab": {
        "severity": 65, "common_name": "Apple Scab",
        "description": "Olive-green to black spots on leaves and fruit caused by Venturia inaequalis fungus. Leaves may yellow and drop prematurely.",
        "treatments": ["Apply fungicide (Captan or Mancozeb) at 7-10 day intervals", "Remove and destroy fallen infected leaves", "Prune trees for better air circulation"],
        "prevention": "Plant scab-resistant varieties. Apply dormant lime sulfur spray before bud break."
    },
    "Apple___Black_rot": {
        "severity": 75, "common_name": "Apple Black Rot",
        "description": "Brown to black rotting lesions on fruit, purple-bordered leaf spots. Caused by Botryosphaeria obtusa fungus.",
        "treatments": ["Remove mummified fruit and dead wood", "Apply captan-based fungicide", "Prune infected branches 15cm below visible infection"],
        "prevention": "Maintain orchard sanitation. Remove all mummified fruits before winter."
    },
    "Apple___Cedar_apple_rust": {
        "severity": 55, "common_name": "Cedar Apple Rust",
        "description": "Bright orange-yellow spots on upper leaf surfaces with tube-like structures beneath. Requires both apple and cedar trees to complete lifecycle.",
        "treatments": ["Apply myclobutanil or triadimefon fungicide", "Remove nearby juniper/cedar trees if possible", "Spray preventively from pink bud stage"],
        "prevention": "Plant rust-resistant apple varieties. Create distance from cedar/juniper trees."
    },
    "Corn_(maize)___Common_rust_": {
        "severity": 70, "common_name": "Corn Common Rust",
        "description": "Oval to elongated brick-red pustules on both leaf surfaces. Caused by Puccinia sorghi. Reduces photosynthesis and yield.",
        "treatments": ["Apply triazole fungicide (propiconazole) at early infection", "Use resistant hybrid varieties next season", "Apply foliar fungicide if >50% leaf area affected"],
        "prevention": "Plant rust-resistant hybrids. Early planting to avoid peak rust season."
    },
    "Tomato___Early_blight": {
        "severity": 65, "common_name": "Tomato Early Blight",
        "description": "Dark brown spots with concentric rings (target-board pattern) on lower leaves first. Caused by Alternaria solani fungus.",
        "treatments": ["Remove infected lower leaves immediately", "Apply copper-based or chlorothalonil fungicide every 7 days", "Stake plants to improve air circulation"],
        "prevention": "Mulch soil to prevent spore splash. Rotate crops — avoid tomatoes in same spot for 3 years."
    },
    "Tomato___Late_blight": {
        "severity": 90, "common_name": "Tomato Late Blight",
        "description": "Water-soaked pale green lesions that turn dark brown, often with white mold on leaf undersides. Caused by Phytophthora infestans — can destroy entire crop rapidly.",
        "treatments": ["Remove and destroy ALL infected plants immediately", "Apply mancozeb or chlorothalonil fungicide urgently", "Do not compost infected material — burn or bag it"],
        "prevention": "Plant resistant varieties. Avoid overhead watering. Monitor during cool, wet weather."
    },
    "Tomato___Leaf_Miner": {
        "severity": 50, "common_name": "Tomato Leaf Miner",
        "description": "Winding white trails (mines) on leaves caused by larvae of Tuta absoluta mining through leaf tissue.",
        "treatments": ["Apply spinosad or abamectin insecticide", "Use yellow sticky traps to monitor adults", "Remove heavily infested leaves"],
        "prevention": "Use insect-proof nets in nurseries. Introduce natural enemies like Nesidiocoris tenuis."
    },
    "Potato___Early_blight": {
        "severity": 60, "common_name": "Potato Early Blight",
        "description": "Dark brown circular spots with concentric rings on older leaves. Caused by Alternaria solani. Reduces tuber yield.",
        "treatments": ["Apply mancozeb or chlorothalonil fungicide", "Remove infected leaves and plant debris", "Ensure adequate potassium nutrition"],
        "prevention": "Use certified disease-free seed. Maintain 3-year crop rotation."
    },
    "Potato___Late_blight": {
        "severity": 95, "common_name": "Potato Late Blight",
        "description": "Dark water-soaked lesions on leaves and stems with white sporulation. Caused by Phytophthora infestans — the same pathogen that caused the Irish Famine.",
        "treatments": ["Apply metalaxyl + mancozeb immediately", "Remove and destroy infected plants completely", "Stop irrigation to slow spread"],
        "prevention": "Plant resistant varieties. Apply preventive fungicide during cool wet weather. Destroy volunteers."
    },
    "Rice___Brown_spot": {
        "severity": 65, "common_name": "Rice Brown Spot",
        "description": "Oval to circular brown spots with grey centers on leaves. Caused by Cochliobolus miyabeanus. Common in nutrient-deficient soils.",
        "treatments": ["Apply tricyclazole or edifenphos fungicide", "Apply potassium and silicon fertilizer", "Ensure proper water management"],
        "prevention": "Use resistant varieties. Maintain proper soil nutrition especially potassium."
    },
    "Wheat___Leaf_rust": {
        "severity": 70, "common_name": "Wheat Leaf Rust",
        "description": "Orange-brown pustules scattered on upper leaf surfaces. Caused by Puccinia triticina. Major cause of yield loss in wheat globally.",
        "treatments": ["Apply propiconazole or tebuconazole fungicide at first sign", "Use resistant wheat varieties", "Apply at flag leaf stage for best protection"],
        "prevention": "Grow rust-resistant varieties. Early planting. Monitor regularly during humid weather."
    }
}

# ═══════════════════════════════
# MODEL LOADING
# ═══════════════════════════════
model = None
classes = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    global model, classes
    
    model_path = './model/plant_disease_model.pth'
    classes_path = './classes.json'
    
    if not os.path.exists(model_path):
        print("⚠️  No trained model found. Using demo mode.")
        print("   Run 'python train_model.py' to train the model first.")
        return False
    
    # Load classes
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    
    # Rebuild model architecture
    model = models.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, len(classes))
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded! {len(classes)} classes, Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    return True

# ═══════════════════════════════
# IMAGE PREPROCESSING
# ═══════════════════════════════
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(base64_string):
    """Convert base64 image to tensor"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    tensor = preprocess(image).unsqueeze(0).to(device)
    return tensor

# ═══════════════════════════════
# PREDICTION
# ═══════════════════════════════
def predict(image_tensor):
    """Run inference and return top predictions"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        top3 = []
        for prob, idx in zip(top3_prob[0], top3_idx[0]):
            top3.append({
                'class': classes[idx.item()],
                'confidence': round(prob.item() * 100, 2)
            })
    
    return top3

def get_disease_info(class_name, confidence):
    """Get disease details from database"""
    # Check if it's healthy
    is_healthy = 'healthy' in class_name.lower()
    
    # Parse crop and disease from class name (format: Crop___Disease)
    parts = class_name.split('___')
    crop = parts[0].replace('_', ' ') if len(parts) > 0 else 'Unknown'
    disease_raw = parts[1].replace('_', ' ') if len(parts) > 1 else class_name
    
    # Look up in disease info database
    info = DISEASE_INFO.get(class_name, 
           DISEASE_INFO.get('healthy' if is_healthy else 'default'))
    
    disease_name = info.get('common_name', disease_raw)
    if is_healthy:
        disease_name = 'Healthy Crop'
    
    return {
        "isPlant": True,
        "disease": disease_name,
        "crop": crop,
        "severity": info['severity'] if not is_healthy else 0,
        "confidence": round(confidence, 1),
        "description": info['description'],
        "treatments": info['treatments'],
        "prevention": info['prevention'],
        "notPlantMessage": ""
    }

# ═══════════════════════════════
# DEMO MODE (when no model trained)
# ═══════════════════════════════
DEMO_RESULTS = [
    {
        "isPlant": True, "disease": "Leaf Rust", "crop": "Wheat",
        "severity": 72, "confidence": 88.5,
        "description": "Orange-brown pustules visible on upper leaf surfaces indicating wheat leaf rust infection at moderate-high severity. Immediate treatment recommended to prevent yield loss.",
        "treatments": ["Apply propiconazole fungicide immediately", "Remove severely infected leaves", "Ensure adequate spacing for air circulation"],
        "prevention": "Plant rust-resistant wheat varieties next season. Monitor weekly during humid weather.",
        "notPlantMessage": ""
    },
    {
        "isPlant": True, "disease": "Early Blight", "crop": "Tomato",
        "severity": 55, "confidence": 91.2,
        "description": "Dark brown spots with concentric ring pattern visible on lower leaves indicating early blight infection. Disease is at manageable stage.",
        "treatments": ["Apply copper-based fungicide every 7 days", "Remove infected lower leaves", "Improve plant spacing"],
        "prevention": "Mulch soil surface. Avoid overhead watering. Rotate crops every 3 years.",
        "notPlantMessage": ""
    }
]
demo_idx = 0

# ═══════════════════════════════
# API ROUTES
# ═══════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "classes": len(classes),
        "device": device
    })

@app.route('/predict', methods=['POST'])
def predict_disease():
    global demo_idx
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Demo mode if no model
        if model is None:
            result = DEMO_RESULTS[demo_idx % len(DEMO_RESULTS)]
            demo_idx += 1
            return jsonify(result)
        
        # Real prediction
        image_tensor = preprocess_image(data['image'])
        top3 = predict(image_tensor)
        
        best = top3[0]
        
        # Check if it's even a plant (low confidence = not a plant)
        if best['confidence'] < 30:
            return jsonify({
                "isPlant": False,
                "disease": "",
                "crop": "",
                "severity": 0,
                "confidence": best['confidence'],
                "description": "",
                "treatments": [],
                "prevention": "",
                "notPlantMessage": "This doesn't appear to be a plant or crop image. Please upload a clear photo of a plant leaf."
            })
        
        result = get_disease_info(best['class'], best['confidence'])
        result['top3'] = top3  # Include top 3 for transparency
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({"classes": classes, "count": len(classes)})

# ═══════════════════════════════
# START SERVER
# ═══════════════════════════════
if __name__ == '__main__':
    print("🌿 AgriCure ML Server Starting...")
    model_loaded = load_model()
    if not model_loaded:
        print("⚠️  Running in DEMO MODE")
        print("   Train the model first with: python train_model.py")
    print(f"\n🚀 Server running at http://localhost:5000")
    print(f"📡 Endpoints:")
    print(f"   GET  /health   - Server status")
    print(f"   POST /predict  - Analyze crop image")
    print(f"   GET  /classes  - List all disease classes\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
