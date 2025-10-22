from flask import Flask, request, jsonify
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch
import io

app = Flask(__name__)

MODEL_ID = "nateraw/food-classifier"

print("Loading food recognition model... 🍔")
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded successfully ✅")

@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to Smart Fridge AI API 🍎",
        "usage": "POST an image to /analyze as multipart/form-data with key 'image'"
    })

@app.route("/analyze", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

    # Preprocess
    inputs = extractor(images=image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    label = model.config.id2label[predicted_class_idx]
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()

    suggestion = f"🍽️ Looks like {label.lower()}! Perfect for a meal idea 😋" if confidence > 0.3 else \
                 "🤔 Not sure what food that is — maybe try another photo?"

    return jsonify({
        "label": label,
        "confidence": round(confidence, 3),
        "suggestion": suggestion
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
