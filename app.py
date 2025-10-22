from flask import Flask, request, jsonify
from PIL import Image
import torch
import io
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLOv5 model for food detection (or any other food detection model)
print("Loading YOLOv5 model for food detection...")
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Using the small model for faster processing
print("Model loaded successfully ✅")

# List of food classes (from YOLOv5 model)
food_classes = [
    "apple", "banana", "orange", "grape", "carrot", "broccoli", "pizza", "sandwich", "cake", "hot dog", "donut", "burger", 
    "fries", "cup", "bottle", "egg", "cheese", "chicken", "steak", "fish", "pasta", "sushi", "soup"
]

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
    
    # Convert image to a numpy array for processing by YOLOv5
    img = np.array(image)

    # Perform inference with YOLOv5 to detect objects
    results = model(img)

    # Parse results for food detection
    detected_foods = []
    for idx, (label, confidence) in enumerate(zip(results.names, results.xywh[0][:, -2])):
        # If the label is a food item and confidence is greater than a threshold (e.g., 0.5)
        if results.names[int(label)] in food_classes and confidence > 0.5:
            detected_foods.append(results.names[int(label)])

    # If no food is detected, return a generic suggestion
    has_food = len(detected_foods) > 0

    suggestion = (
        "🍽️ Eat something light and nutritious to keep focus while studying!"
        if has_food else
        "🚫 No recognizable food detected — maybe your fridge is empty?"
    )

    return jsonify({
        "detected_foods": detected_foods,
        "has_food": has_food,
        "suggestion": suggestion
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
