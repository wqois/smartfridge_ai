from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)

# Load food-tuned YOLOv8 model (change path if you have another)
MODEL_PATH = "TheTechMuseum/food-detection-yolov8"  # or "best.pt"
print("Loading food detection model... 🍕")
model = YOLO(MODEL_PATH)
print("Model loaded ✅")

@app.route("/")
def home():
    return jsonify({
        "message": "🍱 Smart Fridge AI – Food Detection API",
        "usage": "POST image to /analyze with key 'image'. Add ?annotated=1 to get an image with boxes."
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

    # Run YOLOv8 inference
    results = model(image)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [float(x.item()) for x in box.xyxy[0]]
            detections.append({
                "label": model.names[cls],
                "confidence": round(conf, 3),
                "bbox": xyxy
            })

    has_food = len(detections) > 0
    suggestion = "🍽️ Looks delicious!" if has_food else "🚫 No food detected."

    # If user wants annotated image
    if request.args.get("annotated") == "1":
        annotated_path = "/tmp/annotated.jpg"
        results[0].plot(line_width=2)  # draw bounding boxes
        Image.fromarray(results[0].plot()).save(annotated_path)
        return send_file(annotated_path, mimetype="image/jpeg")

    return jsonify({
        "detections": detections,
        "has_food": has_food,
        "suggestion": suggestion
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
