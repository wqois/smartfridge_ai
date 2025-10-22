# app.py
from transformers import AutoProcessor, AutoModelForCausalLM
from flask import Flask, request, jsonify
from PIL import Image
import torch
import io

app = Flask(__name__)

# Load model and processor
MODEL_ID = "microsoft/Florence-2-base"

print("Loading Florence-2 model... (this may take a minute)")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
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

    # Florence prompt for captioning
    prompt = "<DETAILED_CAPTION>"

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Simple heuristic: detect if there’s food
    food_keywords = ["food", "fruit", "vegetable", "meat", "bottle", "egg", "bread", "milk", "cheese", "plate", "dish"]
    has_food = any(word in caption.lower() for word in food_keywords)

    suggestion = (
        "🍽️ Eat something light and nutritious to keep focus while studying!"
        if has_food else
        "🚫 No recognizable food detected — maybe your fridge is empty?"
    )

    return jsonify({
        "description": caption,
        "has_food": has_food,
        "suggestion": suggestion
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
