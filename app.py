import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Florence-2 model endpoint
MODEL_URL = "https://api-inference.huggingface.co/models/microsoft/Florence-2-base-ft"
# Load your Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")  # set this in Render → Environment → Add variable

def analyze_image(image_bytes):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/jpeg"
    }
    try:
        r = requests.post(MODEL_URL, headers=headers, data=image_bytes, timeout=60)
        if r.status_code != 200:
            return None, f"Model error {r.status_code}: {r.text}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

def is_food_object(label):
    food_keywords = [
        "food","fruit","apple","banana","orange","bread","burger","pizza","sandwich",
        "salad","egg","milk","yogurt","chicken","fish","meat","steak","rice","noodle",
        "cake","ice cream","cookie","vegetable","tomato","potato","onion","carrot"
    ]
    return any(word in label.lower() for word in food_keywords)

def get_recommendation(food_list):
    recs = []
    for f in food_list:
        fl = f.lower()
        if any(x in fl for x in ["banana","berries","apple","fish","egg","nuts","milk","yogurt"]):
            recs.append(f"✅ {f}: Great for focus and memory.")
        elif any(x in fl for x in ["pizza","burger","cake","ice cream","fries","soda"]):
            recs.append(f"⚠️ {f}: Not ideal for studying.")
        else:
            recs.append(f"ℹ️ {f}: Neutral for studying.")
    return recs

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()
    data, err = analyze_image(image_bytes)
    if err:
        return jsonify({"error": err}), 500

    labels = []
    if isinstance(data, list):
        for d in data:
            if isinstance(d, dict) and "label" in d:
                labels.append(d["label"])
    elif isinstance(data, dict):
        if "objects" in data:
            labels = [obj.get("label", "") for obj in data["objects"]]
        elif "generated_text" in data:
            labels = [data["generated_text"]]

    foods = [l for l in labels if is_food_object(l)]

    if not foods:
        return jsonify({
            "message": "No food detected.",
            "predictions": labels,
            "recommendations": []
        })

    recs = get_recommendation(foods)
    return jsonify({
        "message": "Food detected successfully!",
        "predictions": foods,
        "recommendations": recs
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
