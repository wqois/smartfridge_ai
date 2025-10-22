import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------
# Public model on Hugging Face (detects real food classes)
MODEL_URL = "https://api-inference.huggingface.co/models/valentinafeve/food-image-classification"
HF_TOKEN = os.environ.get("HF_TOKEN")  # Optional Hugging Face token

# ------------------------------------------------------
# HELPER: send image to Hugging Face
# ------------------------------------------------------
def analyze_image(image_bytes):
    headers = {
        "Content-Type": "image/jpeg"
    }
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        response = requests.post(
            MODEL_URL,
            headers=headers,
            data=image_bytes,   # ✅ send raw bytes instead of multipart form
            timeout=60
        )

        if response.status_code != 200:
            return None, f"Model request failed: {response.status_code} - {response.text}"

        return response.json(), None

    except Exception as e:
        return None, str(e)


# ------------------------------------------------------
# HELPER: make recommendations for study performance
# ------------------------------------------------------
def get_study_recommendations(predictions):
    # Basic nutrition-to-academic link mapping
    study_tips = {
        "apple": "Apples help stabilize blood sugar for long study sessions.",
        "banana": "Bananas boost memory and focus — great before exams.",
        "egg": "Eggs contain choline, which improves brain function.",
        "fish": "Fish (especially salmon) has omega-3, improving memory.",
        "nuts": "Nuts support focus and reduce exam stress.",
        "yogurt": "Yogurt helps mood and concentration through probiotics.",
        "milk": "Milk contains vitamin B12 for healthy brain function.",
        "chocolate": "Dark chocolate improves blood flow to the brain.",
        "bread": "Whole-grain bread provides steady energy for studying.",
        "coffee": "Coffee boosts alertness — just don’t overdo it!",
        "tea": "Tea improves calm focus without caffeine crashes."
    }

    recommendations = []
    for p in predictions:
        food_label = p["label"].lower()
        for key, advice in study_tips.items():
            if key in food_label:
                recommendations.append({
                    "food": key,
                    "advice": advice
                })
    return recommendations


# ------------------------------------------------------
# ROUTES
# ------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    result, error = analyze_image(image_bytes)
    if error:
        return jsonify({"error": error}), 500

    # Model returns list of {label, score}
    predictions = sorted(result, key=lambda x: x["score"], reverse=True)[:5]
    recommendations = get_study_recommendations(predictions)

    return jsonify({
        "predictions": predictions,
        "recommendations": recommendations
    })


# ------------------------------------------------------
# DEPLOYMENT ENTRY POINT
# ------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
