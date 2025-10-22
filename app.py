import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
MODEL_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
HF_TOKEN = os.environ.get("HF_TOKEN")  # Optional token for stability

# ------------------------------------------------------
# HELPER FUNCTION TO CALL HUGGING FACE MODEL
# ------------------------------------------------------
def analyze_image(image_bytes):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/jpeg"  # model expects raw image
    } if HF_TOKEN else {"Content-Type": "image/jpeg"}

    try:
        response = requests.post(
            MODEL_URL,
            headers=headers,
            data=image_bytes,   # ✅ send raw bytes, not files=
            timeout=60
        )
        if response.status_code != 200:
            return None, f"Model request failed: {response.status_code} - {response.text}"
        return response.json(), None
    except Exception as e:
        return None, str(e)


# ------------------------------------------------------
# RECOMMENDATION LOGIC
# ------------------------------------------------------
def get_study_recommendations(predictions):
    # Simplified mapping between food and study performance tips
    tips = {
        "apple": "Great source of glucose — keeps you focused and awake while studying.",
        "banana": "Excellent for memory and energy. Eat before long study sessions.",
        "coffee": "Boosts alertness, but don’t overdrink — it may reduce focus after a few hours.",
        "egg": "High in choline — helps memory and brain health.",
        "fish": "Rich in omega-3 — improves learning ability.",
        "bread": "Carbs help sustained energy, best combined with protein.",
        "chocolate": "Dark chocolate boosts blood flow to the brain — great before exams!",
        "milk": "Contains vitamin B12 — helps the brain work better under stress.",
        "yogurt": "Probiotics support mood and focus.",
        "nuts": "Packed with healthy fats — improves memory retention."
    }

    recommendations = []
    for p in predictions:
        food = p["label"].lower()
        for key, advice in tips.items():
            if key in food:
                recommendations.append({"food": key, "advice": advice})
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

    file = request.files["image"]
    image_bytes = file.read()

    result, error = analyze_image(image_bytes)
    if error:
        return jsonify({"error": error}), 500

    # Hugging Face returns a list of dicts with 'label' and 'score'
    predictions = sorted(result, key=lambda x: x["score"], reverse=True)[:5]

    recs = get_study_recommendations(predictions)

    return jsonify({
        "predictions": predictions,
        "recommendations": recs
    })


# ------------------------------------------------------
# DEPLOYMENT ENTRYPOINT
# ------------------------------------------------------
if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
