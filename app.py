import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

LOGMEAL_TYPE_URL = "https://api.logmeal.com/image/recognition/type/v0.0"
LOGMEAL_DETECT_URL = "https://api.logmeal.com/v2/recognition/dish"
API_KEY = os.environ.get("LOGMEAL_KEY")

def detect_food_type(image_bytes):
    headers = {
        "Content-Type": "image/jpeg",
        "Authorization": f"Bearer {API_KEY}"
    }
    resp = requests.post(LOGMEAL_TYPE_URL, headers=headers, data=image_bytes)
    if resp.status_code != 200:
        return None, f"Food type API error {resp.status_code}: {resp.text}"
    return resp.json(), None


def detect_food_items(image_bytes):
    headers = {
        "Content-Type": "image/jpeg",
        "Authorization": f"Bearer {API_KEY}"
    }
    resp = requests.post(LOGMEAL_DETECT_URL, headers=headers, data=image_bytes)
    if resp.status_code != 200:
        return None, f"Food detect API error {resp.status_code}: {resp.text}"
    return resp.json(), None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    # Step 1: detect whether there is food
    type_data, err = detect_food_type(image_bytes)
    if err:
        return jsonify({"error": err}), 500

    if type_data.get("type") == "non_food":
        return jsonify({
            "message": "No food detected in image.",
            "predictions": [],
            "recommendations": []
        })

    # Step 2: identify food items
    detect_data, err2 = detect_food_items(image_bytes)
    if err2:
        return jsonify({"error": err2}), 500

    items = []
    for d in detect_data.get("recognition_results", []):
        if "name" in d:
            items.append(d["name"])

    if not items:
        return jsonify({"message": "No recognizable food found."})

    # Step 3: recommendations for study performance
    recs = []
    for food in items:
        f = food.lower()
        if any(x in f for x in ["banana", "apple", "berries", "nuts", "egg", "fish", "salmon", "milk", "yogurt"]):
            recs.append(f"✅ {food}: Great for brain energy and memory.")
        elif any(x in f for x in ["pizza", "burger", "soda", "fries", "cake", "ice cream"]):
            recs.append(f"⚠️ {food}: Not ideal for studying, may reduce focus.")
        else:
            recs.append(f"ℹ️ {food}: Neutral effect on studying.")

    return jsonify({
        "message": "Food detected successfully!",
        "predictions": items,
        "recommendations": recs
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
