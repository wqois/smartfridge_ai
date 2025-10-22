import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
MODEL_URL = "https://api-inference.huggingface.co/models/nateraw/food-classifier"
HF_TOKEN = os.environ.get("HF_TOKEN")  # optional: set on Render if needed

def hf_post(image_bytes):
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    resp = requests.post(MODEL_URL, headers=headers, files={"file": image_bytes}, timeout=30)
    return resp

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    image_bytes = file.read()
    resp = hf_post(image_bytes)
    if resp.status_code != 200:
        return jsonify({"error": "API request failed", "details": resp.text}), 500
    results = resp.json()
    # parse results (same logic as earlier)
    # ...
    return jsonify({"predictions": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
