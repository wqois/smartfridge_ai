import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Public Hugging Face model (no auth required for small demo)
MODEL_URL = "https://api-inference.huggingface.co/models/nateraw/food-classifier"

# Food-to-study recommendations mapping (general guidelines)
RECOMMENDATIONS = {
    "egg": "Eggs boost memory thanks to choline. Have boiled or scrambled eggs before study.",
    "banana": "Bananas give quick glucose and potassium for steady brain energy.",
    "apple": "Apples provide slow sugar release and fiber — great before long study sessions.",
    "fish": "Fatty fish (like salmon) contain omega-3s for focus and memory.",
    "yogurt": "Yogurt has probiotics and protein for sustained mental energy.",
    "milk": "Milk gives calcium and protein — good in the morning.",
    "nuts": "Nuts provide healthy fats and magnesium for concentration.",
    "chocolate": "Dark chocolate (70%+) improves blood flow to the brain.",
    "bread": "Wholegrain bread offers complex carbs for lasting energy.",
    "tomato": "Tomatoes have antioxidants for overall brain health.",
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    image_bytes = file.read()

    # Send image to Hugging Face model
    resp = requests.post(MODEL_URL, files={"file": image_bytes})
    if resp.status_code != 200:
        return jsonify({"error": "API request failed", "details": resp.text}), 500

    try:
        results = resp.json()
    except Exception:
        return jsonify({"error": "Invalid JSON from model"}), 500

    # Extract top 5 labels
    predictions = []
    foods = []
    if isinstance(results, list):
        for item in results[:5]:
            label = item.get("label", "")
            score = round(item.get("score", 0) * 100, 1)
            predictions.append({"label": label, "score": score})
            # Match label to our study recommendation
            for key in RECOMMENDATIONS:
                if key.lower() in label.lower():
                    foods.append({
                        "food": key.capitalize(),
                        "advice": RECOMMENDATIONS[key],
                        "confidence": score
                    })

    return jsonify({
        "predictions": predictions,
        "recommendations": foods
    })

if __name__ == "__main__":
    app.run(debug=True)
