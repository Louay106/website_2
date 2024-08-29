from flask import Flask, request, jsonify, send_from_directory, render_template
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
import cv2
import numpy as np
 
import os

app = Flask(__name__)

# Load Pre-trained Model and Image Processor
model_name = "edixo/road_good_damaged_condition"
model = AutoModelForImageClassification.from_pretrained(model_name)
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Directory configuration
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
IMAGES_FOLDER = os.path.join(STATIC_FOLDER, 'images')
STYLES_FOLDER = os.path.join(STATIC_FOLDER, 'styles')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/solutions')
def solutions():
    return render_template('solutions.html')

@app.route('/analyze_road', methods=['POST'])
def analyze_road():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    # Convert image bytes to numpy array
    image_array = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image array
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess the image without padding and truncation
    try:
        inputs = image_processor(images=image, return_tensors="pt")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Make predictions
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()

        # Map Predicted Class to Label
        labels = ["bad", "good"]
        prediction = labels[predicted_class]

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Serve static files from the 'static' directory
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)