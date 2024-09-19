from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import rasterio
import cv2

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\DELL\Downloads\IBM\third_task_cellula\New folder\model_2 (1).h5')

def load_tiff_image(image):
    image = io.BytesIO(image)
    with rasterio.open(image) as src:
        # Read all bands
        image_data = np.stack([src.read(i) for i in range(1, src.count + 1)], axis=-1)
    return image_data

def preprocess_image(image):
    image = load_tiff_image(image)
    # Resize to (128, 128) assuming model expects this size
    image_resized = cv2.resize(image, (128, 128))
    image_resized = image_resized / 255.0  # Normalize
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
    return image_resized

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0, :, :, 0]  # Return the first (and only) channel

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        img = file.read()
        prediction = predict(img)
        # Convert prediction to image for display
        pred_img = (prediction * 255).astype(np.uint8)
        pred_img = Image.fromarray(pred_img)
        pred_img.save('static/prediction.png')
        return render_template('index.html', image_url='static/prediction.png')

if __name__ == '__main__':
    app.run(debug=True)
