# app.py
from flask import Flask, request, render_template
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('digit_recognition_model.h5')

def preprocess_image(image):
    """Preprocess the uploaded image to match model input format."""
    image = ImageOps.grayscale(image)  # Convert to grayscale
    image = image.resize((28, 28))     # Resize to 28x28 pixels
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return image_array.reshape(1, 28, 28, 1)  # Reshape for model input

def predict_digit(image):
    """Predict the digit in the preprocessed image using the model."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return np.argmax(prediction)  # Returns the digit (0–9)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['digit_image']
        if file:
            # Open the image and make a prediction
            image = Image.open(file)
            prediction = predict_digit(image)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
