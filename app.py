from flask import Flask, request, render_template
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__, template_folder='.')

# Load the trained model
model = load_model("model.h5")

# Ensure the static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Define a route for the homepage
@app.route("/")
def home():
    return render_template("index.html") # Render the index.html template

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get the image file from the request
    file = request.files["file"]

    # Save the image to the static directory
    image_path = os.path.join("static", file.filename)
    file.save(image_path)

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))
    image_scaled = image_resized / 255.0
    image_reshaped = np.reshape(image_scaled, [1, 224, 224, 3])

    # Make a prediction using the model
    prediction = model.predict(image_reshaped)
    predicted_label = np.argmax(prediction)

    # Determine the output message
    if predicted_label == 0:
        prediction_text = 'The image represents a Cat'
    else:
        prediction_text = 'The image represents a Dog'

    return render_template("index.html", prediction=prediction_text, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)  # Run the application in debug mode for development
