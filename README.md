# Dog vs Cat Image Classifier

This project is a web-based application that classifies images as either a dog or a cat using a pre-trained MobileNetV2 model. The application is built with Flask, TensorFlow, and Bootstrap for the UI.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Web Application](#web-application)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [License](#license)

## Project Overview

The Dog vs Cat Image Classifier is a machine learning web application that allows users to upload an image of a dog or a cat, and the app will predict whether the uploaded image is of a dog or a cat. The prediction is performed using a deep learning model based on the MobileNetV2 architecture.

## Dataset

The dataset used for this project is the [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle, which contains 25,000 images of dogs and cats.

## Model

The image classification model is built using TensorFlow and TensorFlow Hub. The MobileNetV2 model was used as a feature extractor, and a custom classifier was added on top to perform binary classification (dog vs cat).

### Training Process
1. **Data Preprocessing**: The images are resized to 224x224 pixels and normalized.
2. **Model Architecture**: 
   - MobileNetV2 is used as the base model.
   - A GlobalAveragePooling2D layer is added, followed by a Dense layer with a softmax activation function for classification.
3. **Training**: The model was trained on 2000 images (1000 dogs and 1000 cats) with an 80-20 train-test split.

## Web Application

The web application is built using Flask, and the frontend is styled with Bootstrap. Users can upload an image of a dog or a cat, and the application will display the prediction along with the uploaded image.

## Requirements

- Python 3.x
- Flask
- TensorFlow
- TensorFlow Hub
- OpenCV
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- Bootstrap (for frontend styling)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Pahinithi/-Dog-vs-Cat-Classification-using-Transfer-Learning-Deep-Learning
   cd Dog-vs-Cat-Classifier
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the model and place it in the root directory**:
   - Ensure you have `model.h5` saved in the project root directory.

## Usage

1. **Run the Flask application**:
   ```bash
   python app.py
   ```

2. **Access the application**:
   - Open your web browser and go to `http://127.0.0.1:5000/`.
   - Upload an image and click "Upload & Predict" to see the classification result.

## Demo
https://drive.google.com/file/d/1WaYpf6XO8GpQFOd1Uqtt1dPpxUCytGS0/view?usp=sharing

## License

This project is licensed under the MIT License.



<img width="1728" alt="Screenshot 2024-08-21 at 12 15 40" src="https://github.com/user-attachments/assets/1d202cc9-4e77-4ad0-b39c-b0785dc2a985">



