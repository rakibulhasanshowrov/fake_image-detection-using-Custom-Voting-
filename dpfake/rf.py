import cv2
import numpy as np
import pickle
from skimage.restoration import estimate_sigma
import os

# Directory of the current file
model_dir = os.path.dirname(__file__)

# Function to detect and crop face using OpenCV
def crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        print("No face detected.")
        return None  # Return None if no face is detected
    
    x, y, w, h = faces[0]
    return image[y:y+h, x:x+w]

# Function to calculate blurriness (Laplacian variance)
def calculate_blurriness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian

# Function to estimate noise patterns (using skimage)
def calculate_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma_est = estimate_sigma(gray, average_sigmas=True)
    return sigma_est

# Function to preprocess image and extract features
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    face = crop_face(image)
    if face is None:
        return 0.0, 0.0  # Return default values if no face detected
    
    blurriness = calculate_blurriness(face)
    noise = calculate_noise(face)
    
    return blurriness, noise

# Function to load model and make prediction
def predict(image_path):
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')  # Model path based on the script's directory

    if not os.path.exists(model_path):
        raise ValueError("Model file not found.")
    
    # Load the trained model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Preprocess the image
    blurriness, noise = preprocess_image(image_path)
    
    # Create feature array
    features = np.array([[blurriness, noise]])
    
    # Make prediction
    prediction = model.predict(features)
    confidence = model.predict_proba(features).max()
    
    # Map prediction to class labels
    predicted_class = 1 if prediction[0] == 1 else 0  # 1 = Real, 0 = Fake
    
    return predicted_class, float(confidence)  # Ensure confidence is returned as float

def predict_image(image_path):
    """Function to call for predictions in Django view."""
    try:
        predicted_class, confidence = predict(image_path)
        return int(predicted_class), float(confidence)  # Ensure return types are int and float
    except ValueError as e:
        print(f"Prediction error: {e}")
        return 0, 0.0  # Return default values in case of error
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 0, 0.0  # Return default values in case of unexpected error


