import os
import cv2
import numpy as np
import dlib
import pickle
import sys
import joblib  # Use joblib for loading scaler files
from django.conf import settings
from skimage.feature import local_binary_pattern, hog

# Set the path to the parent directory of 'fake_detection'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Fixed face size for feature extraction
fixed_face_size = (128, 128)

# Load saved SVM model and scaler
model_dir = os.path.dirname(__file__)
with open(os.path.join(model_dir, 'svm_model.pkl'), 'rb') as model_file:
    clf = pickle.load(model_file)

# Load the scaler using joblib
scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))

# Define parameters for LBP
radius = 1
n_points = 8 * radius
method = 'uniform'

# Feature extraction functions
def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize
    return lbp_hist

def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    return hog_features

def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Function to detect face, resize it, and extract features
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image file.")
        return None

    rects = detector(image, 1)
    
    if len(rects) > 0:
        x, y, w, h = rects[0].left(), rects[0].top(), rects[0].width(), rects[0].height()
        face = image[y:y+h, x:x+w]  # Crop to the face region

        # Resize face to fixed size (128x128)
        face_resized = cv2.resize(face, fixed_face_size)

        # Extract features
        lbp_features = extract_lbp(face_resized)
        hog_features = extract_hog(face_resized)
        color_histogram = extract_color_histogram(face_resized)

        # Combine all features
        features = np.hstack([lbp_features, hog_features, color_histogram])
        return features
    else:
        print("No face detected in the image.")
        return None

# Function to predict class and confidence score
def predict_image(image_path):
    features = extract_features(image_path)

    if features is not None:
        # Preprocess the features
        features = scaler.transform([features])

        # Predict the class
        prediction = clf.predict(features)
        confidence = clf.predict_proba(features)

        # Get confidence score for the predicted class
        predicted_class = int(prediction[0])  # 1 = Real, 0 = Fake
        confidence_score = float(confidence[0][predicted_class])  # Index confidence based on prediction

        return predicted_class, confidence_score
    else:
        print("No face detected; returning default values.")
        return 0, 0.0  # Return 0 for class and 0.0 for confidence if no face detected

def predict(image_path):
    """Function to call for predictions in Django view."""
    if os.path.exists(image_path):
        predicted_class, confidence_score = predict_image(image_path)
        return predicted_class, confidence_score
    else:
        print(f"Error: Image path {image_path} does not exist.")
        return 0, 0.0  # Return default values in case of error


