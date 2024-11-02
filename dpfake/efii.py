import torch
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
import os
from torchvision import transforms
from torchvision.models import efficientnet_b0
from django.conf import settings

# Load the EfficientNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0()
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(model.classifier[1].in_features, 1),
    torch.nn.Sigmoid()
)
model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, 'dpfake', 'efficientnet_model.pth'), map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# Initialize face detector
detector = MTCNN()

def detect_and_crop_face(image_path):
    """Detect and crop face from an image."""
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at {image_path}")
        return None
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    detections = detector.detect_faces(image_rgb)

    if detections:
        x, y, width, height = detections[0]['box']
        face = image_rgb[y:y + height, x:x + width]
        face_resized = cv.resize(face, (224, 224))  # Resize to match EfficientNet input size
        return face_resized
    else:
        print("No face detected in the image.")
        return None

def predict_image(image_path):
    """Predict if an image is fake or real, and return prediction and confidence score."""
    # Detect and preprocess the face
    face = detect_and_crop_face(image_path)
    if face is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Same as training normalization
        ])
        
        face_tensor = transform(face).unsqueeze(0).to(device)  # Add batch dimension
        
        # Make prediction (assuming sigmoid output)
        with torch.no_grad():
            prediction = model(face_tensor)
        confidence_score = prediction.item()
        
        # Classify as 1 for Real (confidence > 0.5) and 0 for Fake (confidence <= 0.5)
        predicted_class = 1 if confidence_score > 0.5 else 0
        return int(predicted_class), float(confidence_score)  # Ensure return types are int and float
    else:
        return 0, 0.0  # Return 0 for class and 0.0 for confidence if no face is detected

def predict(image_path):
    """Function to call for predictions in Django view."""
    if os.path.exists(image_path):
        predicted_class, confidence_score = predict_image(image_path)
        return int(predicted_class), float(confidence_score)  # Ensure return types are int and float
    else:
        return 0, 0.0  # Return 0 for class and 0.0 for confidence if the image path does not exist
