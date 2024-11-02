from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import os
from .svm import predict_image as svm_predict
from .rf import predict_image as rf_predict
from .efii import predict_image as efficientnet_predict

def get_final_prediction(svm_class, svm_confidence, rf_class, rf_confidence, efficientnet_class, efficientnet_confidence):
    # Prepare lists for confidence and predictions
    confidences = [svm_confidence, rf_confidence, efficientnet_confidence]
    print("Confidences:", confidences)
    
    predictions = [svm_class, rf_class, efficientnet_class]
    print("Predictions:", predictions)

    # Initialize flags and lists
    fake_detected = False
    fake_confidences = []  # For collecting fake predictions' confidences
    real_confidences = []  # For collecting real predictions' confidences

    # Check for any "fake" predictions with confidence greater than 0.5
    for i in range(3):  # Loop through each model
        if predictions[i] == 0:  # If prediction is fake
            fake_confidences.append(confidences[i])  # Collect confidence
            if confidences[i] > 0.53:  # Check for strong confidence
                fake_detected = True  # Mark that fake was detected
        elif predictions[i] == 1:  # If prediction is real
            real_confidences.append(confidences[i])  # Collect confidence for real predictions

    # If fake detected with high confidence
    if fake_detected:
        print("Fake detected!")
        final_prediction = 0
        # Calculate average confidence for fake detections
        average_confidence = sum(fake_confidences) / len(fake_confidences)
    else:
        print("No strong fake detected.")
        
        # Determine final prediction based on majority vote
        final_prediction = max(set(predictions), key=predictions.count)

        # Calculate average confidence for real predictions
        average_confidence = sum(real_confidences) / len(real_confidences) if real_confidences else 0

    print(f"Final Prediction: {final_prediction} with confidence: {average_confidence}")
    return final_prediction, average_confidence






def clear_uploads_folder(folder_path):
    """Removes all files from the specified folder."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {str(e)}")

def upload_and_predict(request):
    if request.method == "POST" and request.FILES.get('image'):
        # Define the path to the uploads directory within MEDIA_ROOT
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)  # Ensure the directory exists

        # Clear the uploads folder before saving a new file
        clear_uploads_folder(upload_dir)

        # Get the uploaded image and save it to the uploads directory
        image_file = request.FILES['image']
        image_path = os.path.join(upload_dir, image_file.name)
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        
        # Get a relative URL to access the image in the template
        image_url = os.path.join(settings.MEDIA_URL, 'uploads', image_file.name)

        try:
            # Call your prediction functions
            svm_class, svm_confidence = svm_predict(image_path)
            rf_class, rf_confidence = rf_predict(image_path)
            efficientnet_class, efficientnet_confidence = efficientnet_predict(image_path)

            final_prediction, average_confidence = get_final_prediction(
                svm_class, svm_confidence,
                rf_class, rf_confidence,
                efficientnet_class, efficientnet_confidence
            )

            # Prepare the context data for rendering
            context = {
                'svm_prediction': {
                    'class': svm_class,
                    'confidence': svm_confidence,
                },
                'rf_prediction': {
                    'class': rf_class,
                    'confidence': rf_confidence,
                },
                'efficientnet_prediction': {
                    'class': efficientnet_class,
                    'confidence': efficientnet_confidence,
                },
                'average_confidence': average_confidence,
                'final_prediction': {
                    'class': final_prediction,
                    'confidence': average_confidence,
                },
                'image_url': image_url,  # Relative URL for displaying the image
            }
        except Exception as e:
            # Handle exceptions and provide an error message
            context = {
                'error': f"An error occurred during prediction: {str(e)}"
            }

        # Render the prediction results
        return render(request, 'predict_image.html', context)

    # If the request is not a POST or image not uploaded, show an error
    return render(request, 'predict_image.html',)

