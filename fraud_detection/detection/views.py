import numpy as np
import joblib
from django.shortcuts import render
from django.http import JsonResponse

# Load the pre-trained model
model = joblib.load('detection/fraud_model.pkl')

# Define the expected number of features (30)
expected_feature_count = 30

def predict_fraud(request):
    prediction_result = None
    if request.method == 'POST':
        try:
            # Extract features from the POST request data
            data = request.POST.dict()
            features = []
            
            # Skip csrfmiddlewaretoken and convert other values to float
            for key, value in data.items():
                if key == 'csrfmiddlewaretoken':
                    continue  # Skip csrfmiddlewaretoken
                try:
                    features.append(float(value))  # Attempt to convert to float
                except ValueError:
                    return JsonResponse({"error": f"Invalid value for {key}. Must be a numeric value."}, status=400)

            # Ensure that we have the expected number of features (30)
            if len(features) != expected_feature_count:
                return JsonResponse({"error": f"Expected {expected_feature_count} input features, but got {len(features)}."}, status=400)

            features = np.array(features).reshape(1, -1)

            # Make prediction using the loaded model
            prediction = model.predict(features)[0]
            prediction_result = "Fraud" if prediction == 1 else "Not Fraud"

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return render(request, "predict.html", {"prediction": prediction_result})
