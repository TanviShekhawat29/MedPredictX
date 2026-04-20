from backend.router import route_prediction
import joblib

# Load feature names to build correct input
feature_names = joblib.load('backend/diabetes_features.pkl')

# =========================
# TEST DIABETES MODEL
# =========================
sample_data = {}

# Fill all features with 0 (dummy test)
for feature in feature_names:
    sample_data[feature] = 0

pred, prob = route_prediction("diabetes", sample_data)

print("=== Diabetes Prediction ===")
print("Prediction:", pred)
print("Risk Score:", prob)


# =========================
# TEST HEART MODEL
# =========================
heart_sample = {
    "age": 50,
    "sex": 1,
    "cp": 0,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 2
}

pred, prob = route_prediction("heart", heart_sample)

print("\n=== Heart Prediction ===")
print("Prediction:", pred)
print("Risk Score:", prob)