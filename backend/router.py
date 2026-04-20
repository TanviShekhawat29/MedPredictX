import joblib
import pandas as pd

# Load model + scaler
model = joblib.load('backend/diabetes_model.pkl')
scaler = joblib.load('backend/scaler.pkl')
features = joblib.load('backend/diabetes_features.pkl')


def predict(data_dict):

    df = pd.DataFrame([data_dict])
    df = df.reindex(columns=features, fill_value=0)

    df_scaled = scaler.transform(df)

    prob = model.predict_proba(df_scaled)[0][1]

    # ----------------------------
    # CALIBRATION (IMPORTANT)
    # ----------------------------
    risk_score = prob

    if data_dict["age"] < 40:
        risk_score -= 0.1

    if data_dict["num_medications"] < 8:
        risk_score -= 0.1

    if data_dict["number_inpatient"] == 0:
        risk_score -= 0.1

    if data_dict["number_emergency"] == 0:
        risk_score -= 0.1

    risk_score = max(0, min(risk_score, 1))

    prediction = int(risk_score > 0.5)

    return prediction, risk_score