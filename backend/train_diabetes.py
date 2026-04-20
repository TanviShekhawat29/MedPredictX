import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from utils.preprocess_diabetes import preprocess_diabetes


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv('data/diabetes.csv')

# =========================
# 2. PREPROCESS
# =========================
df = preprocess_diabetes(df)

X = df.drop('readmitted', axis=1)
y = df['readmitted']

# =========================
# 3. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 4. HANDLE IMBALANCE (SMOTE)
# =========================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:\n", y_train.value_counts())

# =========================
# 5. FEATURE SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 6. CLASS WEIGHT (AUTO)
# =========================
scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# =========================
# 7. MODEL
# =========================
model = XGBClassifier(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=scale,
    random_state=42,
    eval_metric='auc'
)

# =========================
# 8. TRAIN
# =========================
model.fit(X_train, y_train)

# =========================
# 9. PREDICTION (THRESHOLD TUNING)
# =========================
y_prob = model.predict_proba(X_test)[:, 1]

# Adjust threshold (IMPORTANT)
threshold = 0.48
y_pred = (y_prob > threshold).astype(int)

# =========================
# 10. EVALUATION
# =========================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 11. SAVE MODEL + SCALER
# =========================
joblib.dump(model, 'backend/diabetes_model.pkl')
joblib.dump(scaler, 'backend/scaler.pkl')
joblib.dump(X.columns.tolist(), 'backend/diabetes_features.pkl')
print("\nModel + Scaler saved successfully ✅")