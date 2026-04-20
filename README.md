# 🏥 MedPredictX: AI-Based Diabetes Readmission Risk Predictor

🔗 **Live App:**  
👉 https://medpredictx-dn2xejfajtvow65qhqtwyo.streamlit.app/

---

## 📌 Overview

**MedPredictX** is an AI-powered healthcare application designed to predict the **risk of hospital readmission for diabetic patients**.  
It leverages machine learning techniques to assist in early risk detection and supports better clinical decision-making.

The system is built with a focus on:
- Accuracy
- Simplicity
- Real-world usability
- Interpretability

---

## 🚀 Features

- 🧠 Machine Learning-based prediction (XGBoost)
- 📊 Real-time risk scoring
- 🩺 User-friendly medical input interface
- ⚠ Risk classification (Low / Moderate / High)
- 💡 Actionable health recommendations
- 🌐 Deployed on Streamlit Cloud

---

## 🧪 Dataset

- Source: **Kaggle**
- Dataset: *Diabetes 130-US Hospitals Dataset*
- Records: ~100,000 patients
- Features include:
  - Patient demographics
  - Hospital visits
  - Medications
  - Diagnosis history

---

## ⚙️ Tech Stack

| Category        | Tools Used |
|----------------|-----------|
| Programming    | Python 🐍 |
| ML Library     | Scikit-learn, XGBoost |
| Data Handling  | Pandas, NumPy |
| Web Framework  | Streamlit |
| Model Storage  | Joblib |
| Deployment     | Streamlit Cloud |
| Version Control| Git + GitHub |

---

## 🧠 Model Details

- Algorithm: **XGBoost Classifier**
- Data Handling:
  - Missing value treatment
  - Feature encoding
  - Class imbalance handled using **SMOTE**
- Performance:
  - Accuracy: **~85%**
- Output:
  - Binary classification (High Risk / Low Risk)
  - Probability-based risk score

---

## 🏗️ Project Structure

MedPredictX/
│
├── backend/
│ ├── diabetes_model.pkl
│ ├── scaler.pkl
│ ├── diabetes_features.pkl
│ └── router.py
│
├── frontend/
│ └── app.py
│
├── utils/
│ └── preprocess_diabetes.py
│
├── data/
│ └── diabetes.csv
│
├── requirements.txt
└── README.md


---

## ▶️ How to Run Locally

### 1. Clone Repository
```bash
git clone https://github.com/your-username/medpredictx.git
cd medpredictx

'''bash

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run Application
streamlit run frontend/app.py

---

## 📊 How It Works

User inputs patient details (age, medications, hospital visits, etc.)
Data is preprocessed and scaled
Model predicts readmission probability
Output is converted into:
Risk Level
Recommendation

---
## 💡 Use Cases

🏥 Hospitals → Identify high-risk patients
👨‍⚕ Doctors → Assist in treatment planning
📊 Healthcare analytics → Risk monitoring systems
🎓 Academic project → Predictive modeling with real-world data

---

## 🧠 Future Improvements

Explainable AI (SHAP integration)
Multi-disease prediction system
API-based backend deployment
Integration with hospital management systems

---

## 👨‍💻 Author

Tanvi Shekhawat
B.Tech (AI/ML)

----

## 📜 License

This project is for academic and educational purposes.

---

## ⭐ Acknowledgements

Kaggle for dataset
Streamlit for deployment platform
Open-source ML community

---
