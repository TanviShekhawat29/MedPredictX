import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_diabetes(df):

    # 1. Remove duplicates
    df = df.drop_duplicates()

    # 2. Drop unnecessary columns
    df = df.drop(['encounter_id', 'patient_nbr'], axis=1)

    # 3. Replace '?' with NaN
    df = df.replace('?', np.nan)

    # 4. Drop columns with too many missing values
    df = df.dropna(axis=1, thresh=int(0.7 * len(df)))

    # 5. Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # 6. Fill missing values properly
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 7. Convert target variable
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    # 8. Feature engineering
    df['total_visits'] = (
        df['number_outpatient'] +
        df['number_emergency'] +
        df['number_inpatient']
    )

    # 9. Encode categorical variables
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df