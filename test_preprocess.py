import pandas as pd
from utils.preprocess_diabetes import preprocess_diabetes

df = pd.read_csv('data/diabetes.csv')

df_clean = preprocess_diabetes(df)

print("Shape:", df_clean.shape)
print("\nMissing values:", df_clean.isnull().sum().sum())
print("\nTarget distribution:")
print(df_clean['readmitted'].value_counts())