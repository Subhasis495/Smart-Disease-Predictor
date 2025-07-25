import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Get symptom columns
symptom_columns = [col for col in df.columns if col.startswith("Symptom_")]

# Replace NaNs with "None"
df[symptom_columns] = df[symptom_columns].fillna("")

# Build list of all unique symptoms
all_symptoms = sorted(set(symptom for sublist in df[symptom_columns].values.tolist() for symptom in sublist if symptom))

# Binary encode symptoms
X = []
for _, row in df.iterrows():
    symptoms = set(row[symptom_columns])
    X.append([1 if symptom in symptoms else 0 for symptom in all_symptoms])

X = np.array(X)

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoder
with open("disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print(" Model and label encoder saved successfully.")
