import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoder
model = pickle.load(open("disease_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Load datasets
dataset = pd.read_csv("dataset.csv")
desc_df = pd.read_csv("symptom_Description.csv")
precaution_df = pd.read_csv("symptom_precaution.csv")
severity_df = pd.read_csv("Symptom-severity.csv")

# Prepare symptom list
symptom_columns = [col for col in dataset.columns if col.startswith("Symptom_")]
all_symptoms = pd.unique(dataset[symptom_columns].values.ravel('K'))
all_symptoms = sorted([symptom for symptom in all_symptoms if pd.notna(symptom)])

# UI
st.title("Smart Disease Predictor")

selected_symptoms = st.multiselect("Select your symptoms", all_symptoms)

# Predict button
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Encode input symptoms
        input_vector = np.array([1 if s in selected_symptoms else 0 for s in all_symptoms]).reshape(1, -1)
        prediction = model.predict(input_vector)
        predicted_disease = encoder.inverse_transform(prediction)[0]

        # Description
        desc_row = desc_df[desc_df["Disease"].str.lower() == predicted_disease.lower()]
        description = desc_row["Description"].values[0] if not desc_row.empty else "No description available."

        # Precautions
        prec_row = precaution_df[precaution_df["Disease"].str.lower() == predicted_disease.lower()]
        precautions = []
        if not prec_row.empty:
            for i in range(1, 5):
                col_name = f"Precaution_{i}"
                if col_name in prec_row.columns and pd.notna(prec_row.iloc[0][col_name]):
                    precautions.append(prec_row.iloc[0][col_name])

        # Severity scoring
        severity_dict = dict(zip(severity_df["Symptom"].str.strip(), severity_df["weight"]))
        severity_score = sum(severity_dict.get(symptom.strip(), 0) for symptom in selected_symptoms)

        # Show results
        st.success(f"Predicted Disease: {predicted_disease}")
        st.info(f"Description: {description}")
        
        if precautions:
            st.warning("Precautions:")
            for item in precautions:
                st.markdown(f"- {item}")

        st.info(f"Total Severity Score: `{severity_score}` (based on selected symptoms)")
