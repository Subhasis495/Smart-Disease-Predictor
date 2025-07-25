import joblib

# Load model and encoder
model = joblib.load('disease_model.pkl')
le = joblib.load('label_encoder.pkl')

def predict_disease(symptoms_vector):
    prediction = model.predict([symptoms_vector])[0]
    probabilities = model.predict_proba([symptoms_vector])[0]
    confidence = max(probabilities)
    predicted_disease = le.inverse_transform([prediction])[0]
    return predicted_disease, confidence
