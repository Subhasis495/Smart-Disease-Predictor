# Smart Disease Predictor using ML

- **A smart ML-powered web app that predicts diseases based on symptoms using a Random Forest classifier. It also shows descriptions, severity level, and suggested precautions — all in a clean Streamlit interface.**


## ✅ Features

- 🔍 Disease prediction using symptoms
- 🩺 Disease description from medical datasets
- 💊 Precaution suggestions based on diagnosis
- 🖥️ Simple and interactive Streamlit UI


## 🚀 Live App

- 📌 [View Live App](https://subhasis495-smart-disease-predictor-app-jlitp8.streamlit.app/)


## 🧪 Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Pickle (for saving model)

## 📁 Project Structure
```
smart-disease-predictor/
│
├── app.py                            #  Streamlit UI app
├── train_model.py                    #  Trains and saves ML model
├── requirements.txt                  #  Dependencies for deployment
├── README.md                         #  Project description
│
├── dataset.csv                       #  Disease & symptom dataset
├── symptom_Description.csv           #  Disease descriptions
├── symptom_precaution.csv            #  Precaution tips
├── Symptom-severity.csv              #  Severity weights
│
├── disease_model.pkl                 #  Trained ML model (after training)
└── label_encoder.pkl                 #  Saved label encoder (after training)
```

## 🛠️ Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates model + encoder)
python train_model.py

# 3. Start the web app
streamlit run app.py
Then open http://localhost:8501 in your browser.
```


## 🤝 Contributing
- Pull requests are welcome. For major changes, please open an issue first to discuss.
