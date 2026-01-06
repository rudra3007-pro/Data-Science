import joblib
import numpy as np

model = joblib.load("models/rain_model.pkl")
scaler = joblib.load("models/scaler.pkl")

sample = np.array([[1,2,3,4,5]])
sample = scaler.transform(sample)

prediction = model.predict(sample)

if prediction[0] == 1:
    print("🌧 Rain Tomorrow")
else:
    print("☀ No Rain Tomorrow")
