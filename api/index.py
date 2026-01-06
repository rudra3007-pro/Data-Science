from flask import Flask, render_template, request
import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

model = joblib.load(os.path.join(BASE_DIR, "models", "rain_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [
        float(request.form["temperature"]),
        float(request.form["humidity"]),
        float(request.form["pressure"]),
        float(request.form["wind"]),
        float(request.form["cloud"]),
    ]

    sample = scaler.transform([values])
    pred = model.predict(sample)[0]

    result = "🌧 Rain Tomorrow" if pred == 1 else "☀ No Rain Tomorrow"
    return render_template("index.html", prediction=result)
