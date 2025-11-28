from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# --- Load Dataset ---
df = pd.read_csv("dataset.csv")

# Keep only some basic useful columns (avoid NaN heavy columns)
df = df[['MinTemp', 'MaxTemp', 'Humidity3pm', 'Pressure9am', 'RainTomorrow']]
df.dropna(inplace=True)

# Encode target (Yes/No → 1/0)
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1, 'No':0})

# Split features & labels
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# --- Frontend Handling ---
@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        MinTemp = float(request.form['MinTemp'])
        MaxTemp = float(request.form['MaxTemp'])
        Humidity3pm = float(request.form['Humidity3pm'])
        Pressure9am = float(request.form['Pressure9am'])

        input_data = [[MinTemp, MaxTemp, Humidity3pm, Pressure9am]]
        pred = model.predict(input_data)[0]
        prediction = "Rain" if pred == 1 else "No Rain"

    return render_template_string(open("frontend.html").read(), prediction=prediction)

app.run(debug=True)
