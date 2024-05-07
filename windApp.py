import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import requests

app = Flask(__name__)

# Load the machine learning model
try:
    model = joblib.load('power_prediction.sav')
except Exception as e:
    print("Error loading the model:", e)
    model = None

@app.route('/')
def home():
    return render_template('intro.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/windapi', methods=['POST'])
def windapi():
    city = request.form.get('city')
    apikey = "43ce69715e2133b2300e0f8f7289befd"
    url = "http://api.openweathermap.org/data/2.5/weather?q=" + city + "&appid=" + apikey
    resp = requests.get(url)
    resp = resp.json()
    temp = str(resp["main"]["temp"] - 273.15) + " °C"
    humid = str(resp["main"]["humidity"]) + " %"
    pressure = str(resp["main"]["pressure"]) + " mmHG"
    speed = str(resp["wind"]["speed"]) + " m/s"
    return render_template('predict.html', temp=temp, humid=humid, pressure=pressure, speed=speed)
    
@app.route('/y_predict', methods=['POST'])
def y_predict():
    if model:
        try:
            # Get input data from form
            x_test = [[float(x) for x in request.form.values()]]

            # Make prediction
            prediction = model.predict(x_test)
            output = prediction[0]

            return render_template('predict.html', prediction_text='The energy predicted is {:.2f} KWh'.format(output))
        except Exception as e:
            print("Error during prediction:", e)
            return render_template('predict.html', prediction_text='Error during prediction. Please try again.')
    else:
        return render_template('predict.html', prediction_text='Model is not loaded. Please check the model file.')

if __name__ == "__main__":
    app.run(debug=True)
