from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import pickle



app = Flask(__name__, template_folder="template")
model = pickle.load(open("models/croprecGNB.pkl", "rb"))
print("Model Loaded")

@app.route("/",methods= ['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict",methods=['GET','POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_lst = [[N,P,K,temperature,humidity,ph,rainfall]]
        pred = model.predict(input_lst)
        output = pred
        if output == 'rice':
            return render_template("rice.html")

        elif output == 'maize':
            return render_template("maize.html")

        elif output == 'chickpea':
            return render_template("chickpea.html")

        elif output == 'kidneybeans':
            return render_template("kidneybeans.html")

        elif output == 'pigeonpeas':
            return render_template("pigeonpeas.html")

        elif output == 'mothbeans':
            return render_template("mothbeans.html")

        elif output == 'mungbean':
            return render_template("mungbean.html")

        elif output == 'blackgram':
            return render_template("blackgram.html")

        elif output == 'lentil':
            return render_template("lentil.html")

        elif output == 'pomegranate':
            return render_template("pomegranate.html")

        elif output == 'banana':
            return render_template("banana.html")

        elif output == 'mango':
            return render_template("mango.html")

        elif output == 'grapes':
            return render_template("grapes.html")

        elif output == 'watermelon':
            return render_template("watermelon.html")

        elif output == 'muskmelon':
            return render_template("muskmelon.html")

        elif output == 'apple':
            return render_template("apple.html")

        elif output == 'orange':
            return render_template("orange.html")

        elif output == 'papaya':
            return render_template("papaya.html")

        elif output == 'coconut':
            return render_template("coconut.html")

        elif output == 'cotton':
            return render_template("cotton.html")

        elif output == 'jute':
            return render_template("jute.html")
        else:
            return render_template("coffee.html")

    return render_template("predictor.html")













if __name__== '__main__':
    app.run(debug = True)