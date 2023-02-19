import pickle
from flask import Flask,request,jsonify,url_for,render_template,redirect,flash,session,escape

import numpy as np
import pandas as pd

app=Flask(__name__)
app.config['SECRET_KEY']='1c04928469c5e375cb8a123d94bd551a'


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict_api",methods=['Post'])
def predict_api():
    model=pickle.load(open('regmodel.pkl','rb'))
    scaler=pickle.load(open('scaling.pkl','rb'))
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)

    return jsonify(output[0])

@app.route("/predict",methods=['Post'])
def predict():
    model=pickle.load(open('regmodel.pkl','rb'))
    scaler=pickle.load(open('scaling.pkl','rb'))
    data=[float (x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    output=model.predict(final_input)[0]

    return render_template("home.html",prediction_text=" Tha house price prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)