import flask
import pickle
import numpy as np
from flask import Flask, request, render_template
import pandas as pd

app = flask.Flask(__name__, template_folder="")

#-------- MODEL GOES HERE -----------#

pipe = pickle.load(open("model/pipe.pkl", "rb"))


#-------- ROUTES GO HERE -----------#

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        result = request.form


    predicted = pipe.predict(new)[0]
    if predicted == 1:
        prediction = "YOUR LEFT STROKE JUST WENT VIRAL!!"
    else:
        prediction = "Sit down, be humble. Probably not gonna go viral."
    # prediction = '{:,.2f}%'.format(prediction)
    return render_template('index.html', prediction=prediction)




if __name__ == '__main__':
    '''Connects to the server'''

    app.run(port=5000, debug=True)
