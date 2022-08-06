import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import json
import jsonpickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# model = pickle.load(open('./models/model.pkl', 'rb'))


@app.route("/", methods=['GET'])
def index():
    return "Hello World."


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    columns = ['age', 'sex', 'cp', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
               'oldpeak', 'slope', 'ca', 'thal']
    df = pd.DataFrame([data])
    dataframe = pd.read_csv('datasets/original/heart.csv')
    scaler = StandardScaler()
    scaler.fit_transform(dataframe.drop(columns=['target'], axis=1))
    X = scaler.transform(df)
    model = pickle.load(open('./models/model.pkl', 'rb'))
    output = model.predict(X)
    if output[0] == 1:
        return app.response_class(
            response=json.dumps({
                "message": "You have a heart disease.",
                "precaution": [],
                "has_diabetes": False,
                'has_heart_disease': False,
                "has_tuberculosis": False
            }),
            status=200,
            mimetype='application/json'
        )
    else:
        return app.response_class(
            response=json.dumps({
                "message": "You don't have a heart disease.",
                "precaution": [],
                "has_diabetes": False,
                'has_heart_disease': False,
                "has_tuberculosis": False
            }),
            status=200,
            mimetype='application/json'
        )


if __name__ == '__main__':
    app.run(debug=True)
