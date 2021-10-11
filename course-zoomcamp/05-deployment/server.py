from flask import Flask
from flask import request
from flask import jsonify
import sklearn
import pickle

dv_pickle = "dv.bin"
model_pickle = "model1.bin"

app = Flask('homework')
dv_pickle_file = open(dv_pickle, 'rb')
dv = pickle.load(dv_pickle_file)
dv_pickle_file.close()

model_file = open(model_pickle, 'rb')
model = pickle.load(model_file)
model_file.close()

@app.route('/test', methods=['GET'])
def test():
    return "THIS WORKS"

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
