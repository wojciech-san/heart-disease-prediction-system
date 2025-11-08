import pickle
from flask import Flask
from flask import request
from flask import jsonify

import os

model_file = os.path.join(os.path.dirname(__file__), "models", "random_forest_heart_disease_v1.bin")

#model_file = '/workspaces/heart-disease-prediction-system/models/random_forest_heart_disease_v1.bin'
 
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
 
app = Flask('heart_disease')
 
@app.route('/predict', methods=['POST'])
def predict():
    # json = Python dictionary
    patient = request.get_json()
 
    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0,1] 
    heart_disease = y_pred >= 0.5
 
    result = {
        'heart_disease_probability': float(y_pred),
        'heart_disease': bool(heart_disease)
    }
 
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)