from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle

class PatientData(BaseModel):
    age: float
    sex: str
    cp: str
    trestbps: float
    chol: float
    fbs: str
    restecg: str
    thalch : float
    exang: str
    oldpeak: float
    slope: str
    ca: int
    thal: str

app = FastAPI(title="Heart Disease Prediction API")


model_path = os.path.join(os.path.dirname(__file__), "models", "random_forest_heart_disease_v1.bin")
with open(model_path, "rb") as f_in:
    dv, model = pickle.load(f_in)  

@app.post("/predict")
def predict(data: PatientData):
 
    data_dict = data.dict()
    

    X = dv.transform([data_dict])
    
  
    pred_proba = model.predict_proba(X)[0, 1]
    pred_class = int(model.predict(X)[0])
    
    return {
        "prediction": pred_class,
        "probability": float(pred_proba)
    }
