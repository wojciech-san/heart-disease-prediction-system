import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle
import os

# parameters
n_estimators = 100
max_depth = 4
min_samples_leaf = 5
output_file = f'../models/random_forest_heart_disease_v1.bin'

# data preparation
df = pd.read_csv('/workspaces/heart-disease-prediction-system/data/heart_disease_uci.csv')

df['target'] = (df.num>0).astype(int)
del df['num']

categorical_columns = [col for col in list(df.dtypes[df.dtypes == "object"].index) if col !='dataset']
numerical_columns = [col for col in list(df.dtypes[df.dtypes!="object"].index) if col not in {"id","target"}]

# handle missing values
for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# data splitting
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# training function
def train(df_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf):
    dicts = df_train[categorical_columns + numerical_columns].to_dict(orient='records')
 
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
 
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf, 
        random_state=1, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
 
    return dv, model

# prediction function
def predict(df, dv, model):
     dicts = df[categorical_columns + numerical_columns].to_dict(orient='records')
 
     X = dv.transform(dicts)
     y_pred = model.predict_proba(X)[:,1]
 
     return y_pred

# final training on full training data
dv, model = train(
    df_full_train, 
    df_full_train.target.values, 
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf
)

# test evaluation
y_pred = predict(df_test, dv, model)
y_test = df_test.target.values
 
test_auc = roc_auc_score(y_test, y_pred)
print(f'Test AUC: {test_auc:.3f}')

# saving model
os.makedirs('../models', exist_ok=True)

with open(output_file,'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'Model saved to {output_file}')

# loading and testing with sample patient
model_file = '../models/random_forest_heart_disease_v1.bin'

with open(model_file,'rb') as f_in:
   dv, model = pickle.load(f_in)

patient = {
  'age': 56,
  'sex': 'Male',
  'cp': 'asymptomatic',
  'trestbps': 150.0,
  'chol': 213.0,
  'fbs': True,
  'restecg': 'normal',
  'thalch': 125.0,
  'exang': True,
  'oldpeak': 1.0,
  'slope': 'flat',
  'ca': 0.0,
  'thal': 'normal'
}

X = dv.transform([patient])
y_pred = model.predict_proba(X)[0,1]

print('Patient details:', patient)
print(f'Heart disease probability: {y_pred:.3f}')