import requests

patient = {
    'age': 56,
    'sex': 'Male',
    'cp': 'asymptomatic',
    'trestbps': 150.0,
    'chol': 213.0,
    'fbs': 'True',
    'restecg': 'normal',
    'thalch': 125.0,
    'exang': 'True',
    'oldpeak': 1.0,
    'slope': 'flat',
    'ca': 0.0,
    'thal': 'normal'
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=patient)
print(response.json())