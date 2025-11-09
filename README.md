# ❤️ Heart-Disease-Prediction-System

This repository contains a machine learning project that predicts the likelihood of a patient having heart disease based on clinical parameters. The system uses a **Random Forest Classifier** trained on a public health dataset and is packaged for deployment using **Docker** and **Pipenv**.

## 🌟 Features

* **Exploratory Data Analysis (EDA):** Detailed analysis of the dataset to understand feature distributions and correlations.
* **Machine Learning Model:** Utilizes a **Random Forest Classifier** for robust prediction.
* **Prediction API:** A Python script (`predict.py`) exposes the model via an API for real-time predictions.
* **Dependency Management:** Uses **Pipenv** for consistent environment and dependencies.
* **Containerization:** Ready-to-deploy using **Docker** for reproducibility and easy scaling.

## 💾 Dataset

The model was trained using the **Heart Disease UCI Dataset** available on Kaggle.

* **Source:** [https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data?select=heart_disease_uci.csv](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data?select=heart_disease_uci.csv)
* **File:** `data/heart_disease_uci.csv`
* **Features:** Includes crucial clinical parameters such as `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (serum cholestoral), and others.

## ⚙️ Project Structure

```
Heart-Disease-Prediction-System/
├── data/
│   └── heart_disease_uci.csv               # Original dataset
│
├── models/
│   └── random_forest_heart_disease_v1.bin  # Trained Random Forest model
│
├── notebooks/
│   ├── exploratory-data-analysis.ipynb     # Jupyter notebook for EDA, training, and model comparison
│   └── predict-test.ipynb                  # Notebook for testing the prediction script
│
├── temp/                                  # Temporary files
│
├── Dockerfile                             # Defines the Docker image build process
├── Pipfile                                # Pipenv dependency definitions
├── Pipfile.lock                           # Locked dependencies for deterministic builds
├── predict.py                             # Prediction script / FastAPI endpoint
├── train.py                               # Script used for model training
└── README.md                              # Main description file
```
## 🚀 Getting Started

### Prerequisites

* **Python 3.x**
* **Git**
* **Docker** (for containerized deployment)

### Local Setup

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    cd Heart-Disease-Prediction-System
    ```

2.  **Install Dependencies using Pipenv:**
    ```bash
    pip install pipenv
    pipenv install --dev
    pipenv shell
    ```

3.  **Run the Prediction Application (using Gunicorn/Flask):**
    ```bash
    # Based on your notes: pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
    pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
    ```
    The prediction service should now be running locally at `http://0.0.0.0:9696`.

### 🐳 Docker Deployment

For a production-ready and reproducible environment, you can build and run the application using Docker.

1.  **Build the Docker Image:**
    ```bash
    docker build -t heart-disease-predictor .
    ```

2.  **Run the Container:**
    ```bash
    # Maps the container's port 9696 to your host's port 9696
    docker run -it --rm -p 9696:9696 heart-disease-predictor
    ```
    The prediction service will be available at `http://localhost:9696`.

---

## 📈 Model Training and Evaluation

The model selection process involved testing several popular classification algorithms to determine the most effective one for this dataset.

The following models were trained and evaluated with the resulting accuracy scores:

| Model | Accuracy Score |
| :--- | :--- |
| **RandomForestClassifier** | **0.8756** |
| xgb (XGBoost Classifier) | 0.8702 |
| LogisticRegression | 0.8549 |
| DecisionTreeClassifier | 0.8414 |

Based on these results, the **Random Forest Classifier** achieved the highest accuracy score of **0.8756**, and was therefore chosen as the final model for the prediction system. The final trained model is saved in `models/random_forest_heart_disease_v1.bin`.

For the detailed analysis, model comparison, and feature insights, refer to the `notebooks/exploratory-data-analysis.ipynb` notebook.

---

## 💻 API Usage Example

Once the service is running (either locally or via Docker), you can submit a prediction request using a tool like `curl` or a Python script.

**Request Endpoint:** `http://localhost:9696/predict`

**Example Patient Data:**
```json
{
    "age": 56,
    "sex": "Male",
    "cp": "asymptomatic",
    "trestbps": 150.0,
    "chol": 213.0,
    "fbs": "True",
    "restecg": "normal",
    "thalch": 125.0,
    "exang": "True",
    "oldpeak": 1.0,
    "slope": "flat",
    "ca": 0.0,
    "thal": "normal"
}

{
  "prediction": 1,
  "probability": 0.684711572055718
}