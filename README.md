# Sentiment Analysis – MLOps Project

This project demonstrates an end-to-end **Machine Learning + MLOps workflow** for sentiment analysis, covering model training, experiment tracking, and model serving using FastAPI.

The goal is to build a production-style ML service that can be trained offline and served online via a REST API.

---

## 🚀 Features

- Train a sentiment analysis model
- Track experiments using **MLflow**
- Save trained models locally
- Serve predictions using **FastAPI**
- REST API for real-time inference
- Clean, scalable project structure (`src/` based)

---

## 🗂️ Project Structure

SENTIMENT-MLOPS/
├── src/
│ ├── train.py # Model training logic
│ ├── predict.py # Model loading and inference
│ ├── api.py # FastAPI application
│ └── init.py
├── data/ # Local datasets (ignored in git)
├── models/ # Trained models (ignored in git)
├── mlruns/ # MLflow experiments (ignored in git)
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md


---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/OLAS05/sentiment-mlops.git
cd sentiment-mlops

2️⃣ Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

🧠 Model Training
Run the training script:
python src/train.py

This will:
Train the sentiment model
Log experiments to MLflow
Save the trained model locally

To view MLflow UI:
mlflow ui
Open: http://localhost:5000

🌐 Run the API Server
Start the FastAPI app:
uvicorn src.api:app --host 0.0.0.0 --port 8000

Health Check
GET /

Prediction Endpoint
POST /predict

Sample Request
{
  "texts": ["I love this product!", "This is terrible"]
}

Sample Response
[
  {"label": "POSITIVE", "score": 0.99},
  {"label": "NEGATIVE", "score": 0.98}
]

🛠️ Tech Stack
Python
FastAPI
Hugging Face Transformers
PyTorch
MLflow
Uvicorn
