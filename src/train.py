import os
import pandas as pd
import mlflow
import mlflow.pyfunc
from transformers import pipeline

# ---------------- PATHS ----------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "tickets.csv")
EXPERIMENT_NAME = "sentiment-analysis"
MODEL_NAME = "sentiment-model"

# ---------------- PYFUNC MODEL ----------------
class SentimentPyFuncModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for Hugging Face sentiment model.
    This is an inference-only (pretrained) model.
    """

    def load_context(self, context):
        self.model = pipeline("sentiment-analysis")

    def predict(self, context, model_input):
        # Expecting a pandas DataFrame with column "text"
        texts = model_input["text"].tolist()
        return self.model(texts)

# ---------------- TRAIN & LOG ----------------
def train_and_log_model(csv_path: str):
    """
    Logs a pretrained Hugging Face sentiment model to MLflow.
    NOTE:
    - No fine-tuning is performed
    - CSV is used for metadata only
    """

    df = pd.read_csv(csv_path)

    if "message" not in df.columns:
        raise ValueError("CSV must contain a 'message' column")

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="hf-sentiment-pretrained"):
        # -------- Log metadata --------
        mlflow.log_param("model_type", "huggingface_pipeline")
        mlflow.log_param("task", "sentiment-analysis")
        mlflow.log_param("training_type", "pretrained_inference_only")
        mlflow.log_param("data_size", len(df))

        # -------- Log model --------
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SentimentPyFuncModel()
        )

        print("✅ Model logged successfully to MLflow")
        print("📌 Model name:", MODEL_NAME)
        print("⚠️ Note: No fine-tuning performed (pretrained model)")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    train_and_log_model(CSV_PATH)
