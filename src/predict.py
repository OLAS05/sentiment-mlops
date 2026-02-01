# src/predict.py
import os
import joblib

class SentimentModel:
    """
    Loads the pretrained Hugging Face sentiment pipeline
    saved as a joblib model. Returns raw model predictions
    without overwriting labels.
    """
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__),
            "../models/sentiment_model/model.joblib"
        )
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}.")
        self.model = joblib.load(model_path)

    def predict(self, texts):
        """
        Returns a list of dicts:
        [{'label': 'POSITIVE', 'score': 0.99}, ...]
        """
        raw_preds = self.model(texts)  # directly call HF pipeline

        # HF pipeline already returns list of dicts with 'label' and 'score'
        # Just normalize labels to uppercase strings
        results = []
        for pred in raw_preds:
            label = str(pred.get("label", "")).upper()
            score = float(pred.get("score", 0.0))
            results.append({"label": label, "score": score})

        return results
