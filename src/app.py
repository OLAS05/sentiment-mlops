from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from predict import SentimentModel

app = FastAPI(title="Sentiment Analysis Application")

# ---------------- MODEL LOAD ----------------
# Load model once at startup
model = SentimentModel()

# ---------------- TEMPLATES ----------------
templates = Jinja2Templates(directory="templates")

# ---------------- API SCHEMA ----------------
class InputData(BaseModel):
    texts: list[str]

# ---------------- UI ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "sentiment": "neutral",
            "text": ""
        }
    )

@app.post("/", response_class=HTMLResponse)
async def analyze_text(request: Request):
    form = await request.form()
    text = form.get("text")

    if not text:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": "Please enter some text",
                "sentiment": "neutral",
                "text": ""
            }
        )

    # -------- Normalize input --------
    text = text.strip()

    # -------- Predict --------
    prediction = model.predict([text])[0]

    # -------- DEBUG: See raw output (remove after verification) --------
    print("RAW PREDICTION >>>", prediction)

    raw_label = str(prediction.get("label", "")).upper()
    score = round(prediction.get("score", 0.0), 3)

    # -------- Automatic universal label mapping --------
    # Handles: POSITIVE/NEGATIVE/NEUTRAL, LABEL_0/1/2, numeric labels
    if "POS" in raw_label or raw_label.endswith("1"):
        sentiment = "positive"
        display = f"Positive 😊 (confidence: {score})"
    elif "NEG" in raw_label or raw_label.endswith("0"):
        sentiment = "negative"
        display = f"Negative 😞 (confidence: {score})"
    elif "NEU" in raw_label or raw_label.endswith("2"):
        sentiment = "neutral"
        display = f"Neutral 😐 (confidence: {score})"
    else:
        # Fallback if new label appears
        sentiment = "neutral"
        display = f"Neutral 😐 (confidence: {score})"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": display,
            "sentiment": sentiment,
            "text": text
        }
    )

# ---------------- API ROUTES ----------------
@app.post("/predict")
def predict_sentiment(input_data: InputData):
    if not input_data.texts:
        raise HTTPException(status_code=400, detail="No texts provided.")

    predictions = model.predict(input_data.texts)
    return {"predictions": predictions}

@app.get("/health")
def health_check():
    return {"status": "ok"}
