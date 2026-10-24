from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

try:
    model = tf.keras.models.load_model(os.path.join("models", "fake_news_detection_model.keras"))
except Exception as e:
    model = None
    print("Error loading model:", e)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"probability": None})

@app.post("/predict", response_class=HTMLResponse)
def predict_fake_news(request: Request, newsContent: str = Form(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    else:
        # Perform prediction logic
        prediction = model.predict(tf.constant([[newsContent]], dtype=tf.string))
        probability = float(prediction[0][0])
        verdict = "Real" if probability >= 0.5 else "Fake"

    return templates.TemplateResponse(request, "index.html", {
        "probability": probability,
        "verdict": verdict
    })