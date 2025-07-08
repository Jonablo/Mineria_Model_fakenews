from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="FakeNewsDetector")

# Carga tu modelo y vectorizador una sola vez
model = joblib.load("svm_balanced_calibrated.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    vt = vectorizer.transform([item.text])
    proba = model.predict_proba(vt)[0].tolist()
    pred = int(model.predict(vt)[0])
    return {"prediction": pred, "probabilities": proba}
