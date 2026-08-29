from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path


# Membuat aplikasi FastAPI
app = FastAPI(
    title="Iris KNN Classification API",
    description="API untuk melakukan prediksi bunga Iris menggunakan model KNN",
    version="1.0.0"
)


# Lokasi file model
MODEL_PATH = Path(__file__).resolve().parent / "models" / "iris_knn_model.pkl"


# Load model
model = joblib.load(MODEL_PATH)


# Struktur input data
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Endpoint utama
@app.get("/")
def home():
    return {
        "message": "Iris KNN Classification API is running",
        "model": "KNN",
        "accuracy": 1.0
    }


# Endpoint untuk prediksi
@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }