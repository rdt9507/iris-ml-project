FROM python:3.11-slim

WORKDIR /app

COPY iris_ML_Project.py .
COPY models ./models

RUN pip install --no-cache-dir fastapi uvicorn joblib scikit-learn pydantic

EXPOSE 8000

CMD ["uvicorn", "iris_ML_Project:app", "--host", "0.0.0.0", "--port", "8000"]
