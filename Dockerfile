FROM python:3.12-slim
WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt
COPY model.pkl app.py ./
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
