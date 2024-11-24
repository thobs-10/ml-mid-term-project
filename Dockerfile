# stage 1: Build stage
FROM python:3.10-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# RUN brew install pyenv

COPY [ "requirements.txt", "."]

COPY . .

RUN pip install -r requirements.txt

# stage 2: runtime stage
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STAGED_MODEL_PATH="/app/artifacts/staged_model"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create the artifacts directory structure
RUN mkdir -p /app/artifacts/staged_model

# Ensure the model file is in the correct location
COPY artifacts/staged_model/model_pipeline.joblib /app/artifacts/staged_model/

# Copy only the necessary files from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app


# Expose the port the app runs on
EXPOSE 8000

CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["python", "app.py"]
