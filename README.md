# README: Seoul Bike Sharing System - Demand Prediction

## Project Overview

Urban mobility comfort is significantly enhanced by rental bikes in many cities, making it crucial to have the right number of bikes available at the right time. Waiting times for users must be minimized, and the city needs a stable supply of rental bikes. The challenge lies in predicting the number of bikes required for each hour based on external factors such as weather, date, and time.

The dataset includes hourly records of bike rentals, weather conditions (temperature, humidity, windspeed, visibility, etc.), and date information. This project aims to build a machine learning model that predicts bike demand for a specific hour, ensuring the rental company can optimize resource allocation. Accurate predictions will reduce costs from oversupply and lost revenue from undersupply while improving customer satisfaction.

## Problem Statement

Bike-sharing systems face challenges in balancing supply and demand, often resulting in customer dissatisfaction or unnecessary operational costs. This project addresses the problem by creating a predictive model that estimates bike demand for any given hour based on key features such as weather conditions, date, and time. The predictions will:

- Enable optimized inventory management, ensuring sufficient bikes are available at peak times.
- Prevent oversupply of bikes, which reduces costs associated with underutilized resources.
- Improve customer experience by reducing waiting times and meeting demand effectively.

The solution integrates the machine learning model into the Seoul bike-sharing system, served via a **FastAPI application**. Users interact with the model through API requests made via the app, receiving real-time predictions for bike demand at specific hours.

---

## Lifecycle of Model Creation

1. **Understanding the Problem Statement**  
   Define the project scope and requirements to predict hourly bike demand accurately.

2. **Data Collection**  
   Download the dataset from [[link-to-data](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)].  

3. **Exploratory Data Analysis (EDA)**  
   Explore the data to uncover patterns, trends, and insights.

4. **Data Pre-Processing and Feature Engineering**  
   Handle missing values, create derived features, and select the most relevant features.

5. **Model Training**  
   Train tree-based models (e.g., Random Forest, XGBoost) to establish a robust baseline.

6. **Model Hyperparameter Tuning**  
   Optimize the model by finding the best parameters to improve prediction accuracy.

7. **Model Evaluation**  
   Assess model performance using metrics like RMSE, MAE, and R².

8. **Model Registry**  
   Save the best-performing model for deployment.

9. **Model Deployment**  
   Containerize the model using Docker and push it to DockerHub.

10. **Model Serving**  
    Use FastAPI to expose the model as an endpoint for predictions.

---

## Project Criteria and Achievements

| **Criteria**                          | **Status** |
| ------------------------------------- | ---------- |
| Problem description                   | ✅          |
| Exploratory Data Analysis (EDA)       | ✅          |
| Model Training                        | ✅          |
| Exporting notebook to script          | ✅          |
| Model Deployment                      | ✅          |
| Reproducibility                       | ✅          |
| Dependency and Environment Management | ✅          |
| Containerization                      | ✅          |
| Cloud Deployment                      | ✅          |

---

## How to Run the System

1. **Prepare the Dataset**  
   - Download the data from [insert link here].  
   - Create a folder named `data` and inside it, a subfolder named `raw`. Paste the data into the `raw` folder.

2. **Set Up Environment**  
   - Ensure you have Python 3.10 installed (recommended using `pyenv`).
   - Create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run Pipelines**  
   - Navigate to the `src` folder and run the pipelines:
     ```bash
     python run_pipelines.py
     ```
   - Logs will display progress. You can also run individual pipelines in the `src/pipeline` folder.

4. **Deploy the Model Using Docker**  
   - Build the Docker container:
     ```bash
     docker build -t fastapi-ml-app .
     ```
   - Verify the container contents:
     ```bash
     docker run -it fastapi-ml-app ls -la /app/artifacts
     ```
   - Run the container:
     ```bash
     docker run -p 8000:8000 fastapi-ml-app
     ```

5. **Test the Endpoint**  
   - Use the provided `request.py` script to send requests to the model endpoint and get predictions:
     ```bash
     python request.py
     ```

---

## Automated Deployment with GitHub Actions(Cool part 😎)

As an added feature, the deployment process is automated using GitHub Actions. This enables seamless deployment to DockerHub, allowing you to pull and run the container directly without manual setup:

```bash
docker pull <dockerhub_username>/fastapi-ml-app
docker run -p 8000:8000 <dockerhub_username>/fastapi-ml-app
```

Enjoy real-time predictions for bike demand! 🚴🏽‍♂️