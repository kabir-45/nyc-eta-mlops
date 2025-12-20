# 🚖 NYC Taxi ETA Prediction - MLOps

A production-ready MLOps project that predicts the Estimated Time of Arrival (ETA) for NYC taxi trips. This repository demonstrates a complete End-to-End machine learning lifecycle, from data ingestion to model deployment, using industry-standard tools.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-blue)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue)
![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI%20Pipeline-yellow)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange)
![Grafana](https://img.shields.io/badge/Grafana-Visualization-orange)
![Render](https://img.shields.io/badge/Render-Deployment-black)

---

## ❓ Problem Statement
Predicting the estimated time of arrival (ETA) for taxi rides in a dense urban environment like New York City is a complex challenge. Traffic congestion, time of day, weather conditions, and pickup locations significantly impact travel duration.

Inaccurate ETAs lead to:
* **Poor User Experience:** Riders get frustrated when rides take longer than expected.
* **Inefficient Fleet Management:** Drivers cannot be dispatched effectively if trip durations are unknown.

**The Goal:** Build an automated, scalable machine learning pipeline that continuously trains and deploys models to predict trip duration with high accuracy, minimizing the error (RMSE) between predicted and actual arrival times.

---

The project follows a modular MLOps architecture:

* **Data Pipeline:** Managed by **DVC** to track data versions and processing stages.
* **Experiment Tracking:** **MLflow** tracks parameters, metrics, and stores trained models (hosted on DagsHub).
* **Model Registry:** Production models are versioned and managed via MLflow's Model Registry.
* **Serving:** A **FastAPI** application serves predictions via a REST API.
* **Monitoring:** **Prometheus** collects API metrics and **Grafana** provides real-time dashboards for system health.
* **Containerization:** The application is Dockerized for consistent deployment.
* **CI/CD:** **GitHub Actions** runs automated health checks, training, and deployment on every push.

---

## 🏗️ Architecture
Here is the high-level overview of the MLOps pipeline, including Data Ingestion, Model Training, Deployment, and Monitoring.

![MLOps Architecture](System_Architecture.png)

---

## 📊 Dataset
The project uses the official **NYC Taxi & Limousine Commission (TLC) Trip Record Data**.
* **Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
* **Specific Data Used:** Yellow Taxi Trip Records (January - March 2025)

The raw data is versioned using DVC, allowing us to travel back in time to specific dataset versions.

---

## 🚀 Quick Setup
Follow these steps to run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/kabir-45/nyc-eta-mlops.git](https://github.com/kabir-45/nyc-eta-mlops.git)
cd nyc-eta-mlops
