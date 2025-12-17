# 🚖 NYC Taxi ETA Prediction - MLOps 

A production-ready MLOps project that predicts the Estimated Time of Arrival (ETA) for NYC taxi trips. This repository demonstrates a complete End-to-End machine learning lifecycle, from data ingestion to model deployment, using industry-standard tools.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-blue)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue)
![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI%20Pipeline-yellow)

---

The project follows a modular MLOps architecture:

* **Data Pipeline:** Managed by **DVC** to track data versions and processing stages.
* **Experiment Tracking:** **MLflow** tracks parameters, metrics, and stores trained models.
* **Model Registry:** Production models are versioned and managed via MLflow's Model Registry.
* **Serving:** A **FastAPI** application serves predictions via a REST API.
* **Containerization:** The application is Dockerized for consistent deployment.
* **CI/CD:** **GitHub Actions** runs automated health checks and tests on every push.

---
## 🏗️ Architecture
Here is the high-level overview of the MLOps pipeline, including Data Ingestion, Model Training, and Deployment with Monitoring.

![MLOps Architecture](Gemini_Generated_Image_tghapwtghapwtgha.png)
