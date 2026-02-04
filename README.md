# 🎗️ Breast Cancer Diagnostic System

An end-to-end Machine Learning web application that predicts whether a breast tumor is **Malignant** or **Benign** based on 30 clinical features. This project integrates a Scikit-Learn model with a **FastAPI** backend and is fully **Dockerized** for consistent deployment.

## 🚀 Key Features
- **Machine Learning:** Utilizes a Logistic Regression model trained on the UCI Breast Cancer Wisconsin dataset.
- **FastAPI Backend:** High-performance asynchronous API for handling predictions.
- **Dynamic UI:** A responsive HTML interface built with Jinja2 templates.
- **MLOps Ready:** Includes Docker configuration for containerized deployment and environment consistency.
- **Automated Logging:** Integrated with MLflow principles for experiment tracking (as seen in the development phase).

---

## 🛠️ Project Structure
```text
nti/
├── app.py               # Main FastAPI application logic
├── Dockerfile           # Docker configuration for containerization
├── .dockerignore        # Files/folders to exclude from Docker image
├── requirements.txt     # Python dependencies (Locked versions)
├── model/
│   └── log.pkl          # Serialized (Pickled) Logistic Regression model
└── templates/
    └── index.html       # Web interface (HTML/CSS)
