# Titanic Survival Prediction API - Engineering & Deployment Guide

This document provides the definitive guide for deploying, managing, and scaling the Titanic Survival Prediction system. As the lead engineer who architected this codebase, I have designed this guide to bridge the gap between data science experimentation and production-grade AWS infrastructure.

---

## 1. Overview
This application is a production-ready Machine Learning inference service built with **FastAPI**. It provides a robust pipeline for predicting passenger survival based on the Titanic dataset, integrating modern MLOps practices using **MLflow** for tracking and **Terraform** for infrastructure as code.

### Architecture
1.  **FastAPI Backend (`app/`)**: The core service handling prediction requests, model loading, and business logic.
2.  **ML Pipeline (`app/pipeline/`)**: A modularized pipeline (Ingestion -> Preprocessing -> Training) that can be triggered to update models.
3.  **Service Layer (`app/services/`)**: Decouples prediction logic from the API routing.
4.  **Gateway (`gateway/`)**: A Node.js-based entry point acting as a reverse proxy or orchestrator for the FastAPI service.
5.  **Infrastructure (`deploy/terraform/`)**: Definitive AWS EC2 resource provisioning.
6.  **CI/CD (`.github/workflows/`)**: Automated testing and quality gates.

### Tech Stack Choices
*   **Python 3.10+**: Utilizes modern type hinting and async capabilities.
*   **FastAPI**: Selected for its high performance (Starlette/Pydantic) and automatic OpenAPI documentation.
*   **Terraform**: Ensures the EC2 environment is reproducible and version-controlled.
*   **MLflow**: Used for lifecycle management, ensuring every model version is tracked with its hyperparameters.
*   **Gunicorn/Uvicorn**: Production-grade ASGI server stack for concurrency.

---

## 2. Prerequisites
Before proceeding, ensure your environment meets these exact specifications:

*   **Python**: `3.10.x` or `3.11.x`
*   **Node.js**: `v18.x+` (for the Gateway service)
*   **Terraform**: `v1.5.0+`
*   **AWS CLI**: Configured with an IAM user having `AdministratorAccess` (for infrastructure provisioning).
*   **Operating System**: Linux (Ubuntu 22.04 recommended) or macOS for local development.

---

## 3. Project Structure
The repository is organized to separate concerns between machine learning logic, API delivery, and infrastructure management.

```text
.
├── .env.example                # Template for required environment variables
├── .github/
│   └── workflows/ci.yml       # GitHub Actions: Runs tests on every push/PR
├── app/                        # Main FastAPI Application
│   ├── api/                    # API Route definitions
│   │   └── routes/predict.py   # Prediction endpoint logic
│   ├── core/                   # Global configuration and logging setup
│   │   ├── config.py           # Pydantic Settings management
│   │   └── logging.py          # Structured JSON logging implementation
│   ├── main.py                 # Application entry point
│   ├── models/                 # Pydantic schemas (Request/Response)
│   ├── pipeline/               # ML Pipeline (Ingestion, Preprocessing, Training)
│   │   ├── data_ingestion.py   # Fetches raw data from remote sources
│   │   ├── preprocessing.py    # Feature engineering and scaling
│   │   └── trainer.py          # Model training and MLflow logging
│   └── services/               # Logic for model loading and inference
│       └── model_service.py    # The ModelService class (Singleton)
├── deploy/
│   └── terraform/              # Infrastructure as Code
│       └── main.tf             # AWS EC2, Security Groups, and VPC resources
├── gateway/
│   └── index.js                # Node.js Gateway service entry point
├── requirements.txt            # Python dependency manifest
├── tests/                      # Pytest suite
│   ├── test_api.py             # Integration tests for FastAPI endpoints
│   └── test_pipeline.py        # Unit tests for the ML pipeline
└── .env                        # Local secret storage (Git ignored)
```

---

## 4. Local Setup

### 4.1 Backend Preparation
Follow these steps to initialize the Python environment:

```bash
# 1. Clone the repository
git clone <repository-url>
cd titanic-prediction-api

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and fill in your AWS credentials and MLflow URI
```

### 4.2 Gateway Preparation
The Node.js gateway provides an additional layer of abstraction:

```bash
cd gateway
npm install
cd ..
```

---

## 5. Environment Variables
The application relies on these variables defined in `app/core/config.py` and `.env`.

| Variable | Required | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `MLFLOW_TRACKING_URI` | Yes | URI for the MLflow tracking server | `http://localhost:5000` |
| `AWS_REGION` | Yes | Target AWS region for Boto3 | `us-east-1` |
| `AWS_ACCESS_KEY_ID` | No* | AWS access key for S3/EC2 access | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY`| No* | AWS secret key | `supersecret...` |
| `DATABASE_URL` | No | Optional Postgres/SQLAlchemy link | `postgresql://user:pass@host/db` |
| `GATEWAY_PORT` | Yes | Port for the Node.js Gateway | `3000` |
| `FASTAPI_SERVICE_URL` | Yes | URL where FastAPI is reachable by Gateway | `http://localhost:8000` |
| `SERVER_PORT` | Yes | Port for the FastAPI server | `8000` |

*\*Note: When running on EC2, use IAM Roles instead of hardcoded keys.*

---

## 6. Running Locally

### 6.1 Training the Model
Before running the API, you must have a trained model. The pipeline automates this:

```bash
# This triggers data_ingestion -> preprocessing -> trainer
# Ensure MLflow is running or MLFLOW_TRACKING_URI is set to a local directory
python -m app.pipeline.trainer
```

### 6.2 Starting the FastAPI Server
For development (with hot-reloading):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

For production simulation:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000
```

### 6.3 Starting the Gateway
In a separate terminal:
```bash
cd gateway
export FASTAPI_SERVICE_URL=http://localhost:8000
export GATEWAY_PORT=3000
node index.js
```

### 6.4 Verification
*   **Swagger Docs**: `http://localhost:8000/docs`
*   **Gateway Status**: `http://localhost:3000/health` (Assuming gateway has a health check)

---

## 7. API Endpoints
All routes are prefixed according to the FastAPI app structure.

| Method | Path | Auth | Description | Example Payload |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | None | Check API vitals | N/A |
| `POST` | `/api/v1/predict` | API Key | Predict Titanic survival | `{"Pclass": 3, "Sex": "male", "Age": 22...}` |
| `GET` | `/api/v1/model/metadata`| Internal | Get current model version/metrics | N/A |

### Example Prediction Request
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "Pclass": 1,
  "Sex": "female",
  "Age": 35,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 71.2833,
  "Embarked": "C"
}'
```

---

## 8. Terraform Infrastructure
The `deploy/terraform/main.tf` file defines the "Source of Truth" for the AWS environment.

### Resources Created
*   **aws_instance.web**: The EC2 instance (t3.medium recommended for ML tasks).
*   **aws_security_group**: Opens ports 80 (Nginx), 8000 (FastAPI), 3000 (Gateway), and 22 (SSH).
*   **aws_iam_role**: Grants the EC2 instance access to S3 buckets for model artifacts.

### Infrastructure Deployment
```bash
cd deploy/terraform

# Initialize Terraform providers
terraform init

# Preview changes
terraform plan -out=tfplan

# Apply changes
terraform apply tfplan
```

### Infrastructure Destruction
To avoid ongoing AWS costs:
```bash
terraform destroy
```

---

## 9. AWS EC2 Deployment
Since this project is not containerized, we deploy directly to the EC2 OS.

### 1. Provision & Connect
After `terraform apply`, get your public IP:
```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### 2. Host Environment Setup
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git nginx

# Clone project to the server
git clone <your-repo-url> /var/www/titanic-api
cd /var/www/titanic-api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Systemd Configuration
Create a service file for FastAPI: `/etc/systemd/system/fastapi_app.service`
```ini
[Unit]
Description=Gunicorn instance to serve FastAPI
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/var/www/titanic-api
Environment="PATH=/var/www/titanic-api/venv/bin"
EnvironmentFile=/var/www/titanic-api/.env
ExecStart=/var/www/titanic-api/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000

[Install]
WantedBy=multi-user.target
```

### 4. Enable and Start
```bash
sudo systemctl daemon-reload
sudo systemctl start fastapi_app
sudo systemctl enable fastapi_app
```

### 5. Nginx Reverse Proxy
Configure Nginx to forward port 80 to port 3000 (Gateway) or 8000 (FastAPI).

---

## 10. Production Checklist
Before considering the deployment "Live":

1.  **Logging**: Ensure `app/core/logging.py` is outputting to a centralized log manager (e.g., CloudWatch) via the `CustomJsonFormatter`.
2.  **Environment**: Verify `.env` file does NOT contain development secrets. Use AWS Parameter Store or Secrets Manager.
3.  **Security**: Change the security group in `main.tf` to restrict port 22 to your specific IP address.
4.  **Monitoring**: Verify MLflow is correctly logging metrics for every training run triggered on the EC2.
5.  **Data Validation**: Check that `app/models/schemas.py` correctly handles `NaN` values as per Pydantic logic.
6.  **Persistence**: If using a local MLflow database, ensure it is backed up. Ideally, use a remote RDS instance.

---

## 11. Troubleshooting

### Scenario 1: `ModelService` fails to load model
*   **Cause**: The training pipeline hasn't run, or the MLflow artifact path is incorrect.
*   **Fix**: Run `python -m app.pipeline.trainer` and verify the output path matches `MLFLOW_TRACKING_URI`.

### Scenario 2: EC2 Connection Refused (Port 8000)
*   **Cause**: Gunicorn/FastAPI is bound to `127.0.0.1` instead of `0.0.0.0`.
*   **Fix**: Update the systemd `ExecStart` or `uvicorn` command to use `--bind 0.0.0.0:8000`.

### Scenario 3: Terraform "Access Denied" during apply
*   **Cause**: Local AWS CLI credentials do not have permission to create IAM Roles or VPCs.
*   **Fix**: Attach `AdministratorAccess` or the specific `AmazonEC2FullAccess` policy to your IAM user.

### Scenario 4: ModuleNotFoundError: No module named 'app'
*   **Cause**: Python path is not set correctly when running from the root.
*   **Fix**: Ensure you are running commands with `python -m app.main` or setting `export PYTHONPATH=$PYTHONPATH:.`.

### Scenario 5: Gateway (Node.js) cannot connect to FastAPI
*   **Cause**: `FASTAPI_SERVICE_URL` in the gateway folder's environment is pointing to the wrong IP or port.
*   **Fix**: Verify the internal IP of the EC2 and update `gateway/.env`.

### Scenario 6: Memory Crashes during Training
*   **Cause**: t2.micro instance has insufficient RAM (1GB) for large Scikit-Learn processing.
*   **Fix**: Update `deploy/terraform/main.tf` to use `t3.medium` (4GB RAM) or larger.

---

**Contact**: For infrastructure escalations, contact the Senior DevOps Engineering team. For model performance issues, refer to the `app/pipeline/trainer.py` logging output.