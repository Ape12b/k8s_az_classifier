# Titanic Survival Prediction MLOps Project

## Project Overview

This project deploys a Titanic Survival Prediction API using a GitOps workflow managed by **Azure Arc** and **FluxCD**. The application is containerized with Docker, served through **FastAPI**, and deployed to Kubernetes as a microservice exposed on the local network.

The API uses a dual-model strategy in `main.py`, where predictions are routed to separate male and female survival models. This reflects the different statistical survival patterns in the Titanic dataset.

---

## Technical Specifications

| Component | Specification |
|---|---|
| Python Version | Python 3.11 Slim |
| API Framework | FastAPI |
| ML Frameworks | Scikit-Learn 1.8.0, XGBoost 3.2.0 |
| Container Runtime | Docker |
| Orchestration | Kubernetes |
| GitOps Tooling | Azure Arc, FluxCD |
| Deployment Strategy | Recreate / Rolling Update |
| Service Type | LoadBalancer |
| Exposure | Local network |

---

## Repository Structure

```text
.
├── main.py
├── Dockerfile
├── requirements.txt
├── preprocessor.pkl
├── final_stack_f.pkl
├── final_stack_m.pkl
├── deployment.yaml
├── service.yaml
└── README.md
```

---

## API Endpoint

### Health Check

```http
GET /
```

Expected response:

```json
{
  "message": "Titanic survival API is running"
}
```

### Prediction Endpoint

```http
POST /predict
```

The endpoint accepts Titanic passenger data as JSON and returns a survival prediction.

---

## Sample Request

```bash
curl -X POST "http://192.168.1.213/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Pclass": 3,
       "Name": "Braund, Mr. Owen Harris",
       "Sex": "male",
       "Age": 22,
       "SibSp": 1,
       "Parch": 0,
       "Ticket": "A/5 21171",
       "Fare": 7.25,
       "Cabin": null,
       "Embarked": "S"
     }'
```

Example response:

```json
{
  "survived": 0,
  "sex_model_used": "male"
}
```

---

## Build and Deployment Commands

| Action | Command |
|---|---|
| Build Docker Image | `docker build -t abajpai8/titanic-predictor:v3 .` |
| Push Docker Image | `docker push abajpai8/titanic-predictor:v3` |
| Force Flux Sync | `sudo kubectl annotate kustomization titanic-app-config-titanic -n flux-system reconcile.fluxcd.io/requestedAt=$(date +%s) --overwrite` |

---

## Debugging and Maintenance Commands

| Action | Command |
|---|---|
| Check Pod Status | `sudo kubectl get pods` |
| View App Logs | `sudo kubectl logs -l app=titanic` |
| Inspect Pod Failure | `sudo kubectl describe pod <pod-name>` |
| Get Service IP | `sudo kubectl get svc titanic-service` |
| Check Flux Kustomizations | `sudo kubectl get kustomizations -n flux-system` |

---

## Model Artifacts

The application expects the following files to be available inside the container:

```text
preprocessor.pkl
final_stack_f.pkl
final_stack_m.pkl
```

These files are loaded at FastAPI startup:

```python
preprocessor = joblib.load("preprocessor.pkl")
model_f = joblib.load("final_stack_f.pkl")
model_m = joblib.load("final_stack_m.pkl")
```

---

## Feature Engineering Summary

The API performs real-time feature engineering to match the training pipeline:

- Extracts passenger title from `Name`
- Handles rare and normalized title categories
- Imputes missing `Age` using title-based medians
- Imputes missing `Fare` using the training median
- Creates `Family_Size`
- Creates `IsAlone`
- Extracts `Deck` from `Cabin`
- Adds real-time-safe defaults for:
  - `Ticket_Group_Size`
  - `Family_Survival`
- Creates binned features:
  - `Fare_Bin`
  - `Age_Bin`

The final feature set is transformed with the saved `preprocessor.pkl` before being passed to the appropriate model.

---

## Current Status

The project is operational. Kubernetes pods are running, the service is exposed through a LoadBalancer on the local network, and the `/predict` endpoint returns prediction JSON.

---

## Project Summary

This MLOps implementation transitions Titanic survival research into a production-grade microservice. By using **Azure Arc** and **FluxCD**, the project creates a hybrid-cloud GitOps workflow where local infrastructure operates as an edge Kubernetes node.

The containerized setup ensures that dependencies such as **Scikit-Learn 1.8.0** and **XGBoost 3.2.0** remain consistent across deployments. The FastAPI service provides a clean prediction interface, while Kubernetes handles deployment, scaling, service exposure, and operational management.
