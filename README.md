# Titanic Survival Prediction MLOps Project

## Project Overview

This project deploys a Titanic Survival Prediction API using a GitOps workflow managed by Azure Arc and FluxCD. The application is containerized with Docker, served through FastAPI, and deployed to Kubernetes as a microservice exposed through both a local Kubernetes LoadBalancer and a public Cloudflare Tunnel.

The project also includes a browser-based interactive interface built with Gradio, allowing users to interact with the machine learning model through a web UI.

---

# Architecture

```text
GitHub Repository
        ↓
FluxCD GitOps Sync
        ↓
Azure Arc Connected Kubernetes Cluster
        ↓
FastAPI Application
        ├── REST API (/predict)
        └── Gradio GUI (/gui)
        ↓
Kubernetes LoadBalancer Service
        ↓
Cloudflare Tunnel
        ↓
Public HTTPS Endpoint
```

---

# Technical Specifications

| Component | Specification |
|---|---|
| Python Version | Python 3.11 Slim |
| API Framework | FastAPI |
| GUI Framework | Gradio |
| ML Frameworks | Scikit-Learn 1.8.0, XGBoost 3.2.0 |
| Container Runtime | Docker |
| Orchestration | Kubernetes |
| GitOps Tooling | Azure Arc, FluxCD |
| Deployment Strategy | Recreate / Rolling Update |
| Service Type | LoadBalancer |
| Public Exposure | Cloudflare Tunnel |
| Domain | apratim-projects.com |

---

# Public URLs

| Service | URL |
|---|---|
| Root API | `https://api.apratim-projects.com` |
| Prediction Endpoint | `https://api.apratim-projects.com/predict` |
| Gradio GUI | `https://api.apratim-projects.com/gui` |

---

# Cloudflare Tunnel Setup

## Authenticate with Cloudflare

```bash
cloudflared login
```

## Create Named Tunnel

```bash
cloudflared tunnel create titanic-api
```

## Configure Tunnel

```yaml
tunnel: <TUNNEL-UUID>
credentials-file: /home/user/.cloudflared/<UUID>.json

ingress:
  - hostname: api.apratim-projects.com
    service: http://192.168.1.213:80

  - service: http_status:404
```

## Create DNS Route

```bash
cloudflared tunnel route dns titanic-api api.apratim-projects.com
```

## Run Tunnel

```bash
cloudflared tunnel run titanic-api
```

---

# API Example

```bash
curl -X POST "https://api.apratim-projects.com/predict" \
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

---

# Gradio GUI

The project includes a mounted Gradio interface accessible at:

```text
https://api.apratim-projects.com/gui
```

The interface provides:

- Passenger name input
- Sex selector
- Age slider
- Passenger class dropdown
- Fare input
- Family size inputs
- Embarkation selection
- Cabin entry
- Real-time prediction results

---

# Security Recommendations

| Security Layer | Purpose |
|---|---|
| Cloudflare Zero Trust | Authentication and access control |
| API Keys | Restrict anonymous API usage |
| HTTPS Enforcement | Encrypt traffic |
| Rate Limiting | Prevent abuse |
| WAF Rules | Block malicious requests |

---

# Build and Deployment Commands

| Action | Command |
|---|---|
| Build Docker Image | `docker build -t abajpai8/titanic-predictor:v3 .` |
| Push Docker Image | `docker push abajpai8/titanic-predictor:v3` |
| Force Flux Sync | `sudo kubectl annotate kustomization titanic-app-config-titanic -n flux-system reconcile.fluxcd.io/requestedAt=$(date +%s) --overwrite` |

---

# Debugging Commands

| Action | Command |
|---|---|
| Check Pods | `sudo kubectl get pods` |
| View Logs | `sudo kubectl logs -l app=titanic` |
| Describe Pod | `sudo kubectl describe pod <pod-name>` |
| Get Service IP | `sudo kubectl get svc titanic-service` |

---

# Project Summary

This project demonstrates:

- Kubernetes orchestration
- GitOps deployment with FluxCD
- Azure Arc hybrid-cloud management
- FastAPI REST APIs
- Gradio interactive ML interfaces
- Cloudflare secure tunneling
- Public HTTPS deployment
- Containerized machine learning serving

The result is a production-style hybrid-cloud MLOps deployment accessible through both APIs and web interfaces.
