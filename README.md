# ATM Cash Demand Prediction – Guwahati

**Group ID:** G020  
**Problem ID:** MP231  
**Course:** SCSE3040 MLOps – Bennett University

## Members

| Enrolment No. | Name |
|---|---|
| S24CSEU2132 | MOULIKGAUR |
| S24CSEU0737 | HarshiniSaharan |
| S24CSEU2228 | RishabhYadav |
| S24CSEU2087 | SABHYAGAUTAM |

## Problem Description

Predict daily ATM cash withdrawals (`withdrawal_thousand_rs`) for ATMs in Guwahati.  
The client is Guwahati Cooperative Bank. This is a regression task evaluated with MAE.

## Project Structure

```
.
├── README.md
├── requirements.txt          # all libraries pinned
├── params.yaml               # settings read by code at runtime
├── src/atm_demand/           # Python package
│   ├── __init__.py
│   ├── data.py               # data loading and 7-issue audit
│   ├── clean.py              # cleaning pipeline
│   ├── train.py              # model training + MLflow tracking
│   └── monitor.py            # PSI + drift monitoring
├── tests/                    # pytest tests
│   └── test_api.py
├── app/
│   └── main.py               # FastAPI service
├── data/raw/                 # DVC-tracked CSVs
├── models/
│   └── model.joblib          # champion model
├── predictions/
│   └── holdout_predictions.csv
├── reports/
│   └── findings.yaml
├── mlflow.db                 # MLflow SQLite tracking store
├── Dockerfile
├── .dockerignore
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
└── .github/workflows/
    └── ci.yml
```

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Clean data and train v1 model
python -m src.atm_demand.clean
python -m src.atm_demand.train

# 4. Run monitoring and retrain v2
python -m src.atm_demand.monitor

# 5. Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 6. Test the API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"nearby_shops":80,"days_to_salary":5,"footfall_index":60,
       "distance_to_market_km":3.5,"avg_ticket_rs":2500,
       "other_atms_1km":5,"atm_type":"mall","area":"Market"}'

# 7. Run tests
pytest tests/ -v

# 8. Build and run Docker container
docker build -t mp231-api:1.0 .
docker run -p 8000:8000 mp231-api:1.0
```

## MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

## Data Issues Found

Five of the seven possible data issues were present in `train.csv`:

1. **Leaking column** – `cash_reloaded_next_day` (Spearman r=0.988 with target, absent in holdout)
2. **Sentinel value** – `distance_to_market_km` has 9999.0 as a missing-value placeholder (64 rows)
3. **Duplicate rows** – 46 extra copies of records
4. **Target typos** – 11 values typed 10× too large (gap between 1134 and 4076)
5. **Spelling variants** – `atm_type` has 3 real categories written 19 different ways

## Deployment

```bash
# Docker
docker build -t mp231-api:1.0 .
docker run -d -p 8000:8000 mp231-api:1.0

# Kubernetes (requires a local cluster)
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```
