# Predictive Maintenance Live API

**Live prediction and anomaly detection backend for machine-tool predictive maintenance.**
**Team Member 1 Role**

## Setup
1. Requires Python 3.11
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`

## Data Configuration
* **AI4I 2020 Data:** Place `ai4i2020.csv` (UCI dataset 601) in `data/raw/`.
* **C-MAPSS FD001 Data:** Place files in `data/raw/cmapss/` (For Member 2).

## Run Order (Model Training)
Execute the following modules in sequence from the root directory:
1. `python -m src.preprocessing`
2. `python -m src.failure_model`
3. `python -m src.anomaly`
4. `python -m src.decision`

## Running the API
Start the FastAPI server:
```bash
uvicorn src.api:app --reload