# Subscription Churn Causal Analysis

A full-stack web application for analyzing subscription churn using causal inference methods (Instrumental Variables, Regression Discontinuity Design, and Causal Forest).

## Features

- **Causal Inference Methods:**
  - Instrumental Variables (2SLS)
  - Regression Discontinuity Design (RDD)
  - Causal Forest / Uplift Modeling

- **Interactive Dashboard:**
  - CSV data upload
  - Configurable analysis parameters
  - Real-time results visualization
  - Monte Carlo power analysis simulation

- **Key Metrics:**
  - Causal Effect (ATE) vs Naive Correlation
  - Bias reduction percentage
  - Statistical significance and confidence intervals
  - Heterogeneous treatment effects by segment

## Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Access the application:
   - Streamlit UI: http://localhost:8501
   - FastAPI API: http://localhost:8000

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t churn-causal-analysis .
```

2. Run the container:
```bash
docker run -p 8000:8000 -p 8501:8501 churn-causal-analysis
```

### Deployment on Render/Railway

1. Connect your repository to Render/Railway
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python app.py`
4. Expose ports: 8000 (API) and 8501 (Streamlit)

## CSV Data Format

Your CSV file should include at minimum:
- `customer_id`: Unique customer identifier
- `engagement_score`: Customer engagement metric
- `subscription_length`: Length of subscription in days
- `churn_flag`: Binary outcome (0/1)
- `feature_treatment`: Binary treatment indicator (0/1)

Optional columns:
- `signup_date`: Date of signup
- `plan_type`: Subscription plan type
- `country`: Customer country
- `revenue`: Customer revenue

## API Endpoints

- `POST /upload`: Upload CSV file
- `POST /analyze`: Run causal inference analysis
- `POST /simulate`: Run Monte Carlo power analysis

## Example Results

- **Causal Effect (ATE):** 23% (p < 0.05)
- **Naive Correlation:** 41%
- **Bias Reduction:** 44%
- **95% CI:** [18%, 28%]

## Technology Stack

- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Causal Inference:** statsmodels, econml, causalml
- **Visualization:** Plotly
- **Deployment:** Docker

## License

MIT License

