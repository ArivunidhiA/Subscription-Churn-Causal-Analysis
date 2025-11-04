# Quick Start Guide

## ✅ Setup Complete!

Your Churn Causal Analysis application is ready to deploy.

### What's Been Created:

1. ✅ **app.py** - Full-stack application (FastAPI + Streamlit)
2. ✅ **Sample Data** - `data/sample_churn_data.csv` (5,000 customers)
3. ✅ **Docker Configuration** - Ready for deployment
4. ✅ **All Dependencies** - Listed in requirements.txt

---

## 🚀 Local Development

### Option 1: Direct Python (Recommended for testing)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py
```

The app will:
- Start FastAPI on http://localhost:8000
- Start Streamlit on http://localhost:8501
- Auto-generate streamlit_app.py on first run

### Option 2: Docker (Recommended for deployment)

```bash
# 1. Build the Docker image
docker build -t churn-causal-analysis .

# 2. Run the container
docker run -p 8000:8000 -p 8501:8501 churn-causal-analysis
```

---

## 📊 Using the Application

1. **Open Streamlit UI**: http://localhost:8501
2. **Upload CSV** (or use the sample data):
   - Sample file: `data/sample_churn_data.csv`
3. **Configure Analysis**:
   - Select causal method (IV, RDD, or CausalForest)
   - Choose treatment and outcome columns
   - Set significance level
4. **Run Analysis** - Click "🚀 Run Analysis"
5. **View Results**:
   - Causal Effect vs Naive Correlation
   - Method-specific visualizations
   - Statistical significance metrics

---

## 🧪 Test the Sample Data

The sample data includes:
- **5,000 customers**
- **True causal effect**: -23% (treatment reduces churn)
- **Naive correlation**: ~-34% (confounded by engagement)
- Expected results: Causal method should show ~23% effect, naive shows ~34%

---

## 📈 Expected Results with Sample Data

When analyzing the sample data:
- **Causal Effect (ATE)**: ~23% reduction in churn
- **Naive Correlation**: ~34% (inflated due to confounding)
- **Bias Reduction**: ~32% (showing the value of causal methods)
- **P-value**: < 0.05 (statistically significant)

---

## 🔧 Troubleshooting

### Import Errors
If you see dependency conflicts locally, use Docker - it will install clean versions.

### Port Conflicts
If ports 8000 or 8501 are in use:
- FastAPI: Change port in `app.py` line with `uvicorn.run`
- Streamlit: Change port in `streamlit_app.py` or command line

### File Not Found
- Ensure `data/` directory exists
- Upload CSV through the Streamlit UI

---

## 📦 Deployment

### Render.com
1. Connect your GitHub repository
2. Select "Web Service"
3. Use the Dockerfile
4. Set ports: 8000, 8501

### Railway
1. Connect repository
2. Railway will auto-detect Dockerfile
3. Expose ports 8000 and 8501

---

## 🎯 Next Steps

1. ✅ Sample data generated - ready to test
2. ✅ All files created - ready to deploy
3. 🚀 Run `python app.py` to start
4. 📊 Upload data and analyze!

---

**Status**: ✅ **READY TO RUN**

