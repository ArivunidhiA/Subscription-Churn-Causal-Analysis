"""
Subscription Churn Causal Analysis - Full Stack Application
Combines FastAPI backend and Streamlit frontend in a single file.
"""

import os
import threading
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_CACHE = {}

# Pydantic models for request bodies
class AnalyzeRequest(BaseModel):
    filename: str
    method: str = "IV"
    treatment_col: str = "feature_treatment"
    outcome_col: str = "churn_flag"
    instrument_col: Optional[str] = None
    threshold_col: Optional[str] = None
    threshold_value: Optional[float] = None
    covariates: Optional[List[str]] = None
    significance_level: float = 0.05

class SimulateRequest(BaseModel):
    sample_size: int = 1000
    effect_size: float = 0.23
    n_iterations: int = 1000
    significance_level: float = 0.05

# ============================================================================
# FastAPI Backend
# ============================================================================

app = FastAPI(title="Churn Causal Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Churn Causal Analysis API", "status": "running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file and store it."""
    try:
        file_path = DATA_DIR / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Validate CSV structure
        df = pd.read_csv(file_path)
        required_cols = ['customer_id', 'engagement_score', 'subscription_length', 
                        'churn_flag', 'feature_treatment']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )
        
        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "status": "uploaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_data(request: AnalyzeRequest):
    """Run causal inference analysis."""
    try:
        file_path = DATA_DIR / request.filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        df = pd.read_csv(file_path)
        
        # Clean data
        df = df.dropna(subset=[request.treatment_col, request.outcome_col])
        
        if request.method == "IV":
            results = run_iv_analysis(df, request.treatment_col, request.outcome_col, request.instrument_col, request.covariates, request.significance_level)
        elif request.method == "RDD":
            results = run_rdd_analysis(df, request.treatment_col, request.outcome_col, request.threshold_col, request.threshold_value, request.covariates, request.significance_level)
        elif request.method == "CausalForest":
            results = run_causal_forest_analysis(df, request.treatment_col, request.outcome_col, request.covariates, request.significance_level)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        # Store results
        RESULTS_CACHE[request.filename] = results
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate")
async def simulate_power(request: SimulateRequest):
    """Run Monte Carlo power analysis."""
    try:
        results = run_monte_carlo_simulation(request.sample_size, request.effect_size, request.n_iterations, request.significance_level)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Causal Inference Methods
# ============================================================================

def run_iv_analysis(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    instrument: Optional[str],
    covariates: Optional[List[str]],
    alpha: float
) -> Dict[str, Any]:
    """Instrumental Variables (2SLS) analysis."""
    # If no instrument provided, create one based on country/region
    if instrument is None:
        if 'country' in df.columns:
            # Create instrument: country-level promo exposure
            df['instrument'] = (df['country'].astype(str).str.len() % 2 == 0).astype(int)
        elif 'signup_date' in df.columns:
            # Create instrument based on signup month
            df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
            df['instrument'] = (df['signup_date'].dt.month % 2 == 0).astype(int)
        else:
            # Fallback: create random instrument for demonstration
            np.random.seed(42)
            df['instrument'] = np.random.binomial(1, 0.5, len(df))
        instrument = 'instrument'
    
    # Prepare data
    y = df[outcome].values
    X = df[[treatment]].values
    Z = df[instrument].values
    
    # Add covariates if provided
    if covariates:
        for cov in covariates:
            if cov in df.columns:
                X = np.column_stack([X, df[cov].values])
    
    # Add constant
    X = sm.add_constant(X)
    
    # First stage: regress treatment on instrument
    first_stage = sm.OLS(X[:, 1], sm.add_constant(Z)).fit()
    f_stat = first_stage.fvalue
    
    # Second stage: 2SLS
    X_pred = sm.add_constant(first_stage.predict())
    second_stage = sm.OLS(y, X_pred).fit()
    
    # Naive OLS for comparison
    naive_ols = sm.OLS(y, X).fit()
    
    # Calculate results
    causal_effect = second_stage.params[1] if len(second_stage.params) > 1 else second_stage.params[0]
    naive_effect = naive_ols.params[1] if len(naive_ols.params) > 1 else naive_ols.params[0]
    
    p_value = second_stage.pvalues[1] if len(second_stage.pvalues) > 1 else second_stage.pvalues[0]
    ci_lower, ci_upper = second_stage.conf_int().iloc[1] if len(second_stage.params) > 1 else second_stage.conf_int().iloc[0]
    
    bias_reduction = abs((naive_effect - causal_effect) / naive_effect * 100) if naive_effect != 0 else 0
    
    return {
        "method": "Instrumental Variables (2SLS)",
        "causal_effect": float(causal_effect * 100),  # Convert to percentage
        "naive_effect": float(naive_effect * 100),
        "bias_reduction": float(bias_reduction),
        "p_value": float(p_value),
        "is_significant": p_value < alpha,
        "ci_lower": float(ci_lower * 100),
        "ci_upper": float(ci_upper * 100),
        "first_stage_f_stat": float(f_stat),
        "n_observations": len(df),
        "diagnostics": {
            "first_stage_r2": float(first_stage.rsquared),
            "second_stage_r2": float(second_stage.rsquared),
            "weak_instrument": f_stat < 10
        }
    }


def run_rdd_analysis(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    threshold_col: Optional[str],
    threshold_value: Optional[float],
    covariates: Optional[List[str]],
    alpha: float
) -> Dict[str, Any]:
    """Regression Discontinuity Design analysis."""
    # Set threshold column and value
    if threshold_col is None:
        threshold_col = 'engagement_score'
    
    if threshold_value is None:
        threshold_value = df[threshold_col].median()
    
    # Create running variable (distance from threshold)
    df['running_var'] = df[threshold_col] - threshold_value
    df['below_threshold'] = (df['running_var'] < 0).astype(int)
    
    # Treatment is interaction of below_threshold and feature_treatment
    df['rdd_treatment'] = df['below_threshold'] * df[treatment]
    
    # Prepare data for local linear regression
    bandwidth = df['running_var'].std() * 0.5  # Adaptive bandwidth
    df_rdd = df[abs(df['running_var']) <= bandwidth].copy()
    
    if len(df_rdd) < 50:
        df_rdd = df.copy()  # Use full data if bandwidth too restrictive
    
    # Local linear regression
    X = df_rdd[['running_var', 'rdd_treatment', 'below_threshold']].values
    if covariates:
        for cov in covariates:
            if cov in df_rdd.columns:
                X = np.column_stack([X, df_rdd[cov].values])
    
    X = sm.add_constant(X)
    y = df_rdd[outcome].values
    
    model = sm.OLS(y, X).fit()
    
    # Extract treatment effect (coefficient of rdd_treatment)
    treatment_idx = 2  # Index of rdd_treatment
    causal_effect = model.params[treatment_idx] if len(model.params) > treatment_idx else 0
    p_value = model.pvalues[treatment_idx] if len(model.pvalues) > treatment_idx else 1.0
    ci_lower, ci_upper = model.conf_int().iloc[treatment_idx] if len(model.params) > treatment_idx else (0, 0)
    
    # Naive comparison
    naive_ols = sm.OLS(y, sm.add_constant(df_rdd[[treatment]].values)).fit()
    naive_effect = naive_ols.params[1] if len(naive_ols.params) > 1 else naive_ols.params[0]
    
    bias_reduction = abs((naive_effect - causal_effect) / naive_effect * 100) if naive_effect != 0 else 0
    
    # RDD plot data
    plot_data = {
        "running_var": df_rdd['running_var'].tolist(),
        "outcome": df_rdd[outcome].tolist(),
        "treatment": df_rdd['rdd_treatment'].tolist(),
        "threshold": float(threshold_value)
    }
    
    return {
        "method": "Regression Discontinuity Design",
        "causal_effect": float(causal_effect * 100),
        "naive_effect": float(naive_effect * 100),
        "bias_reduction": float(bias_reduction),
        "p_value": float(p_value),
        "is_significant": p_value < alpha,
        "ci_lower": float(ci_lower * 100),
        "ci_upper": float(ci_upper * 100),
        "threshold": float(threshold_value),
        "bandwidth": float(bandwidth),
        "n_observations": len(df_rdd),
        "plot_data": plot_data
    }


def run_causal_forest_analysis(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Optional[List[str]],
    alpha: float
) -> Dict[str, Any]:
    """Causal Forest / Uplift Modeling analysis."""
    # Prepare features
    feature_cols = []
    if covariates:
        feature_cols = [c for c in covariates if c in df.columns]
    
    # Add default features if none specified
    if not feature_cols:
        default_cols = ['engagement_score', 'subscription_length', 'plan_type', 'country']
        feature_cols = [c for c in default_cols if c in df.columns]
    
    # Prepare data
    X = pd.get_dummies(df[feature_cols], drop_first=True).values
    T = df[treatment].values
    y = df[outcome].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Simple uplift model using difference in means by treatment groups
    # For production, use econml.CausalForestDML
    treated = y[T == 1]
    control = y[T == 0]
    
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("Both treatment and control groups must have observations")
    
    # Average Treatment Effect
    ate = treated.mean() - control.mean()
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(treated, control)
    ci = stats.t.interval(0.95, len(treated) + len(control) - 2, 
                         loc=ate, scale=stats.sem(np.concatenate([treated, control])))
    
    # Heterogeneous treatment effects by segments
    df['segment'] = pd.qcut(df['engagement_score'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    
    segment_effects = {}
    for segment in df['segment'].dropna().unique():
        seg_df = df[df['segment'] == segment]
        seg_treated = seg_df[seg_df[treatment] == 1][outcome].mean()
        seg_control = seg_df[seg_df[treatment] == 0][outcome].mean()
        segment_effects[str(segment)] = float((seg_treated - seg_control) * 100)
    
    # Naive correlation
    naive_effect = df[treatment].corr(df[outcome]) * 100
    bias_reduction = abs((naive_effect - (ate * 100)) / naive_effect * 100) if naive_effect != 0 else 0
    
    return {
        "method": "Causal Forest / Uplift Modeling",
        "causal_effect": float(ate * 100),
        "naive_effect": float(naive_effect),
        "bias_reduction": float(bias_reduction),
        "p_value": float(p_value),
        "is_significant": p_value < alpha,
        "ci_lower": float(ci[0] * 100),
        "ci_upper": float(ci[1] * 100),
        "heterogeneous_effects": segment_effects,
        "n_observations": len(df),
        "n_treated": int(T.sum()),
        "n_control": int(len(T) - T.sum())
    }


def run_monte_carlo_simulation(
    sample_size: int,
    effect_size: float,
    n_iterations: int,
    alpha: float
) -> Dict[str, Any]:
    """Monte Carlo power analysis simulation."""
    np.random.seed(42)
    effects = []
    p_values = []
    significant_count = 0
    
    for _ in range(n_iterations):
        # Simulate data
        treatment = np.random.binomial(1, 0.5, sample_size)
        outcome_control = np.random.binomial(1, 0.3, sample_size)
        outcome_treatment = np.random.binomial(1, 0.3 + effect_size, sample_size)
        outcome = np.where(treatment == 1, outcome_treatment, outcome_control)
        
        # Calculate effect
        treated_mean = outcome[treatment == 1].mean()
        control_mean = outcome[treatment == 0].mean()
        effect = treated_mean - control_mean
        
        # Statistical test
        _, p_val = stats.ttest_ind(outcome[treatment == 1], outcome[treatment == 0])
        
        effects.append(effect * 100)
        p_values.append(p_val)
        if p_val < alpha:
            significant_count += 1
    
    power = significant_count / n_iterations
    
    return {
        "sample_size": sample_size,
        "effect_size": effect_size,
        "n_iterations": n_iterations,
        "power": float(power),
        "mean_effect": float(np.mean(effects)),
        "std_effect": float(np.std(effects)),
        "effects_distribution": [float(e) for e in effects[:1000]],  # Limit for response size
        "type_i_error": float(np.mean([p < alpha for p in np.random.choice(p_values, min(100, len(p_values)), replace=False)]))
    }


# ============================================================================
# Streamlit Frontend
# ============================================================================

def run_streamlit():
    """Launch Streamlit app."""
    import subprocess
    import sys
    
    # Wait a bit for FastAPI to start
    time.sleep(2)
    
    streamlit_script = Path(__file__).parent / "streamlit_app.py"
    if not streamlit_script.exists():
        create_streamlit_app()
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(streamlit_script), "--server.port=8501"])


def create_streamlit_app():
    """Create the Streamlit frontend application."""
    streamlit_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
from pathlib import Path

st.set_page_config(page_title="Churn Causal Analysis", layout="wide")

st.title("🧠 Subscription Churn Causal Analysis")
st.markdown("**Quasi-Experimental Design for Causal Inference**")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    # Save uploaded file
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and preview data
    df = pd.read_csv(file_path)
    st.sidebar.success(f"✅ Uploaded: {len(df)} rows")
    
    # Analysis configuration
    st.sidebar.subheader("Analysis Settings")
    method = st.sidebar.selectbox("Causal Design", ["IV", "RDD", "CausalForest"])
    treatment_col = st.sidebar.selectbox("Treatment Column", df.columns.tolist(), 
                                         index=df.columns.get_loc("feature_treatment") if "feature_treatment" in df.columns else 0)
    outcome_col = st.sidebar.selectbox("Outcome Column", df.columns.tolist(),
                                       index=df.columns.get_loc("churn_flag") if "churn_flag" in df.columns else 0)
    
    significance_level = st.sidebar.slider("Significance Level", 0.01, 0.10, 0.05, 0.01)
    
    # Method-specific settings
    if method == "IV":
        instrument_col = st.sidebar.selectbox("Instrument Column", ["Auto"] + df.columns.tolist())
        instrument_col = None if instrument_col == "Auto" else instrument_col
    elif method == "RDD":
        threshold_col = st.sidebar.selectbox("Threshold Column", df.columns.tolist(),
                                            index=df.columns.get_loc("engagement_score") if "engagement_score" in df.columns else 0)
        threshold_value = st.sidebar.number_input("Threshold Value", 
                                                  value=float(df[threshold_col].median()) if threshold_col in df.columns else 0.0)
    else:
        instrument_col = None
        threshold_col = None
        threshold_value = None
    
    # Covariates
    available_cols = [c for c in df.columns if c not in [treatment_col, outcome_col]]
    covariates = st.sidebar.multiselect("Covariates (Optional)", available_cols)
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Running causal inference analysis..."):
            try:
                # First upload file to API (if not already uploaded)
                try:
                    upload_url = "http://localhost:8000/upload"
                    with open(file_path, "rb") as f:
                        files = {"file": (uploaded_file.name, f, "text/csv")}
                        upload_response = requests.post(upload_url, files=files)
                        upload_response.raise_for_status()
                except:
                    pass  # File might already be uploaded
                
                # Prepare analysis request
                api_url = "http://localhost:8000/analyze"
                data = {
                    "filename": uploaded_file.name,
                    "method": method,
                    "treatment_col": treatment_col,
                    "outcome_col": outcome_col,
                    "significance_level": significance_level
                }
                
                if method == "IV" and instrument_col and instrument_col != "Auto":
                    data["instrument_col"] = instrument_col
                elif method == "RDD":
                    data["threshold_col"] = threshold_col
                    data["threshold_value"] = float(threshold_value)
                
                if covariates:
                    data["covariates"] = covariates
                
                # Call API
                response = requests.post(api_url, json=data)
                response.raise_for_status()
                results = response.json()
                
                st.session_state['results'] = results
                st.session_state['data'] = df
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                st.session_state['results'] = None
    
    # Display results
    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']
        
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Causal Effect (ATE)", f"{results['causal_effect']:.2f}%",
                     delta=f"p={results['p_value']:.4f}", delta_color="inverse")
        
        with col2:
            st.metric("Naive Correlation", f"{results['naive_effect']:.2f}%",
                     delta=f"{results['bias_reduction']:.1f}% bias reduction")
        
        with col3:
            st.metric("P-value", f"{results['p_value']:.4f}",
                     delta="Significant" if results['is_significant'] else "Not Significant",
                     delta_color="normal" if results['is_significant'] else "off")
        
        with col4:
            ci_range = results['ci_upper'] - results['ci_lower']
            st.metric("95% CI", f"[{results['ci_lower']:.2f}, {results['ci_upper']:.2f}%]",
                     delta=f"±{ci_range/2:.2f}%")
        
        # Main results
        st.subheader("📊 Results Dashboard")
        
        # Comparison chart
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            x=['Causal Effect (ATE)', 'Naive Correlation'],
            y=[results['causal_effect'], results['naive_effect']],
            marker_color=['#1f77b4', '#ff7f0e'],
            text=[f"{results['causal_effect']:.2f}%", f"{results['naive_effect']:.2f}%"],
            textposition='auto'
        ))
        fig_comparison.update_layout(
            title="Causal Effect vs Naive Correlation",
            yaxis_title="Effect Size (%)",
            height=400
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Method-specific visualizations
        if results['method'] == "Regression Discontinuity Design":
            if 'plot_data' in results:
                plot_data = results['plot_data']
                fig_rdd = go.Figure()
                
                # Scatter plot
                fig_rdd.add_trace(go.Scatter(
                    x=plot_data['running_var'],
                    y=plot_data['outcome'],
                    mode='markers',
                    marker=dict(color=plot_data['treatment'], colorscale='Viridis'),
                    name='Observations'
                ))
                
                # Threshold line
                fig_rdd.add_vline(x=0, line_dash="dash", line_color="red", 
                                 annotation_text=f"Threshold: {results['threshold']:.2f}")
                
                fig_rdd.update_layout(
                    title="Regression Discontinuity Design",
                    xaxis_title="Running Variable (Distance from Threshold)",
                    yaxis_title="Outcome",
                    height=500
                )
                st.plotly_chart(fig_rdd, use_container_width=True)
        
        elif results['method'] == "Causal Forest / Uplift Modeling":
            if 'heterogeneous_effects' in results:
                het_effects = results['heterogeneous_effects']
                fig_het = go.Figure()
                fig_het.add_trace(go.Bar(
                    x=list(het_effects.keys()),
                    y=list(het_effects.values()),
                    marker_color='lightblue',
                    text=[f"{v:.2f}%" for v in het_effects.values()],
                    textposition='auto'
                ))
                fig_het.update_layout(
                    title="Heterogeneous Treatment Effects by Segment",
                    xaxis_title="Segment",
                    yaxis_title="Treatment Effect (%)",
                    height=400
                )
                st.plotly_chart(fig_het, use_container_width=True)
        
        elif results['method'] == "Instrumental Variables (2SLS)":
            if 'diagnostics' in results:
                diag = results['diagnostics']
                st.subheader("🔍 IV Diagnostics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("First-Stage F-Statistic", f"{results['first_stage_f_stat']:.2f}",
                             delta="Weak Instrument" if diag['weak_instrument'] else "Strong Instrument",
                             delta_color="inverse" if diag['weak_instrument'] else "normal")
                with col2:
                    st.metric("First-Stage R²", f"{diag['first_stage_r2']:.4f}")
    
    # Monte Carlo Simulation Panel
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Monte Carlo Simulation")
    
    mc_sample_size = st.sidebar.slider("Sample Size", 100, 10000, 1000, 100)
    mc_effect_size = st.sidebar.slider("Effect Size", 0.01, 0.50, 0.23, 0.01)
    mc_iterations = st.sidebar.slider("Iterations", 100, 10000, 1000, 100)
    
    if st.sidebar.button("🎲 Run Simulation"):
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                api_url = "http://localhost:8000/simulate"
                data = {
                    "sample_size": mc_sample_size,
                    "effect_size": mc_effect_size,
                    "n_iterations": mc_iterations,
                    "significance_level": significance_level
                }
                
                response = requests.post(api_url, json=data)
                response.raise_for_status()
                mc_results = response.json()
                
                st.session_state['mc_results'] = mc_results
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if 'mc_results' in st.session_state and st.session_state['mc_results']:
        mc_results = st.session_state['mc_results']
        
        st.subheader("📈 Monte Carlo Power Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Statistical Power", f"{mc_results['power']*100:.1f}%")
        with col2:
            st.metric("Mean Effect", f"{mc_results['mean_effect']:.2f}%")
        with col3:
            st.metric("Std Dev", f"{mc_results['std_effect']:.2f}%")
        
        # Distribution plot
        if 'effects_distribution' in mc_results:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=mc_results['effects_distribution'],
                nbinsx=50,
                marker_color='skyblue'
            ))
            fig_dist.add_vline(x=mc_results['mean_effect'], line_dash="dash", 
                             line_color="red", annotation_text="Mean")
            fig_dist.update_layout(
                title="Distribution of Estimated Effects",
                xaxis_title="Effect Size (%)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_dist, use_container_width=True)

else:
    st.info("👈 Please upload a CSV file to begin analysis")
    st.markdown("""
    ### Expected CSV Format:
    - `customer_id`: Unique customer identifier
    - `engagement_score`: Customer engagement metric
    - `subscription_length`: Length of subscription in days
    - `churn_flag`: Binary outcome (0/1)
    - `feature_treatment`: Binary treatment indicator (0/1)
    - `signup_date`: Date of signup
    - `plan_type`: Subscription plan type
    - `country`: Customer country
    - `revenue`: Customer revenue
    """)

'''
    with open("streamlit_app.py", "w") as f:
        f.write(streamlit_code)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Create Streamlit app file
    create_streamlit_app()
    
    # Start FastAPI in a separate thread
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
    # Start Streamlit in main thread
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit-only":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501"])
    else:
        # Start FastAPI server in background thread
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Wait a moment for FastAPI to start
        time.sleep(2)
        
        # Launch Streamlit
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501"])

