"""
Generate sample CSV data for testing the Churn Causal Analysis application.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

n_customers = 5000

# Generate customer IDs
customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]

# Generate engagement scores (0-100)
engagement_scores = np.random.normal(50, 20, n_customers)
engagement_scores = np.clip(engagement_scores, 0, 100)

# Generate subscription lengths (days)
subscription_lengths = np.random.lognormal(4.5, 0.8, n_customers).astype(int)
subscription_lengths = np.clip(subscription_lengths, 1, 2000)

# Generate treatment (feature_treatment) - correlated with engagement
# Higher engagement customers more likely to get treatment
treatment_prob = 1 / (1 + np.exp(-(engagement_scores - 50) / 10))
feature_treatment = np.random.binomial(1, treatment_prob, n_customers)

# Generate churn flag
# True causal effect: treatment reduces churn by 23%
# But engagement also affects churn (confounding)
true_effect = -0.23
churn_prob = 0.3 - 0.002 * engagement_scores + true_effect * feature_treatment
churn_prob = np.clip(churn_prob, 0.01, 0.99)
churn_flag = np.random.binomial(1, churn_prob, n_customers)

# Generate signup dates (last 2 years)
start_date = datetime.now() - timedelta(days=730)
signup_dates = [start_date + timedelta(days=np.random.randint(0, 730)) 
                for _ in range(n_customers)]

# Generate plan types
plan_types = np.random.choice(['Basic', 'Premium', 'Enterprise'], 
                              n_customers, p=[0.5, 0.3, 0.2])

# Generate countries
countries = np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia'], 
                            n_customers, p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.1])

# Generate revenue (correlated with plan type and engagement)
revenue_base = {'Basic': 20, 'Premium': 50, 'Enterprise': 150}
revenue = np.array([float(revenue_base[pt]) for pt in plan_types])
revenue += np.random.normal(0, revenue * 0.2)
revenue += engagement_scores * 0.5
revenue = np.maximum(revenue, 5).astype(float)

# Create DataFrame
df = pd.DataFrame({
    'customer_id': customer_ids,
    'engagement_score': engagement_scores,
    'subscription_length': subscription_lengths,
    'churn_flag': churn_flag,
    'feature_treatment': feature_treatment,
    'signup_date': signup_dates,
    'plan_type': plan_types,
    'country': countries,
    'revenue': revenue
})

# Save to CSV
output_file = 'data/sample_churn_data.csv'
df.to_csv(output_file, index=False)

print(f"✅ Generated sample data with {len(df)} customers")
print(f"📁 Saved to: {output_file}")
print(f"\n📊 Summary Statistics:")
print(f"   Churn rate: {df['churn_flag'].mean():.2%}")
print(f"   Treatment rate: {df['feature_treatment'].mean():.2%}")
print(f"   Mean engagement: {df['engagement_score'].mean():.1f}")
print(f"\n📈 Naive correlation (treatment vs churn): {df['feature_treatment'].corr(df['churn_flag']):.4f}")
print(f"📈 True causal effect: {true_effect:.2%}")

