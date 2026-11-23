"""
Exploratory Data Analysis (EDA) module for Customer Churn.
Computes statistical summaries, segment-level churn rates, and behavioral patterns.
"""

import os
import json
import numpy as np
import pandas as pd

def run_eda(csv_path="data/telco_customer_churn.csv", output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Clean TotalCharges for numeric calculations
    df['TotalCharges_clean'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    df['TotalCharges_clean'] = df['TotalCharges_clean'].fillna(0.0)
    
    total_customers = len(df)
    churn_count = (df['Churn'] == 'Yes').sum()
    non_churn_count = (df['Churn'] == 'No').sum()
    churn_rate = float(churn_count / total_customers)
    
    summary = {
        "dataset_metrics": {
            "total_records": int(total_customers),
            "features_count": int(df.shape[1] - 1), # excluding TotalCharges_clean
            "churn_count": int(churn_count),
            "non_churn_count": int(non_churn_count),
            "churn_rate_pct": round(churn_rate * 100, 2)
        },
        "tenure_analysis": {
            "overall_mean_tenure": round(float(df['tenure'].mean()), 2),
            "overall_median_tenure": round(float(df['tenure'].median()), 2),
            "churned_mean_tenure": round(float(df[df['Churn'] == 'Yes']['tenure'].mean()), 2),
            "retained_mean_tenure": round(float(df[df['Churn'] == 'No']['tenure'].mean()), 2),
        },
        "charges_analysis": {
            "overall_mean_monthly": round(float(df['MonthlyCharges'].mean()), 2),
            "churned_mean_monthly": round(float(df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()), 2),
            "retained_mean_monthly": round(float(df[df['Churn'] == 'No']['MonthlyCharges'].mean()), 2),
            "overall_mean_total": round(float(df['TotalCharges_clean'].mean()), 2),
            "churned_mean_total": round(float(df[df['Churn'] == 'Yes']['TotalCharges_clean'].mean()), 2),
            "retained_mean_total": round(float(df[df['Churn'] == 'No']['TotalCharges_clean'].mean()), 2),
        },
        "segment_churn_rates": {}
    }
    
    # Categorical breakdown
    categorical_cols = [
        'Contract', 'InternetService', 'PaymentMethod', 'TechSupport',
        'OnlineSecurity', 'PaperlessBilling', 'SeniorCitizen', 'Partner',
        'Dependents', 'PhoneService', 'MultipleLines', 'DeviceProtection', 'OnlineBackup'
    ]
    
    for col in categorical_cols:
        cat_summary = {}
        for val, group in df.groupby(col):
            cnt = len(group)
            ch_cnt = (group['Churn'] == 'Yes').sum()
            rate = ch_cnt / cnt if cnt > 0 else 0
            val_str = str(val)
            if col == 'SeniorCitizen':
                val_str = "Yes (Senior)" if val == 1 else "No (Non-Senior)"
            cat_summary[val_str] = {
                "total_customers": int(cnt),
                "churned": int(ch_cnt),
                "churn_rate_pct": round(rate * 100, 2)
            }
        summary["segment_churn_rates"][col] = cat_summary

    # Tenure cohorts
    df['tenure_cohort'] = pd.cut(
        df['tenure'], 
        bins=[-1, 12, 24, 48, 72], 
        labels=['0-12 months', '13-24 months', '25-48 months', '49-72 months']
    )
    cohort_summary = {}
    for cohort, group in df.groupby('tenure_cohort', observed=False):
        cnt = len(group)
        ch_cnt = (group['Churn'] == 'Yes').sum()
        cohort_summary[str(cohort)] = {
            "total_customers": int(cnt),
            "churned": int(ch_cnt),
            "churn_rate_pct": round((ch_cnt / cnt) * 100, 2) if cnt > 0 else 0
        }
    summary["tenure_cohorts"] = cohort_summary
    
    # Save to JSON
    json_path = os.path.join(output_dir, "eda_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"EDA Summary successfully generated and saved to {json_path}")
    print("\n" + "="*50)
    print("KEY EDA FINDINGS:")
    print("="*50)
    print(f"Total Customers: {total_customers:,} | Overall Churn Rate: {summary['dataset_metrics']['churn_rate_pct']}%")
    print(f"Mean Tenure: Retained = {summary['tenure_analysis']['retained_mean_tenure']} mos vs Churned = {summary['tenure_analysis']['churned_mean_tenure']} mos")
    print(f"Mean Monthly Charges: Retained = ${summary['charges_analysis']['retained_mean_monthly']} vs Churned = ${summary['charges_analysis']['churned_mean_monthly']}")
    print("\nChurn Rate by Contract Type:")
    for k, v in summary["segment_churn_rates"]["Contract"].items():
        print(f"  - {k:<15}: {v['churn_rate_pct']}% ({v['churned']}/{v['total_customers']})")
    print("\nChurn Rate by Internet Service:")
    for k, v in summary["segment_churn_rates"]["InternetService"].items():
        print(f"  - {k:<15}: {v['churn_rate_pct']}% ({v['churned']}/{v['total_customers']})")
    print("\nChurn Rate by Tech Support:")
    for k, v in summary["segment_churn_rates"]["TechSupport"].items():
        print(f"  - {k:<25}: {v['churn_rate_pct']}% ({v['churned']}/{v['total_customers']})")
    print("\nChurn Rate by Tenure Cohort:")
    for k, v in summary["tenure_cohorts"].items():
        print(f"  - {k:<15}: {v['churn_rate_pct']}% ({v['churned']}/{v['total_customers']})")
    print("="*50)
    
    return summary

if __name__ == '__main__':
    run_eda()

