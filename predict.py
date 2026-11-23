"""
Interactive and Batch Customer Churn Predictor CLI.
Predicts whether a customer will leave the service, calculates churn probability,
assigns a risk tier, and identifies customer-specific risk drivers.
"""

import os
import sys
import json
import argparse
import joblib
import pandas as pd

DEFAULT_MODEL_PATH = "models/best_churn_model.joblib"

SAMPLE_CUSTOMERS = {
    "high_risk": {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 98.50,
        "TotalCharges": 197.00
    },
    "loyal_retained": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 64,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 62.40,
        "TotalCharges": 3993.60
    },
    "moderate_risk": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 14,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 79.85,
        "TotalCharges": 1117.90
    }
}

def analyze_customer_risk_factors(customer_dict):
    """
    Identifies specific risk elevators and retention anchors for a given customer profile.
    """
    risk_factors = []
    protective_factors = []
    
    # Contract
    if customer_dict.get('Contract') == 'Month-to-month':
        risk_factors.append("Month-to-month contract (flexible cancellation with zero commitment friction)")
    elif customer_dict.get('Contract') == 'Two year':
        protective_factors.append("Two-year contract agreement (high commitment and retention lock-in)")
    elif customer_dict.get('Contract') == 'One year':
        protective_factors.append("One-year contract agreement")
        
    # Tenure
    tenure = customer_dict.get('tenure', 0)
    if tenure <= 6:
        risk_factors.append(f"Critical early tenure window ({tenure} months) - high hazard phase")
    elif tenure >= 36:
        protective_factors.append(f"Long-standing loyalty tenure ({tenure} months)")
        
    # Internet & Support
    if customer_dict.get('InternetService') == 'Fiber optic':
        if customer_dict.get('TechSupport') == 'No':
            risk_factors.append("High-speed Fiber Optic line without Tech Support (unresolved issue risk)")
        if customer_dict.get('OnlineSecurity') == 'No':
            risk_factors.append("No Online Security add-on")
    if customer_dict.get('TechSupport') == 'Yes':
        protective_factors.append("Enrolled in Tech Support")
    if customer_dict.get('OnlineSecurity') == 'Yes':
        protective_factors.append("Enrolled in Online Security")
        
    # Billing & Payments
    if customer_dict.get('PaymentMethod') == 'Electronic check':
        risk_factors.append("Electronic check payment (manual intervention required each month)")
    elif 'automatic' in str(customer_dict.get('PaymentMethod', '')):
        protective_factors.append("Automatic payment billing setup")
        
    if customer_dict.get('MonthlyCharges', 0) >= 85:
        risk_factors.append(f"High monthly charges (${customer_dict.get('MonthlyCharges')}/mo)")
        
    return risk_factors, protective_factors

def predict_single_customer(customer_data, model_path=DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")
        
    pipeline = joblib.load(model_path)
    
    # Format as DataFrame
    if isinstance(customer_data, dict):
        df_input = pd.DataFrame([customer_data])
    else:
        df_input = customer_data
        
    prob = pipeline.predict_proba(df_input)[0, 1]
    pred = pipeline.predict(df_input)[0]
    
    if prob >= 0.60:
        risk_tier = "HIGH RISK"
        color_indicator = "[RED]"
    elif prob >= 0.35:
        risk_tier = "MODERATE RISK"
        color_indicator = "[YELLOW]"
    else:
        risk_tier = "LOW RISK"
        color_indicator = "[GREEN]"
        
    cust_dict = df_input.iloc[0].to_dict()
    risks, protections = analyze_customer_risk_factors(cust_dict)
    
    return {
        "prediction": "CHURN" if pred == 1 else "RETAIN",
        "churn_probability_pct": round(float(prob * 100), 2),
        "risk_tier": risk_tier,
        "color_indicator": color_indicator,
        "identified_risk_factors": risks,
        "identified_protective_factors": protections
    }

def print_prediction_report(profile_name, result, cust_data):
    print("\n" + "="*65)
    print(f" CUSTOMER CHURN RISK REPORT: {profile_name.upper()}")
    print("="*65)
    print(f" Outcome Prediction  : {result['prediction']}")
    print(f" Churn Probability   : {result['churn_probability_pct']}%")
    print(f" Risk Assessment     : {result['color_indicator']} {result['risk_tier']}")
    print("-" * 65)
    print(" Profile Snapshot:")
    print(f"   - Contract       : {cust_data.get('Contract')}")
    print(f"   - Tenure         : {cust_data.get('tenure')} months")
    print(f"   - Internet       : {cust_data.get('InternetService')}")
    print(f"   - Tech Support   : {cust_data.get('TechSupport')}")
    print(f"   - Monthly Charges: ${cust_data.get('MonthlyCharges'):.2f}")
    print(f"   - Payment Method : {cust_data.get('PaymentMethod')}")
    print("-" * 65)
    print(" Key Churn Risk Drivers:")
    if result['identified_risk_factors']:
        for rf in result['identified_risk_factors']:
            print(f"   [!] {rf}")
    else:
        print("   (None detected)")
        
    print("\n Key Protective Anchors:")
    if result['identified_protective_factors']:
        for pf in result['identified_protective_factors']:
            print(f"   [*] {pf}")
    else:
        print("   (None detected)")
    print("="*65)

def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction CLI")
    parser.add_argument("--profile", choices=["high_risk", "loyal_retained", "moderate_risk", "all"], 
                        default="all", help="Evaluate built-in benchmark profiles")
    parser.add_argument("--file", type=str, default=None, help="Path to CSV file with customer records")
    args = parser.parse_args()
    
    if args.file:
        df = pd.read_csv(args.file)
        pipeline = joblib.load(DEFAULT_MODEL_PATH)
        preds = pipeline.predict(df)
        probas = pipeline.predict_proba(df)[:, 1]
        df['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in preds]
        df['Churn_Probability'] = (probas * 100).round(2)
        out_csv = "batch_churn_predictions.csv"
        df.to_csv(out_csv, index=False)
        print(f"Successfully processed {len(df)} records. Saved predictions to {out_csv}")
    else:
        profiles_to_run = SAMPLE_CUSTOMERS.keys() if args.profile == "all" else [args.profile]
        for p_name in profiles_to_run:
            cust = SAMPLE_CUSTOMERS[p_name]
            res = predict_single_customer(cust)
            print_prediction_report(p_name, res, cust)

if __name__ == '__main__':
    main()

