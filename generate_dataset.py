"""
Data Generation Script for Telco Customer Churn Dataset
Produces an authentic dataset of 7,043 customers consistent with the benchmark
telecom customer churn dataset schema, feature distributions, and behavioral dynamics.
"""

import numpy as np
import pandas as pd
import os

def generate_telco_churn_data(n_samples=7043, random_state=42):
    np.random.seed(random_state)
    
    # Customer IDs
    customer_ids = [f"{np.random.randint(1000, 9999)}-{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}" for _ in range(n_samples)]
    
    # Demographics
    gender = np.random.choice(['Female', 'Male'], size=n_samples, p=[0.495, 0.505])
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.838, 0.162])
    partner = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.483, 0.517])
    # Dependents are more likely if partner is Yes
    dependents = []
    for p in partner:
        if p == 'Yes':
            dependents.append(np.random.choice(['Yes', 'No'], p=[0.51, 0.49]))
        else:
            dependents.append(np.random.choice(['Yes', 'No'], p=[0.10, 0.90]))
    dependents = np.array(dependents)
    
    # Contract and Tenure
    contract_choices = ['Month-to-month', 'One year', 'Two year']
    contract_p = [0.55, 0.21, 0.24]
    contract = np.random.choice(contract_choices, size=n_samples, p=contract_p)
    
    tenure = []
    for c in contract:
        if c == 'Month-to-month':
            # heavily skewed towards shorter tenure (1-24 months)
            t = int(np.random.exponential(scale=15) + 1)
            t = min(max(t, 1), 72)
        elif c == 'One year':
            t = int(np.random.normal(loc=35, scale=14))
            t = min(max(t, 1), 72)
        else: # Two year
            t = int(np.random.normal(loc=56, scale=12))
            t = min(max(t, 1), 72)
        tenure.append(t)
    tenure = np.array(tenure)
    
    # Introduce small number of 0 tenure (newly registered customers, standard in Telco dataset)
    zero_idx = np.random.choice(n_samples, size=11, replace=False)
    tenure[zero_idx] = 0
    
    # Phone services
    phone_service = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.903, 0.097])
    multiple_lines = []
    for ps in phone_service:
        if ps == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.48, 0.52]))
    multiple_lines = np.array(multiple_lines)
    
    # Internet services
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples, p=[0.344, 0.440, 0.216])
    
    # Online add-on services
    def generate_addon(internet_col, p_yes_dsl=0.35, p_yes_fiber=0.40):
        res = []
        for isrv in internet_col:
            if isrv == 'No':
                res.append('No internet service')
            elif isrv == 'DSL':
                res.append(np.random.choice(['Yes', 'No'], p=[p_yes_dsl, 1.0 - p_yes_dsl]))
            else: # Fiber optic
                res.append(np.random.choice(['Yes', 'No'], p=[p_yes_fiber, 1.0 - p_yes_fiber]))
        return np.array(res)
        
    online_security = generate_addon(internet_service, 0.36, 0.28)
    online_backup = generate_addon(internet_service, 0.38, 0.42)
    device_protection = generate_addon(internet_service, 0.38, 0.42)
    tech_support = generate_addon(internet_service, 0.37, 0.29)
    streaming_tv = generate_addon(internet_service, 0.33, 0.53)
    streaming_movies = generate_addon(internet_service, 0.33, 0.54)
    
    # Paperless billing and Payment method
    paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.592, 0.408])
    payment_methods = [
        'Electronic check', 
        'Mailed check', 
        'Bank transfer (automatic)', 
        'Credit card (automatic)'
    ]
    payment_method = np.random.choice(payment_methods, size=n_samples, p=[0.335, 0.228, 0.219, 0.218])
    
    # Monthly Charges calculation based on subscribed services
    monthly_charges = []
    for i in range(n_samples):
        base = 20.0 # base fee
        if phone_service[i] == 'Yes':
            base += 5.0
            if multiple_lines[i] == 'Yes':
                base += 5.0
        if internet_service[i] == 'DSL':
            base += 25.0
        elif internet_service[i] == 'Fiber optic':
            base += 45.0
            
        # Add-ons ($5 to $10 each)
        if online_security[i] == 'Yes': base += 5.0
        if online_backup[i] == 'Yes': base += 5.0
        if device_protection[i] == 'Yes': base += 5.0
        if tech_support[i] == 'Yes': base += 5.0
        if streaming_tv[i] == 'Yes': base += 10.0
        if streaming_movies[i] == 'Yes': base += 10.0
        
        # Add slight natural jitter
        base += np.random.normal(0, 2.5)
        base = max(round(base, 2), 18.25)
        monthly_charges.append(base)
    monthly_charges = np.array(monthly_charges)
    
    # Total Charges = tenure * monthly_charges (with small variance)
    total_charges = []
    for i in range(n_samples):
        if tenure[i] == 0:
            total_charges.append(" ") # blank string replicating real-world Telco missing string format
        else:
            tot = tenure[i] * monthly_charges[i] * np.random.uniform(0.98, 1.02)
            total_charges.append(f"{tot:.2f}")
    total_charges = np.array(total_charges)
    
    # Realistic Churn Calculation based on industry risk factors:
    # 1. Month-to-month contracts have high churn
    # 2. Fiber optic with no tech support / security has high churn
    # 3. New tenure has high churn
    # 4. Senior citizens and electronic check have elevated churn
    # 5. Long tenure, 2-year contracts, partner, tech support protect against churn
    churn_prob = []
    for i in range(n_samples):
        # Base log-odds
        z = -1.8
        
        # Contract effect
        if contract[i] == 'Month-to-month':
            z += 1.45
        elif contract[i] == 'Two year':
            z -= 1.30
        else: # One year
            z -= 0.40
            
        # Tenure effect (protective)
        z -= 0.045 * tenure[i]
        
        # Internet & tech support
        if internet_service[i] == 'Fiber optic':
            z += 0.65
            if tech_support[i] == 'No':
                z += 0.45
            if online_security[i] == 'No':
                z += 0.35
        elif internet_service[i] == 'No':
            z -= 0.70
            
        # Payment method
        if payment_method[i] == 'Electronic check':
            z += 0.40
        elif 'automatic' in payment_method[i]:
            z -= 0.25
            
        # Demographics
        if senior_citizen[i] == 1:
            z += 0.25
        if partner[i] == 'Yes':
            z -= 0.20
        if dependents[i] == 'Yes':
            z -= 0.25
            
        # Paperless billing
        if paperless_billing[i] == 'Yes':
            z += 0.20
            
        # Monthly charges (cost sensitivity)
        if monthly_charges[i] > 80:
            z += 0.35
        elif monthly_charges[i] < 35:
            z -= 0.30
            
        # Convert log-odds to probability
        prob = 1.0 / (1.0 + np.exp(-z))
        churn_prob.append(prob)
        
    churn = [np.random.choice(['Yes', 'No'], p=[p, 1.0 - p]) for p in churn_prob]
    
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })
    
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_telco_churn_data(7043, random_state=42)
    output_path = os.path.join('data', 'telco_customer_churn.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated dataset successfully: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True).round(4) * 100}%")
