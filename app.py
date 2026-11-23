"""
Streamlit Web Application: Customer Churn Prediction & Risk Analytics Dashboard.
Provides interactive churn scoring, risk factor breakdown, and EDA exploration.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .risk-card {
        padding: 22px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    
    /* High Risk */
    .risk-high {
        background: #2b1114 !important;
        border: 2px solid #ef5350 !important;
        border-left: 8px solid #d32f2f !important;
    }
    .risk-high .risk-title {
        color: #ff6b6b !important;
        font-weight: 700;
        font-size: 1.35rem;
        margin-top: 0;
        margin-bottom: 6px;
    }
    .risk-high .risk-score {
        color: #ff5252 !important;
        font-size: 3.2rem;
        font-weight: 800;
        margin: 4px 0 6px 0;
        line-height: 1;
    }
    .risk-high .risk-label {
        color: #f1f5f9 !important;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .risk-high .risk-desc {
        color: #cbd5e1 !important;
        font-size: 0.95rem;
        line-height: 1.4;
        margin-bottom: 0;
    }

    /* Medium Risk */
    .risk-medium {
        background: #2b220d !important;
        border: 2px solid #fbc02d !important;
        border-left: 8px solid #f57f17 !important;
    }
    .risk-medium .risk-title {
        color: #ffd54f !important;
        font-weight: 700;
        font-size: 1.35rem;
        margin-top: 0;
        margin-bottom: 6px;
    }
    .risk-medium .risk-score {
        color: #ffca28 !important;
        font-size: 3.2rem;
        font-weight: 800;
        margin: 4px 0 6px 0;
        line-height: 1;
    }
    .risk-medium .risk-label {
        color: #f1f5f9 !important;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .risk-medium .risk-desc {
        color: #cbd5e1 !important;
        font-size: 0.95rem;
        line-height: 1.4;
        margin-bottom: 0;
    }

    /* Low Risk */
    .risk-low {
        background: #0f2918 !important;
        border: 2px solid #4caf50 !important;
        border-left: 8px solid #2e7d32 !important;
    }
    .risk-low .risk-title {
        color: #69f0ae !important;
        font-weight: 700;
        font-size: 1.35rem;
        margin-top: 0;
        margin-bottom: 6px;
    }
    .risk-low .risk-score {
        color: #00e676 !important;
        font-size: 3.2rem;
        font-weight: 800;
        margin: 4px 0 6px 0;
        line-height: 1;
    }
    .risk-low .risk-label {
        color: #f1f5f9 !important;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .risk-low .risk-desc {
        color: #cbd5e1 !important;
        font-size: 0.95rem;
        line-height: 1.4;
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "models/best_churn_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_reports():
    eda_data, model_data, explain_data = {}, {}, {}
    if os.path.exists("reports/eda_summary.json"):
        with open("reports/eda_summary.json") as f:
            eda_data = json.load(f)
    if os.path.exists("reports/model_comparison.json"):
        with open("reports/model_comparison.json") as f:
            model_data = json.load(f)
    if os.path.exists("reports/feature_importance.json"):
        with open("reports/feature_importance.json") as f:
            explain_data = json.load(f)
    return eda_data, model_data, explain_data

model = load_model()
eda_data, model_data, explain_data = load_reports()

st.title("🎯 Customer Churn Prediction & Risk Analytics")
st.markdown("Predict customer defection, identify behavioral risk factors, and deploy proactive retention strategies.")

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
churn_rate = eda_data.get("dataset_metrics", {}).get("churn_rate_pct", 23.2)
best_auc = model_data.get("best_test_roc_auc", 0.85)
best_name = model_data.get("best_model", "Gradient_Boosting").replace("_", " ")
total_cust = eda_data.get("dataset_metrics", {}).get("total_records", 7043)

with col1:
    st.metric("Total Customers Analyzed", f"{total_cust:,}")
with col2:
    st.metric("Baseline Churn Rate", f"{churn_rate}%")
with col3:
    st.metric("Top Model ROC-AUC", f"{best_auc:.3f}")
with col4:
    st.metric("Champion Algorithm", best_name)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-Time Churn Predictor",
    "📈 Key Churn Drivers",
    "📊 Customer EDA Insights",
    "🤖 Model Performance Benchmark"
])

# ----------------- TAB 1: Real-Time Churn Predictor -----------------
with tab1:
    st.subheader("Interactive Customer Risk Scoring")
    st.caption("Adjust customer attributes to simulate churn probability and discover critical risk factors.")
    
    col_input, col_pred = st.columns([1.2, 1])
    
    with col_input:
        with st.expander("👤 Customer Demographics", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            with c2:
                partner = st.selectbox("Has Partner", ["Yes", "No"])
                dependents = st.selectbox("Has Dependents", ["Yes", "No"])
                
        with st.expander("💼 Account & Contract Attributes", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                tenure = st.slider("Tenure with Company (Months)", min_value=0, max_value=72, value=12)
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            with c2:
                payment_method = st.selectbox(
                    "Payment Method", 
                    [
                        "Electronic check", 
                        "Mailed check", 
                        "Bank transfer (automatic)", 
                        "Credit card (automatic)"
                    ]
                )
                monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=125.0, value=75.0, step=0.5)
                est_total = max(round(tenure * monthly_charges, 2), 0.0)
                st.caption(f"Estimated Total Charges: **${est_total:,.2f}**")
                
        with st.expander("🌐 Services Subscribed", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"] if phone == "Yes" else ["No phone service"])
                internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"] if internet != "No" else ["No internet service"])
            with c2:
                tech_supp = st.selectbox("Tech Support", ["No", "Yes", "No internet service"] if internet != "No" else ["No internet service"])
                online_bk = st.selectbox("Online Backup", ["No", "Yes", "No internet service"] if internet != "No" else ["No internet service"])
                device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"] if internet != "No" else ["No internet service"])
                stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"] if internet != "No" else ["No internet service"])
                stream_mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"] if internet != "No" else ["No internet service"])

    with col_pred:
        st.markdown("### Risk Evaluation Result")
        
        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple_lines,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_bk,
            'DeviceProtection': device_prot,
            'TechSupport': tech_supp,
            'StreamingTV': stream_tv,
            'StreamingMovies': stream_mov,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': est_total
        }])
        
        if model is not None:
            proba = model.predict_proba(input_data)[0, 1]
            churn_pct = round(proba * 100, 1)
            
            # Risk tiers
            if churn_pct >= 60.0:
                tier_class = "risk-high"
                badge = "🔴 HIGH CHURN RISK"
                strategy = "Immediate retention outreach required. Proactively offer a discounted annual contract switch or bundle complimentary Tech Support."
            elif churn_pct >= 35.0:
                tier_class = "risk-medium"
                badge = "🟡 MODERATE RISK"
                strategy = "Monitor engagement and promote loyalty rewards or automatic payment setup to reduce friction."
            else:
                tier_class = "risk-low"
                badge = "🟢 LOW RISK (RETAINED)"
                strategy = "Customer demonstrates high brand loyalty. Candidate for upsell or cross-sell opportunities."
                
            st.markdown(f"""
            <div class="risk-card {tier_class}">
                <div class="risk-title">{badge}</div>
                <div class="risk-score">{churn_pct}%</div>
                <div class="risk-label">Predicted Churn Probability</div>
                <div class="risk-desc"><strong>Action Playbook:</strong> {strategy}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Detected Risk Drivers:")
            detected_risks = []
            if contract == "Month-to-month":
                detected_risks.append("⚠️ **Month-to-Month Contract**: High vulnerability to competitive switching.")
            if tenure <= 6:
                detected_risks.append("⚠️ **Early Lifecycle**: Customer in the high-hazard 0-6 months onboarding window.")
            if internet == "Fiber optic" and tech_supp == "No":
                detected_risks.append("⚠️ **Fiber Optic without Tech Support**: Technical frustration risk with high bill.")
            if payment_method == "Electronic check":
                detected_risks.append("⚠️ **Electronic Check**: Lack of automated payment increases churn hazard.")
            if monthly_charges > 85:
                detected_risks.append(f"⚠️ **High Monthly Cost (${monthly_charges}/mo)**: Heightened price sensitivity.")
                
            if detected_risks:
                for r in detected_risks:
                    st.markdown(r)
            else:
                st.info("✅ No critical risk drivers detected.")
                
            st.markdown("#### Detected Retention Anchors:")
            detected_anchors = []
            if contract == "Two year":
                detected_anchors.append("🛡️ **2-Year Contract**: Exceptional long-term commitment stability.")
            elif contract == "One year":
                detected_anchors.append("🛡️ **1-Year Contract**: Favorable annual commitment.")
            if tenure >= 36:
                detected_anchors.append("🛡️ **High Tenure**: Proven loyalty (> 3 years).")
            if tech_supp == "Yes":
                detected_anchors.append("🛡️ **Tech Support Active**: Solves customer frustration points.")
            if online_sec == "Yes":
                detected_anchors.append("🛡️ **Online Security Active**: Increases stickiness and product utility.")
            if "automatic" in payment_method:
                detected_anchors.append("🛡️ **Auto-Pay Enrolled**: Seamless recurrent payment.")
                
            if detected_anchors:
                for a in detected_anchors:
                    st.markdown(a)
            else:
                st.caption("No strong retention anchors identified.")
        else:
            st.warning("Model not found. Please train models first.")

# ----------------- TAB 2: Key Drivers -----------------
with tab2:
    st.subheader("Key Drivers of Customer Churn & Retention")
    st.write("Identified via Logistic Regression Odds Ratios and Tree-based Permutation Importances.")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### 🚨 Top Risk Elevators (Increases Churn Odds)")
        risk_factors = explain_data.get("top_churn_risk_factors_odds_ratio", [])
        if risk_factors:
            rf_df = pd.DataFrame(risk_factors)[['feature', 'odds_ratio', 'impact']]
            st.dataframe(rf_df, width='stretch')
            
    with col_d2:
        st.markdown("#### 🛡️ Top Retention Anchors (Decreases Churn Odds)")
        ret_factors = explain_data.get("top_retention_factors_odds_ratio", [])
        if ret_factors:
            ret_df = pd.DataFrame(ret_factors)[['feature', 'odds_ratio', 'impact']]
            st.dataframe(ret_df, width='stretch')
            
    st.markdown("---")
    st.subheader("Strategic Retention Playbook")
    recs = explain_data.get("business_recommendations", [])
    for rec in recs:
        with st.expander(f"📌 {rec['category']}: {rec['driver']}", expanded=True):
            st.markdown(f"**Finding**: {rec['finding']}")
            st.markdown(f"**Recommendation**: {rec['recommendation']}")

# ----------------- TAB 3: EDA Insights -----------------
with tab3:
    st.subheader("Exploratory Data Analysis: Customer Behavior Patterns")
    seg_data = eda_data.get("segment_churn_rates", {})
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### Churn by Contract Type")
        if "Contract" in seg_data:
            c_df = pd.DataFrame(seg_data["Contract"]).T
            c_df = c_df.reset_index().rename(columns={"index": "Contract", "churn_rate_pct": "Churn Rate (%)"})
            st.bar_chart(data=c_df.set_index("Contract")["Churn Rate (%)"])
            st.dataframe(c_df, width='stretch')
            
    with col_e2:
        st.markdown("#### Churn by Internet Service")
        if "InternetService" in seg_data:
            i_df = pd.DataFrame(seg_data["InternetService"]).T
            i_df = i_df.reset_index().rename(columns={"index": "Internet Service", "churn_rate_pct": "Churn Rate (%)"})
            st.bar_chart(data=i_df.set_index("Internet Service")["Churn Rate (%)"])
            st.dataframe(i_df, width='stretch')
            
    col_e3, col_e4 = st.columns(2)
    with col_e3:
        st.markdown("#### Churn by Tenure Cohort")
        cohorts = eda_data.get("tenure_cohorts", {})
        if cohorts:
            t_df = pd.DataFrame(cohorts).T.reset_index().rename(columns={"index": "Tenure Cohort", "churn_rate_pct": "Churn Rate (%)"})
            st.bar_chart(data=t_df.set_index("Tenure Cohort")["Churn Rate (%)"])
            st.dataframe(t_df, width='stretch')
            
    with col_e4:
        st.markdown("#### Churn by Payment Method")
        if "PaymentMethod" in seg_data:
            p_df = pd.DataFrame(seg_data["PaymentMethod"]).T.reset_index().rename(columns={"index": "Payment Method", "churn_rate_pct": "Churn Rate (%)"})
            st.bar_chart(data=p_df.set_index("Payment Method")["Churn Rate (%)"])
            st.dataframe(p_df, width='stretch')

# ----------------- TAB 4: Model Benchmarks -----------------
with tab4:
    st.subheader("Model Performance Comparison")
    st.markdown("All models evaluated with 5-fold Stratified Cross-Validation and holdout test set.")
    
    all_models = model_data.get("all_models", {})
    if all_models:
        rows = []
        for m_name, metrics in all_models.items():
            rows.append({
                "Model": m_name.replace("_", " "),
                "Test ROC-AUC": metrics.get("test_roc_auc"),
                "CV ROC-AUC": f"{metrics.get('cv_roc_auc_mean')} ± {metrics.get('cv_roc_auc_std')}",
                "Accuracy": f"{metrics.get('test_accuracy')*100:.1f}%",
                "Precision": f"{metrics.get('test_precision')*100:.1f}%",
                "Recall": f"{metrics.get('test_recall')*100:.1f}%",
                "F1-Score": f"{metrics.get('test_f1'):.3f}"
            })
        m_df = pd.DataFrame(rows).sort_values(by="Test ROC-AUC", ascending=False)
        st.dataframe(m_df, width='stretch')
        
        st.markdown("#### Confusion Matrix (Best Model)")
        best_cm = all_models.get(model_data.get("best_model", ""), {}).get("confusion_matrix", {})
        if best_cm:
            cm_col1, cm_col2 = st.columns(2)
            with cm_col1:
                st.metric("True Negatives (Retained correctly)", f"{best_cm.get('true_negative', 0):,}")
                st.metric("False Negatives (Missed churners)", f"{best_cm.get('false_negative', 0):,}")
            with cm_col2:
                st.metric("True Positives (Identified churners)", f"{best_cm.get('true_positive', 0):,}")
                st.metric("False Positives (False alarms)", f"{best_cm.get('false_positive', 0):,}")

