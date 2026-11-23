"""
Model Training and Evaluation Module.
Trains and compares Logistic Regression, Random Forest, Gradient Boosting, and Decision Tree.
Evaluates using 5-fold Stratified Cross Validation and Holdout Test metrics.
Saves the best end-to-end Pipeline to models/best_churn_model.joblib.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

from src.preprocess import prepare_data, get_preprocessor, get_feature_names

def train_and_evaluate_models(csv_path="data/telco_customer_churn.csv", output_dir="models", reports_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("\n--- 1. Preparing Data ---")
    X_train, X_test, y_train, y_test, raw_df = prepare_data(csv_path, test_size=0.20, random_state=42)
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")
    print(f"Class distribution: Non-Churn={(y_train == 0).sum()} ({100*(y_train == 0).mean():.1f}%), Churn={(y_train == 1).sum()} ({100*(y_train == 1).mean():.1f}%)")
    
    # Define candidate classifiers
    candidate_classifiers = {
        "Logistic_Regression": LogisticRegression(
            class_weight='balanced', 
            max_iter=1000, 
            random_state=42, 
            solver='lbfgs'
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150, 
            max_depth=8, 
            min_samples_split=10, 
            min_samples_leaf=4, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=150, 
            learning_rate=0.08, 
            max_depth=4, 
            min_samples_split=10, 
            min_samples_leaf=4, 
            random_state=42
        ),
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=5, 
            class_weight='balanced', 
            random_state=42
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_model_name = None
    best_roc_auc = -1.0
    best_pipeline = None
    
    print("\n--- 2. Training and Cross-Validating Models ---")
    for name, clf in candidate_classifiers.items():
        preprocessor = get_preprocessor()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # 5-fold CV on ROC-AUC and F1
        cv_roc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        
        # Fit on full train set
        pipeline.fit(X_train, y_train)
        
        # Evaluate on holdout test set
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = pipeline.decision_function(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            "cv_roc_auc_mean": round(float(cv_roc_scores.mean()), 4),
            "cv_roc_auc_std": round(float(cv_roc_scores.std()), 4),
            "cv_f1_mean": round(float(cv_f1_scores.mean()), 4),
            "cv_f1_std": round(float(cv_f1_scores.std()), 4),
            "test_accuracy": round(float(acc), 4),
            "test_balanced_accuracy": round(float(bal_acc), 4),
            "test_precision": round(float(prec), 4),
            "test_recall": round(float(rec), 4),
            "test_f1": round(float(f1), 4),
            "test_roc_auc": round(float(roc_auc), 4),
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1]
            }
        }
        
        # Save individual model pipeline
        model_path = os.path.join(output_dir, f"{name.lower()}_pipeline.joblib")
        joblib.dump(pipeline, model_path)
        
        print(f"[{name:<20}] CV ROC-AUC: {cv_roc_scores.mean():.4f} (+/- {cv_roc_scores.std():.4f}) | Test ROC-AUC: {roc_auc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        # Prioritize ROC-AUC for ranking ranking capabilities
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n>>> Best Model: {best_model_name} (Test ROC-AUC: {best_roc_auc:.4f})")
    
    # Save best model to models/best_churn_model.joblib
    best_model_path = os.path.join(output_dir, "best_churn_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"Saved best model pipeline to {best_model_path}")
    
    # Also save metadata
    metadata = {
        "best_model": best_model_name,
        "best_test_roc_auc": round(float(best_roc_auc), 4),
        "all_models": results
    }
    
    reports_path = os.path.join(reports_dir, "model_comparison.json")
    with open(reports_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model comparison reports to {reports_path}")
    
    return metadata, best_pipeline

if __name__ == '__main__':
    train_and_evaluate_models()

