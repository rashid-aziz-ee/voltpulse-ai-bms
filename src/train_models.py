import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
import json

def train_models():
    print("Loading telemetry data...")
    data_path = os.path.join("data", "battery_telemetry.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Please run data_generator.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Feature Selection
    features = ['Voltage_V', 'Current_A', 'Temperature_C', 'Cycle_Count']
    X = df[features]
    
    # Target 1: Thermal Risk (Classification)
    y_risk = df['Thermal_Risk']
    
    # Target 2: Remaining Useful Life (Regression)
    y_rul = df['RUL_Cycles']
    
    # Train-test split
    X_train, X_test, yr_train, yr_test, yl_train, yl_test = train_test_split(
        X, y_risk, y_rul, test_size=0.2, random_state=42
    )
    
    os.makedirs("models", exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Train XGBoost Classifier for Thermal Risk
    # ---------------------------------------------------------
    print("\nTraining XGBoost Classifier for Thermal Risk...")
    xgb_model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, yr_train)
    
    # Evaluate
    yr_pred = xgb_model.predict(X_test)
    print("Thermal Risk Accuracy:", accuracy_score(yr_test, yr_pred))
    print(classification_report(yr_test, yr_pred, zero_division=0))
    
    # Save XGBoost model
    xgb_model_path = os.path.join("models", "xgboost_thermal_model.json")
    xgb_model.save_model(xgb_model_path)
    print(f"✅ Saved XGBoost Thermal Model to {xgb_model_path}")
    
    # ---------------------------------------------------------
    # 2. Train Random Forest Regressor for RUL
    # ---------------------------------------------------------
    print("\nTraining Random Forest Regressor for Remaining Useful Life (RUL)...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    rf_model.fit(X_train, yl_train)
    
    # Evaluate
    yl_pred = rf_model.predict(X_test)
    print("RUL Mean Absolute Error:", mean_absolute_error(yl_test, yl_pred))
    print("RUL R2 Score:", r2_score(yl_test, yl_pred))
    
    # Save Random Forest model
    rf_model_path = os.path.join("models", "rul_rf_model.joblib")
    joblib.dump(rf_model, rf_model_path)
    print(f"✅ Saved Random Forest RUL Model to {rf_model_path}")

if __name__ == "__main__":
    train_models()
