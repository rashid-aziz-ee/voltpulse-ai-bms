# VoltPulse-AI BMS

**AI-Driven Predictive Battery Management & Thermal Safeguard System for EV & Grid Energy Storage**

VoltPulse-AI is an interactive dashboard platform that processes real-time telemetry to predict SoC (State of Charge), SoH (State of Health), Thermal Runaway Risk, and RUL (Remaining Useful Life), generating automated emergency load drop commands when critical conditions are met.

## Features
- **Thermal Runaway Detection**: XGBoost Classifier to detect and prevent cell overheating and fires.
- **RUL Prediction**: Random Forest Regressor to forecast remaining battery cycles based on degradation patterns.
- **Dynamic Load Shedding**: Automated simulation to isolate failing cells.

## Tech Stack
- Python, Streamlit, Pandas, Plotly
- XGBoost, Scikit-Learn

## Getting Started
1. Install requirements: `pip install -r requirements.txt`
2. Generate synthetic data: `python src/data_generator.py`
3. Train models: `python src/train_models.py`
4. Run dashboard: `streamlit run app.py`
