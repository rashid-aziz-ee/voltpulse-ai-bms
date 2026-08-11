# NeuroBank AI 🧠⚡
**Predictive Battery Bank Management System for Grid Stations**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![Gemini AI](https://img.shields.io/badge/Google-Gemini%202.5-orange.svg)](https://deepmind.google/technologies/gemini/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-green.svg)](https://scikit-learn.org/)

NeuroBank AI is an enterprise-grade, cloud-connected IoT Edge platform designed to predict, manage, and secure large-scale battery banks in 132kV Grid Stations. It combines real-time IoT telemetry, machine learning algorithms, and a Generative AI Co-Pilot to completely eliminate unplanned downtime, prevent catastrophic thermal runaways, and maximize operational ROI.

## 🚀 Key Features

*   **🔋 55-Cell Matrix Live Telemetry:** Real-time SQLite-backed edge computing simulation monitoring voltage and temperature across a 55-cell industrial battery grid.
*   **🔥 Thermal Runaway Prevention (XGBoost):** An advanced ML classifier instantly identifies cells crossing critical temperature thresholds (75°C+) and simulates automated load-shedding and relay isolation.
*   **⏳ Remaining Useful Life Forecasting (Random Forest):** Accurately predicts the remaining lifecycle (RUL) of individual cells, optimizing CAPEX by preventing premature battery replacements.
*   **🤖 AI Grid Engineer Co-Pilot (Gemini 2.5):** A conversational AI assistant with live grid context and chat memory that can perform deep analytics, calculate health scores, and answer complex engineering questions.
*   **🚨 Automated Emergency Alert System:** A robust SMTP-based notification engine that autonomously dispatches alerts to on-duty engineers the millisecond a cell reaches critical status.
*   **💰 Financial ROI Dashboard:** Translates technical data into executive business value, calculating prevented outage costs ($500K/incident) and retained CAPEX through smart load shedding.

## 🛠️ Technology Stack

*   **Frontend & Dashboard:** Streamlit, Plotly (for interactive historical degradation charts), Custom CSS (Dark Mode/Neon UI).
*   **Machine Learning:** Scikit-Learn, XGBoost, Random Forest Regressor/Classifier.
*   **Generative AI:** Google Generative AI SDK (`gemini-2.5-flash`).
*   **Backend & Data Engineering:** Python, SQLite, Pandas, NumPy.
*   **IoT Simulation Engine:** Multi-threaded asynchronous background data generation.

## ⚙️ Getting Started

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/rashid-aziz-ee/voltpulse-ai-bms.git
cd voltpulse-ai-bms
pip install -r requirements.txt
```

### 2. Generate Synthetic Data & Train ML Models
```bash
python src/data_generator.py
python src/train_models.py
```

### 3. Run the IoT Streamer (Terminal 1)
This script simulates the live IoT edge devices updating the SQLite database.
```bash
python src/iot_streamer.py
```

### 4. Launch the Dashboard (Terminal 2)
```bash
streamlit run app.py
```
*(Ensure you enter your Google Gemini API Key in the sidebar to activate the AI Co-Pilot).*

**Built for Impact. Driven by AI.**
