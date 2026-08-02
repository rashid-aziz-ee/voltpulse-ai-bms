import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import xgboost as xgb
import os
import time

st.set_page_config(page_title="VoltPulse-AI 132kV BMS", page_icon="🔋", layout="wide")

# CSS Styling for Dashboard
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e2f; padding: 20px; border-radius: 10px; margin: 10px 0;
        border-left: 5px solid #00ffcc; color: white;
    }
    .critical-card { border-left: 5px solid #ff4444; }
    .warning-card { border-left: 5px solid #ffaa00; }
    .safe-card { border-left: 5px solid #00ffcc; }
    
    .emergency-alert {
        background-color: rgba(255, 68, 68, 0.2);
        color: #ff4444; padding: 20px; border-radius: 10px;
        border: 2px solid #ff4444; font-weight: bold;
        animation: pulse 1.5s infinite;
        margin-top: 15px;
    }
    @keyframes pulse {
        0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; }
    }
    
    /* Make buttons full width in columns */
    div[data-testid="stButton"] > button {
        width: 100%;
        height: 50px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(os.path.join("models", "xgboost_thermal_model.json"))
        rf_model = joblib.load(os.path.join("models", "rul_rf_model.joblib"))
        return xgb_model, rf_model
    except Exception as e:
        return None, None

xgb_model, rf_model = load_models()

@st.cache_data
def load_data():
    path = os.path.join("data", "battery_telemetry.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

df = load_data()

st.title("⚡ 132kV Grid Station BMS - 55 Cell Matrix")
st.markdown("Enterprise-grade AI platform for 110V Battery Bank Health & Thermal Safeguards.")

if df.empty or xgb_model is None or rf_model is None:
    st.error("Data or Models missing. Please run `data_generator.py` and `train_models.py` first.")
    st.stop()

# -------------------------------------------------------------------
# Session State Setup
# -------------------------------------------------------------------
if 'sim_time' not in st.session_state:
    st.session_state.sim_time = df['Timestamp'].unique()[0]
if 'sim_running' not in st.session_state:
    st.session_state.sim_running = False
if 'selected_cell' not in st.session_state:
    st.session_state.selected_cell = None

# -------------------------------------------------------------------
# Top Control Bar
# -------------------------------------------------------------------
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("▶ Start Live Scan" if not st.session_state.sim_running else "⏸ Pause Scan"):
        st.session_state.sim_running = not st.session_state.sim_running

timestamps = df['Timestamp'].unique()
current_ts_idx = np.where(timestamps == st.session_state.sim_time)[0][0]

if st.session_state.sim_running:
    current_ts_idx = (current_ts_idx + 1) % len(timestamps)
    st.session_state.sim_time = timestamps[current_ts_idx]

# -------------------------------------------------------------------
# Data Inference
# -------------------------------------------------------------------
current_pack_df = df[df['Timestamp'] == st.session_state.sim_time].copy()
features = ['Voltage_V', 'Current_A', 'Temperature_C', 'Cycle_Count']

if not current_pack_df.empty:
    X_pack = current_pack_df[features]
    current_pack_df['Thermal_Risk_Pred'] = xgb_model.predict(X_pack)
    current_pack_df['RUL_Pred'] = rf_model.predict(X_pack)

total_voltage = current_pack_df['Voltage_V'].sum()
avg_current = current_pack_df['Current_A'].mean()
max_temp = current_pack_df['Temperature_C'].max()

# Helper dict for easy access
cell_data_dict = current_pack_df.set_index('Cell_ID').to_dict('index')

st.markdown("---")

# -------------------------------------------------------------------
# VIEW ROUTING
# -------------------------------------------------------------------
if st.session_state.selected_cell is None:
    # ==========================================
    # MAIN OVERVIEW SCREEN
    # ==========================================
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card safe-card'><h4>Total Pack Voltage</h4><h2>{total_voltage:.1f} V</h2><p>Target: ~110V (55 Cells x 2V)</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card safe-card'><h4>Average Draw Current</h4><h2>{avg_current:.1f} A</h2><p>Live Load</p></div>", unsafe_allow_html=True)
    temp_class = "safe-card" if max_temp < 60 else ("warning-card" if max_temp < 75 else "critical-card")
    c3.markdown(f"<div class='metric-card {temp_class}'><h4>Max Pack Temperature</h4><h2>{max_temp:.1f} °C</h2><p>Ambient/Cell Max</p></div>", unsafe_allow_html=True)

    st.subheader("🔋 55-Cell Matrix Visualization")
    st.write(f"Live Snapshot @ {st.session_state.sim_time} (Click any cell for details)")
    
    st.markdown("**Legend:** 🟢 Optimal (2.0V-2.3V) | 🟡 Warning (<1.9V or >60°C) | 🔴 Critical (Fire Risk)")
    
    # 55 Cell Grid: 11 columns x 5 rows
    grid_cols = st.columns(11)
    
    for i in range(1, 56):
        cell_id = f"Cell_{i}"
        col_idx = (i - 1) % 11
        
        # Determine cell status
        status_icon = "🟢"
        if cell_id in cell_data_dict:
            risk = cell_data_dict[cell_id]['Thermal_Risk_Pred']
            if risk == 1:
                status_icon = "🟡"
            elif risk == 2:
                status_icon = "🔴"
        
        with grid_cols[col_idx]:
            # If button clicked, set state and rerun
            if st.button(f"{status_icon} {i}", key=f"btn_{i}"):
                st.session_state.selected_cell = cell_id
                st.rerun()

else:
    # ==========================================
    # DETAILED CELL FOCUS SCREEN
    # ==========================================
    focus_cell = st.session_state.selected_cell
    
    if st.button("⬅️ Back to Main Screen", type="primary"):
        st.session_state.selected_cell = None
        st.rerun()
        
    st.subheader(f"🔎 Detailed Analysis: {focus_cell}")
    st.write(f"Live Data @ {st.session_state.sim_time}")
    
    if focus_cell in cell_data_dict:
        c_data = cell_data_dict[focus_cell]
        risk_level = c_data['Thermal_Risk_Pred']
        
        if risk_level == 0:
            status_html = "<div class='metric-card safe-card'><h4>AI Status</h4><h2 style='color:#00ffcc'>✅ OPTIMAL</h2><p>No thermal anomalies detected.</p></div>"
        elif risk_level == 1:
            status_html = "<div class='metric-card warning-card'><h4>AI Status</h4><h2 style='color:#ffaa00'>⚠️ WARNING</h2><p>Cell degrading. Passive cooling initiated.</p></div>"
        else:
            status_html = f"""
            <div class='emergency-alert'>
                <h3>🚨 CRITICAL THERMAL RUNAWAY</h3>
                <p>Cell failure imminent. Isolating relay and shedding load immediately.</p>
            </div>
            """
            
        colA, colB = st.columns([2, 1])
        
        with colA:
            mc1, mc2 = st.columns(2)
            mc1.markdown(f"<div class='metric-card safe-card'><h4>Voltage</h4><h2>{c_data['Voltage_V']:.2f} V</h2><p>Target: 2.0V</p></div>", unsafe_allow_html=True)
            mc2.markdown(f"<div class='metric-card safe-card'><h4>Current</h4><h2>{c_data['Current_A']:.1f} A</h2></div>", unsafe_allow_html=True)
            
            mc3, mc4 = st.columns(2)
            temp_card = "safe-card" if c_data['Temperature_C'] < 60 else ("warning-card" if c_data['Temperature_C'] < 75 else "critical-card")
            mc3.markdown(f"<div class='metric-card {temp_card}'><h4>Temperature</h4><h2>{c_data['Temperature_C']:.1f} °C</h2></div>", unsafe_allow_html=True)
            mc4.markdown(f"<div class='metric-card safe-card'><h4>Predicted RUL</h4><h2>{int(c_data['RUL_Pred'])} Cycles</h2><p>Current Age: {c_data['Cycle_Count']}</p></div>", unsafe_allow_html=True)
            
        with colB:
            st.markdown(status_html, unsafe_allow_html=True)
            
        st.markdown("---")
        st.subheader("Historical Degradation")
        
        # Historical Chart for this specific cell
        history_df = df[df['Cell_ID'] == focus_cell].copy()
        history_df = history_df[history_df['Timestamp'] <= st.session_state.sim_time]
        history_df = history_df.tail(40) # Last 40 readings
        
        fig = px.line(history_df, x="Timestamp", y=["Voltage_V", "Temperature_C"], 
                      title=f"Telemetry Timeline ({focus_cell})", markers=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Data not available for {focus_cell} at this timestamp.")

# -------------------------------------------------------------------
# Auto-Refresh Loop Execution
# -------------------------------------------------------------------
if st.session_state.sim_running:
    time.sleep(2.0)
    st.rerun()
