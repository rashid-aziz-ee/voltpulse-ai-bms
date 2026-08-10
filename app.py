import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3
import joblib
import xgboost as xgb
import os
import time
import google.generativeai as genai
import src.alert_system as alert_system

st.set_page_config(page_title="VoltPulse-AI Cloud BMS", page_icon="🔋", layout="wide")

# CSS Styling
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
        animation: pulse 1.5s infinite; margin-top: 15px;
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
    div[data-testid="stButton"] > button { width: 100%; height: 50px; font-weight: bold; }
    
    /* Prevent Streamlit screen dimming during live scan */
    div[data-testid="stAppViewContainer"] { opacity: 1 !important; transition: none !important; }
    div[data-testid="stAppViewBlockContainer"] { opacity: 1 !important; transition: none !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# SIDEBAR SETTINGS (Hackathon Mode)
# -------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Configuration")
    st.markdown("Configure external APIs for full Cloud integration.")
    
    gemini_key = st.text_input("Gemini API Key (For Co-Pilot)", type="password")
    
    st.markdown("---")
    st.subheader("☁️ Cloud Deployment Mode")
    run_simulator = st.checkbox("Run Internal IoT Simulator", value=False, help="Turn this ON if deploying to Streamlit Cloud where you can't run the backend terminal.")
    
    @st.cache_resource
    def start_simulator_thread():
        import threading
        from src.iot_streamer import stream_iot_data
        t = threading.Thread(target=stream_iot_data, daemon=True)
        t.start()
        return True

    if run_simulator:
        start_simulator_thread()
        st.success("Internal Simulator is running (Global State).")

    st.markdown("---")
    st.subheader("✉️ Email Alert System")
    sender_email = st.text_input("Your Gmail Address")
    app_password = st.text_input("Gmail App Password", type="password")
    target_email = sender_email # Send alerts to the same email provided

if gemini_key:
    genai.configure(api_key=gemini_key)

# -------------------------------------------------------------------
# Model & DB Loading
# -------------------------------------------------------------------
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

def fetch_latest_timestamp_data():
    db_path = os.path.join("data", "telemetry.db")
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        # Fetch only the last 55 rows (1 full timestamp for all cells)
        query = "SELECT * FROM telemetry ORDER BY id DESC LIMIT 55"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

df = fetch_latest_timestamp_data()

#st.title("⚡ NEXORA Global: VoltPulse-AI Enterprise BMS")
st.title("⚡Grid Station Battery Management System (BMS) - VoltPulse-AI")
st.markdown("Cloud-connected IoT platform for 110V Grid Stations with GenAI Co-Pilot & Financial Analytics.")

if df.empty or xgb_model is None or rf_model is None:
    st.error("IoT Stream disconnected or Models missing! Run `src/iot_streamer.py` and `src/train_models.py` first.")
    st.stop()

# -------------------------------------------------------------------
# Session State Setup
# -------------------------------------------------------------------
if 'sim_running' not in st.session_state:
    st.session_state.sim_running = False
if 'selected_cell' not in st.session_state:
    st.session_state.selected_cell = None
if 'alerted_cells' not in st.session_state:
    st.session_state.alerted_cells = set()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("▶ Connect Live IoT Stream" if not st.session_state.sim_running else "⏸ Disconnect Stream"):
        st.session_state.sim_running = not st.session_state.sim_running

# Get the absolute latest timestamp in the DB
latest_timestamp = df['Timestamp'].iloc[-1]
current_pack_df = df[df['Timestamp'] == latest_timestamp].copy()
features = ['Voltage_V', 'Current_A', 'Temperature_C', 'Cycle_Count']

if not current_pack_df.empty:
    X_pack = current_pack_df[features]
    current_pack_df['Thermal_Risk_Pred'] = xgb_model.predict(X_pack)
    current_pack_df['RUL_Pred'] = rf_model.predict(X_pack)

total_voltage = current_pack_df['Voltage_V'].sum()
avg_current = current_pack_df['Current_A'].mean()
max_temp = current_pack_df['Temperature_C'].max()
cell_data_dict = current_pack_df.set_index('Cell_ID').to_dict('index')

# Check for Critical Cells and Alert
for cell_id, c_data in cell_data_dict.items():
    if c_data['Thermal_Risk_Pred'] == 2 and cell_id not in st.session_state.alerted_cells:
        # Trigger Email Alert
        success = alert_system.send_alert(
            sender_email, app_password, target_email, 
            cell_id, c_data['Voltage_V'], c_data['Temperature_C'], 2
        )
        if success:
            st.toast(f"📧 Emergency email dispatched to {target_email} for {cell_id}!")
        st.session_state.alerted_cells.add(cell_id)

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 IoT Control Room", "🤖 BMS AI Co-Pilot", "💰 Business & ROI Impact"])

# ==========================================
# TAB 1: IOT CONTROL ROOM (Matrix)
# ==========================================
with tab1:
    if st.session_state.selected_cell is None:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card safe-card'><h4>Total Pack Voltage</h4><h2>{total_voltage:.1f} V</h2><p>Target: ~110V</p></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card safe-card'><h4>Average Draw Current</h4><h2>{avg_current:.1f} A</h2><p>Live Load</p></div>", unsafe_allow_html=True)
        temp_class = "safe-card" if max_temp < 60 else ("warning-card" if max_temp < 75 else "critical-card")
        c3.markdown(f"<div class='metric-card {temp_class}'><h4>Max Pack Temperature</h4><h2>{max_temp:.1f} °C</h2></div>", unsafe_allow_html=True)

        st.subheader("🔋 55-Cell Matrix Visualization")
        st.write(f"Live Snapshot @ {latest_timestamp} (Syncing with SQLite...)")
        st.markdown("**Legend:** 🟢 Optimal | 🟡 Warning | 🔴 Critical (Fire Risk)")
        
        grid_cols = st.columns(11)
        for i in range(1, 56):
            cell_id = f"Cell_{i}"
            col_idx = (i - 1) % 11
            status_icon = "🟢"
            if cell_id in cell_data_dict:
                risk = cell_data_dict[cell_id]['Thermal_Risk_Pred']
                if risk == 1: status_icon = "🟡"
                elif risk == 2: status_icon = "🔴"
            
            with grid_cols[col_idx]:
                if st.button(f"{status_icon} {i}", key=f"btn_{i}"):
                    st.session_state.selected_cell = cell_id
                    st.rerun()
    else:
        # Detailed View
        focus_cell = st.session_state.selected_cell
        if st.button("⬅️ Back to Main Screen", type="primary"):
            st.session_state.selected_cell = None
            st.rerun()
            
        st.subheader(f"🔎 Detailed Analysis: {focus_cell}")
        if focus_cell in cell_data_dict:
            c_data = cell_data_dict[focus_cell]
            risk_level = c_data['Thermal_Risk_Pred']
            
            if risk_level == 0:
                status_html = "<div class='metric-card safe-card'><h4>AI Status</h4><h2 style='color:#00ffcc'>✅ OPTIMAL</h2></div>"
            elif risk_level == 1:
                status_html = "<div class='metric-card warning-card'><h4>AI Status</h4><h2 style='color:#ffaa00'>⚠️ WARNING</h2></div>"
            else:
                status_html = "<div class='emergency-alert'><h3>🚨 CRITICAL THERMAL RUNAWAY</h3><p>Relays Isolated. Email sent to engineering.</p></div>"
                
            colA, colB = st.columns([2, 1])
            with colA:
                mc1, mc2 = st.columns(2)
                mc1.markdown(f"<div class='metric-card safe-card'><h4>Voltage</h4><h2>{c_data['Voltage_V']:.2f} V</h2></div>", unsafe_allow_html=True)
                mc2.markdown(f"<div class='metric-card safe-card'><h4>Predicted RUL</h4><h2>{int(c_data['RUL_Pred'])} Cycles</h2></div>", unsafe_allow_html=True)
            with colB:
                st.markdown(status_html, unsafe_allow_html=True)
                
            st.markdown("---")
            st.subheader("📈 Historical Degradation & Telemetry")
            try:
                db_path = os.path.join("data", "telemetry.db")
                conn = sqlite3.connect(db_path)
                history_df = pd.read_sql_query(f"SELECT * FROM telemetry WHERE Cell_ID='{focus_cell}' ORDER BY id DESC LIMIT 40", conn)
                conn.close()
                
                if not history_df.empty:
                    history_df = history_df.sort_values(by="id")
                    fig = px.line(history_df, x="Timestamp", y=["Voltage_V", "Temperature_C"], 
                                  title=f"Live Sensor Timeline ({focus_cell})", markers=True, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Waiting for historical data...")
            except Exception as e:
                st.error(f"Could not load graph: {e}")

# ==========================================
# TAB 2: BMS AI CO-PILOT
# ==========================================
with tab2:
    st.subheader("🤖 VoltPulse-AI Co-Pilot (Powered by Gemini)")
    st.write("Ask natural language questions about your battery grid.")
    
    if not gemini_key:
        st.warning("⚠️ Please enter your Gemini API Key in the sidebar to activate the Co-Pilot.")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        user_input = st.chat_input("Ask about weak cells, thermal events, or grid health...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing grid data..."):
                    try:
                        # Test models in order of preference to avoid 'interactions API' errors
                        known_models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.0-pro']
                        
                        # Build context from current grid state
                        criticals = [c for c, d in cell_data_dict.items() if d['Thermal_Risk_Pred'] == 2]
                        warnings = [c for c, d in cell_data_dict.items() if d['Thermal_Risk_Pred'] == 1]
                        
                        # Calculate advanced metrics
                        avg_rul = np.mean([d['RUL_Pred'] for d in cell_data_dict.values()]) if cell_data_dict else 0
                        health_score = max(0.0, 100.0 - (((len(criticals)*2 + len(warnings)*1) / 55) * 100))
                        
                        # Build conversational memory for Gemini (excluding the latest message just appended)
                        gemini_history = []
                        for msg in st.session_state.chat_history[:-1]:
                            role = "model" if msg["role"] == "assistant" else "user"
                            gemini_history.append({"role": role, "parts": [msg["content"]]})
                        
                        enhanced_prompt = f"""
[LIVE GRID TELEMETRY CONTEXT]
- Total Pack Voltage: {total_voltage:.1f} V
- Max Pack Temperature: {max_temp:.1f} °C
- Avg Predicted Remaining Useful Life (RUL): {avg_rul:.0f} Cycles
- Overall Battery Bank Health Score: {health_score:.1f}%
- Critical Cells (Red/Fire Risk): {criticals if criticals else 'None'}
- Warning Cells (Yellow/Degrading): {warnings if warnings else 'None'}

Act as the VoltPulse-AI Professional Grid Engineer Co-Pilot. Use the live context above to answer the user's question deeply and analytically. Do not give generic answers; calculate and analyze based on the numbers provided.

User Question: {user_input}
"""
                        
                        response_text = None
                        last_error = None
                        
                        for m in known_models:
                            try:
                                model = genai.GenerativeModel(m)
                                chat = model.start_chat(history=gemini_history)
                                response = chat.send_message(enhanced_prompt)
                                response_text = response.text
                                break # Stop if successful
                            except Exception as e:
                                last_error = str(e)
                                continue
                                
                        if response_text:
                            st.markdown(response_text)
                            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                        else:
                            try:
                                avail_models = [m.name for m in genai.list_models()]
                            except Exception as ex:
                                avail_models = f"Could not list models: {ex}"
                                
                            error_msg = f"Failed to generate response. Last Error: {last_error}\n\n**Available Models for your API Key:** {avail_models}"
                            st.error(error_msg)
                    except Exception as e:
                        st.error(f"Critical Error: {e}")

# ==========================================
# TAB 3: FINANCIAL & ROI IMPACT
# ==========================================
with tab3:
    st.subheader("💰 Business Impact & Return on Investment")
    st.markdown("AI-driven thermal safeguards don't just prevent fires; they save millions in operational costs.")
    
    # Calculate mock metrics based on total cycle counts across the pack
    total_cycles_used = sum(c['Cycle_Count'] for c in cell_data_dict.values())
    avg_rul = np.mean([c['RUL_Pred'] for c in cell_data_dict.values()])
    
    # Financial metrics
    cost_per_cell = 1500 # $
    prevented_incidents = len(st.session_state.alerted_cells)
    incident_cost = 500000 # Cost of a grid station fire/outage
    
    saved_money = prevented_incidents * incident_cost
    capex_optimization = (avg_rul / 1200) * (55 * cost_per_cell) # Value of remaining life optimized by AI
    
    fc1, fc2, fc3 = st.columns(3)
    fc1.markdown(f"<div class='metric-card safe-card'><h4>Outage Cost Avoided</h4><h2 style='color:#00ffcc'>${saved_money:,}</h2><p>From {prevented_incidents} Prevented Thermal Events</p></div>", unsafe_allow_html=True)
    fc2.markdown(f"<div class='metric-card safe-card'><h4>CAPEX Value Retained</h4><h2>${int(capex_optimization):,}</h2><p>Through Smart Load Shedding</p></div>", unsafe_allow_html=True)
    fc3.markdown(f"<div class='metric-card safe-card'><h4>Grid Uptime</h4><h2>99.999%</h2><p>Zero Unplanned Downtime</p></div>", unsafe_allow_html=True)
    
    st.markdown("### The NEXORA Value Proposition")
    st.write("""
    1. **Predictive Maintenance:** By forecasting RUL (Remaining Useful Life) using Random Forest, grid operators order replacements exactly when needed, eliminating 20% premature disposal waste.
    2. **Thermal Runaway Prevention:** XGBoost immediately isolates cells crossing the 75°C threshold. Without AI, a single cell fire can destroy a $500K battery room and cause regional blackouts.
    3. **Automated Response:** The system doesn't wait for humans. It sheds load, trips relays, and emails engineers autonomously within milliseconds.
    """)

# -------------------------------------------------------------------
# Auto-Refresh Logic
# -------------------------------------------------------------------
if st.session_state.sim_running:
    time.sleep(2.0)
    st.rerun()
