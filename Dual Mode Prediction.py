# ============================================================================
# LETTUCE GROWTH DIGITAL TWIN & ML PREDICTOR
# Description: A Streamlit application for predicting and simulating lettuce 
#              growth based on environmental sensor data.
#              Uses Dual Machine Learning Models (Gradient Boosting):
#              1. Tracker Model: Predicts absolute length based on time + sensors.
#              2. Optimizer Model: Predicts growth rate (cm/day) to provide
#                 real-time, stage-specific optimization suggestions.
# ============================================================================

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================================
# THINGSPEAK CONFIGURATION
# ============================================================================
CHANNEL_ID = "3326913"          # Unique identifier for the ThingSpeak channel
READ_API_KEY = "1YBWG6QWWPA9TNLT"  # API key to read data from the channel

# ============================================================================
# DOMAIN CONSTANTS - VALID RANGES AND OPTIMAL ZONES
# ============================================================================
RANGES = {
    "Temp": (18.0, 33.5),      # Temperature range in Celsius
    "Hum": (50.0, 80.0),        # Humidity range in percentage
    "TDS": (400.0, 800.0),      # Total Dissolved Solids (nutrient) range in ppm
    "pH": (6.0, 6.8)            # pH level range
}

OPTIMA = {
    "Temp": (27.0, 30.0),       # Optimal temperature range for lettuce
    "Hum": (62.0, 72.0),        # Optimal humidity range
    "TDS": (560.0, 680.0),      # Optimal nutrient concentration range
    "pH": (6.2, 6.5)            # Optimal pH for nutrient uptake
}

UNITS = {
    "Temp": "°C",               
    "Hum": "%",                 
    "TDS": " ppm",              
    "pH": ""                    
}

KEYS = ["Temp", "Hum", "TDS", "pH"]  # List of all sensor types

# ============================================================================
# COLOR PALETTE DEFINITIONS
# ============================================================================
CLR_GOOD = "#2e7d32"            # Dark green: indicates healthy/optimal condition
CLR_WARN = "#e65100"            # Orange: indicates suboptimal condition
CLR_BAD = "#b71c1c"             # Dark red: indicates off-range/dangerous condition

# Growth phase colors - used to visually distinguish different stages of plant growth
CLR_PHASE = ["#9FE1CB", "#5DCAA5", "#1D9E75"]  

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clamp(v, mn, mx):
    """Restrict a value to a specified range."""
    return max(mn, min(mx, v))

def health_status(key, val):
    """Determine the health status ('good', 'warn', 'bad') of a sensor reading."""
    lo, hi = OPTIMA[key]
    
    if lo <= val <= hi:
        return "good"
    
    span = RANGES[key][1] - RANGES[key][0]
    dist = (lo - val) / span if val < lo else (val - hi) / span
    
    return "bad" if dist > 0.15 else "warn"

def get_rate_suggestions(model_rate, temp, hum, tds, ph, current_day):
    """
    Generate AI-powered optimization suggestions for sensor adjustments.
    This locks the 'current_day' so the AI only suggests changing the environment,
    not time itself, maximizing the Growth Rate (cm/day) for that specific life stage.
    """
    vals = [temp, hum, tds, ph, current_day]
    base_rate = model_rate.predict([vals])[0]
    tips = []
    
    # Loop ONLY through the 4 sensors, explicitly ignoring current_day
    for i, k in enumerate(KEYS):
        rMin, rMax = RANGES[k]
        
        # Calculate hypothetical values: 12% increase and 12% decrease
        up_v = clamp(vals[i] * 1.12, rMin, rMax)
        dn_v = clamp(vals[i] * 0.88, rMin, rMax)
        
        up_in, dn_in = vals[:], vals[:]
        up_in[i], dn_in[i] = up_v, dn_v
        
        # Get predictions for increased and decreased values
        pred_up = model_rate.predict([up_in])[0]
        pred_dn = model_rate.predict([dn_in])[0]
        
        h = health_status(k, vals[i])
        
        # If increasing or decreasing speeds up growth by >0.05 cm/day, suggest it!
        if pred_up - base_rate > 0.05 and pred_up >= pred_dn:
            tips.append({"key": k, "dir": "up", "boost": pred_up - base_rate, "new_val": up_v, "health": h})
        elif pred_dn - base_rate > 0.05:
            tips.append({"key": k, "dir": "down", "boost": pred_dn - base_rate, "new_val": dn_v, "health": h})
        else:
            tips.append({"key": k, "dir": "ok", "boost": 0, "new_val": vals[i], "health": h})
            
    tips.sort(key=lambda x: -x["boost"])
    return tips, base_rate

# ============================================================================
# DATA LOADING FROM THINGSPEAK
# ============================================================================

@st.cache_data(show_spinner="Fetching data from ThingSpeak...")
def load_thingspeak():
    """Fetch sensor data from ThingSpeak IoT platform and prepare it for analysis."""
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
    
    resp = requests.get(url, params={"api_key": READ_API_KEY, "results": 8000})
    feeds = resp.json().get("feeds", [])
    if not feeds:
        return None
    
    df = pd.DataFrame(feeds)
    
    # Rename ThingSpeak field columns to human-readable names
    df = df.rename(columns={
        "field1": "Temp", "field2": "Hum", "field3": "TDS",
        "field4": "pH", "field5": "Growth_Days", "field6": "Growth_Length"
    })
    
    # Convert string values to numeric types
    for col in ["Temp", "Hum", "TDS", "pH", "Growth_Days", "Growth_Length"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Remove incomplete records and prevent division by zero
    df = df.dropna(subset=["Temp", "Hum", "TDS", "pH", "Growth_Length", "Growth_Days"])
    df = df[df["Growth_Days"] > 0]
    
    # ENGINEER NEW TARGET: Growth Rate (cm/day)
    df["Growth_Rate"] = df["Growth_Length"] / df["Growth_Days"] 
    
    return df.reset_index(drop=True)

# ============================================================================
# STREAMLIT APP CONFIGURATION AND INITIALIZATION
# ============================================================================

st.set_page_config(page_title="Lettuce Digital Twin Simulator", layout="wide")
st.title("🌱 Interactive Lettuce Digital Twin")

df = load_thingspeak()

if df is None or df.empty:
    st.error("No data found in ThingSpeak channel. Please check connection.")
    st.stop()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Initialize model storage
if "model_rate" not in st.session_state:
    st.session_state.model_rate = None
    st.session_state.model_len = None

# Initialize Simulation (Digital Twin) Memory
if "sim_day" not in st.session_state:
    st.session_state.sim_day = 1
    st.session_state.sim_length = 1.0 # Starting seed length
    st.session_state.sim_history = [{"day": 1, "length": 1.0, "temp": 28.0, "tds": 550.0, "hum": 65.0, "ph": 6.4}]

# ============================================================================
# MAIN LAYOUT - TOP BAR WITH THREE COLUMNS
# ============================================================================

col_model, col_sensor, col_result = st.columns(3)

# ============================================================================
# COLUMN 1: AI MODEL TRAINING
# ============================================================================
with col_model:
    st.subheader("1. AI Models")
    
    if st.button("Train Twin Models", use_container_width=True):
        # 1. Train Tracker Model (Absolute Length)
        X_len = df[["Temp", "Hum", "TDS", "pH", "Growth_Days"]]
        y_len = df["Growth_Length"]
        m_len = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        m_len.fit(X_len, y_len)
        
        # 2. Train Optimizer Model (Growth Rate) - Includes Growth_Days for life stage!
        X_rate = df[["Temp", "Hum", "TDS", "pH", "Growth_Days"]]
        y_rate = df["Growth_Rate"]
        m_rate = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        m_rate.fit(X_rate, y_rate)
        
        # Store models
        st.session_state.model_len = m_len
        st.session_state.model_rate = m_rate
        
        # Calculate Tracker Accuracy for display
        y_pred = m_len.predict(X_len)
        mape = np.mean(np.abs((y_len - y_pred) / y_len)) * 100
        st.session_state.acc = 100 - mape
        
        st.success("Digital Twin Models Trained!")

    if st.session_state.model_len:
        st.metric("Tracker Accuracy", f"{st.session_state.acc:.2f}%")
        st.caption("Ready to simulate.")
    else:
        st.info("Awaiting training...")

# ============================================================================
# COLUMN 2: SENSOR INPUT CONTROLS
# ============================================================================
with col_sensor:
    st.subheader("2. Environment Controls")
    
    # Grab last known sensors from simulation history to persist user choices
    last_h = st.session_state.sim_history[-1]
    
    temp = st.number_input("Temperature (°C)", min_value=18.0, max_value=33.5, value=float(last_h.get("temp", 28.0)), step=0.5)
    hum = st.number_input("Humidity (%)", min_value=50.0, max_value=80.0, value=float(last_h.get("hum", 65.0)), step=1.0)
    tds = st.number_input("TDS / Nutrients (ppm)", min_value=400.0, max_value=800.0, value=float(last_h.get("tds", 550.0)), step=10.0)
    ph = st.number_input("pH", min_value=6.0, max_value=6.8, value=float(last_h.get("ph", 6.4)), step=0.1)

# ============================================================================
# COLUMN 3: SIMULATION ACTIONS
# ============================================================================
with col_result:
    st.subheader("3. Simulation Actions")
    st.metric("Simulation Day", f"Day {st.session_state.sim_day}")
    st.metric("Total Length", f"{st.session_state.sim_length:.2f} cm")
    
    # Let the user choose how long to run the simulation before pausing
    break_days = st.number_input("Simulation Break (Days):", min_value=1, max_value=15, value=5, help="How many days to simulate before pausing for AI suggestions.")
    
    if st.button("🚀 Simulate Next Phase", use_container_width=True):
        if not st.session_state.model_rate:
            st.warning("Please train the models first!")
        else:
            # Loop through the break days, predicting rate for EACH day dynamically
            for d in range(break_days):
                # Predict rate using current sensors AND the specific day it is on
                predicted_rate = st.session_state.model_rate.predict([[temp, hum, tds, ph, st.session_state.sim_day]])[0]
                
                st.session_state.sim_day += 1
                noise = np.random.normal(0, 0.05) # Add a tiny bit of natural variance
                st.session_state.sim_length += max(0, predicted_rate + noise)
                
                # Log step in history
                st.session_state.sim_history.append({
                    "day": st.session_state.sim_day,
                    "length": st.session_state.sim_length,
                    "temp": temp, "tds": tds, "hum": hum, "ph": ph
                })
            st.rerun()

    if st.button("🔄 Reset Digital Twin", use_container_width=True):
        st.session_state.sim_day = 1
        st.session_state.sim_length = 1.0
        st.session_state.sim_history = [{"day": 1, "length": 1.0, "temp": 28.0, "tds": 550.0, "hum": 65.0, "ph": 6.4}]
        st.rerun()

st.divider()

# ============================================================================
# TABS FOR DETAILED ANALYSIS AND VISUALIZATION
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Simulation & AI Suggestions", 
    "Analysis Plots", 
    "Dataset Preview", 
    "Simulation History"
])

# ============================================================================
# TAB 1: SIMULATION TIMELINE AND AI
# ============================================================================
with tab1:
    left_col, right_col = st.columns([3, 1])
    
    with left_col:
        # Plot 1 & 2: The Timeline & Sensor Health
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # --- SUBPLOT 1: ACTUAL TRAJECTORY TIMELINE ---
        ax1 = axes[0]
        days = [h["day"] for h in st.session_state.sim_history]
        lengths = [h["length"] for h in st.session_state.sim_history]
        
        # Color phases
        for idx, (lo, hi) in enumerate([(1, 16), (17, 32), (33, 48)]):
            ax1.axvspan(lo, hi, alpha=0.07, color=CLR_PHASE[idx])
            
        ax1.plot(days, lengths, marker='o', color="#1D9E75", linewidth=2, markersize=4, label="Simulated Growth")
        ax1.set_xlim(0, 50)
        ax1.set_xlabel("Growth Day")
        ax1.set_ylabel("Total Length (cm)")
        ax1.set_title("Plant Growth Over Time")
        ax1.grid(alpha=0.3)
        
        # Create legend with growth phases
        handles = [mpatches.Patch(color=c, label=l, alpha=0.7) for c, l in zip(CLR_PHASE, ["Seedling", "Vegetative", "Mature"])]
        handles.append(plt.Line2D([0], [0], color="#1D9E75", marker='o', label="Trajectory"))
        ax1.legend(handles=handles, fontsize=7, loc="upper left")

        # --- SUBPLOT 2: SENSOR HEALTH BARS ---
        ax2 = axes[1]
        ax2.set_title("Current Sensor Health vs Optimal Range", fontsize=10)
        current_vals = {"Temp": temp, "Hum": hum, "TDS": tds, "pH": ph}
        clr_map = {"good": CLR_GOOD, "warn": CLR_WARN, "bad": CLR_BAD}
        
        for i, k in enumerate(KEYS):
            rMin, rMax = RANGES[k]
            oLo, oHi = OPTIMA[k]
            val = current_vals[k]
            norm = lambda v, mn=rMin, mx=rMax: (v - mn) / (mx - mn)
            
            ax2.barh(i, 1, left=0, color="#F1EFE8", height=0.5)
            ax2.barh(i, norm(oHi) - norm(oLo), left=norm(oLo), color="#EAF3DE", height=0.5)
            
            s = health_status(k, val)
            ax2.plot(norm(val), i, "o", markersize=9, color=clr_map[s])
            ax2.text(1.03, i, f"{val}{UNITS[k]}", va="center", fontsize=8)
            ax2.text(-0.03, i, k, va="center", ha="right", fontsize=8)
            
        ax2.set_xlim(-0.18, 1.25)
        ax2.set_ylim(-0.6, len(KEYS) - 0.4)
        ax2.axis("off")
        
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right_col:
        st.markdown(f"**🧠 AI Agronomist (Day {st.session_state.sim_day})**")
        if st.session_state.model_rate:
            # Pass the current simulation day to get stage-specific advice!
            tips, base_rate = get_rate_suggestions(st.session_state.model_rate, temp, hum, tds, ph, st.session_state.sim_day)
            
            st.info(f"Predicted Speed: **{base_rate:.2f} cm/day**")
            
            actionable = [t for t in tips if t["dir"] != "ok"]
            if not actionable:
                st.success("All sensors are well-optimized! Growth is near-peak for this plant stage.")
            else:
                st.write("**To grow FASTER next phase:**")
                for t in tips:
                    k, d, nv = t["key"], t["dir"], t["new_val"]
                    if d == "ok": continue
                    arrow = "🔼 INCREASE" if d == "up" else "🔽 DECREASE"
                    fmt = f"{nv:.1f}" if k in ("Temp", "pH") else str(int(round(nv / 10) * 10))
                    
                    h_tag = f"[{t['health'].upper()}]"
                    st.markdown(f"{arrow} **{k}** to ~{fmt}{UNITS[k]}  \n*(+{t['boost']:.2f} cm/day)*")
        else:
            st.warning("Train models to receive suggestions.")

# ============================================================================
# TAB 2: ANALYSIS PLOTS
# ============================================================================
with tab2:
    if not st.session_state.model_len:
        st.info("Train the models first to see analysis plots.")
    else:
        plot_choice = st.radio("Choose plot:", [
            "Feature Importance (Total Length Model)", 
            "Feature Importance (Growth Rate Model)",
            "Correlation Heatmap"
        ], horizontal=True)
        
        if "Total Length" in plot_choice:
            imp_df = pd.DataFrame({
                "Feature": ["Temp", "Hum", "TDS", "pH", "Growth_Days"],
                "Importance": st.session_state.model_len.feature_importances_
            }).sort_values("Importance", ascending=False)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Importance", y="Feature", data=imp_df, palette="viridis", ax=ax)
            ax.set_title("What drives Absolute Length? (Notice Time dominates)")
            st.pyplot(fig)
            plt.close(fig)
            
        elif "Growth Rate" in plot_choice:
            imp_df = pd.DataFrame({
                "Feature": ["Temp", "Hum", "TDS", "pH", "Growth_Days"],
                "Importance": st.session_state.model_rate.feature_importances_
            }).sort_values("Importance", ascending=False)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Importance", y="Feature", data=imp_df, palette="magma", ax=ax)
            ax.set_title("What drives Growth Speed? (Notice Sensors matter more here!)")
            st.pyplot(fig)
            plt.close(fig)
            
        elif plot_choice == "Correlation Heatmap":
            cols = ["Temp", "Hum", "TDS", "pH", "Growth_Days", "Growth_Length", "Growth_Rate"]
            corr = df[cols].corr()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
            plt.close(fig)

# ============================================================================
# TAB 3: DATASET PREVIEW
# ============================================================================
with tab3:
    st.subheader(f"ThingSpeak Data - {len(df)} rows")
    display_cols = ["Temp", "Hum", "TDS", "pH", "Growth_Days", "Growth_Length", "Growth_Rate"]
    st.dataframe(df[display_cols], use_container_width=True)

# ============================================================================
# TAB 4: SIMULATION HISTORY
# ============================================================================
with tab4:
    st.subheader("Simulation Memory Log")
    if len(st.session_state.sim_history) <= 1:
        st.info("No phases simulated yet. Run the simulation to build history.")
    else:
        hist_df = pd.DataFrame(st.session_state.sim_history)
        hist_df.columns = [c.capitalize() for c in hist_df.columns]
        st.dataframe(hist_df, use_container_width=True)