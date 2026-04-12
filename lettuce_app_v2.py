# ============================================================================
# LETTUCE GROWTH ML PREDICTOR — DUAL MODE v2
# Mode 1: Predict Growth Length at a specific day
# Mode 2: Full 48-day simulation with AI suggestions at break intervals
# Growth cycle: Day 1 → Day 48 (confirmed from dataset)
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================================
# CONFIG
# ============================================================================
CHANNEL_ID   = "3326913"
READ_API_KEY = "1YBWG6QWWPA9TNLT"
MAX_DAYS     = 48

RANGES = {"Temp": (18.0, 33.5), "Hum": (50.0, 80.0), "TDS": (400.0, 800.0), "pH": (6.0, 6.8)}
OPTIMA = {"Temp": (27.0, 30.0), "Hum": (62.0, 72.0), "TDS": (560.0, 680.0), "pH": (6.2, 6.5)}
UNITS  = {"Temp": "°C", "Hum": "%", "TDS": " ppm", "pH": ""}
KEYS   = ["Temp", "Hum", "TDS", "pH"]
CLR_PHASE = ["#9FE1CB", "#5DCAA5", "#1D9E75"]

# ============================================================================
# HELPERS
# ============================================================================
def clamp(v, mn, mx):
    return max(mn, min(mx, v))

def health_status(key, val):
    lo, hi = OPTIMA[key]
    if lo <= val <= hi:
        return "good"
    span = RANGES[key][1] - RANGES[key][0]
    dist = (lo - val) / span if val < lo else (val - hi) / span
    return "bad" if dist > 0.15 else "warn"

def get_phase(day):
    if day <= 16:  return "🌱 Seedling"
    if day <= 32:  return "🌿 Vegetative"
    return "🥬 Mature"

def get_suggestions(model_rate, day, temp, hum, tds, ph):
    current   = [temp, hum, tds, ph]
    base_rate = model_rate.predict([[temp, hum, tds, ph, day]])[0]
    tips = []
    for i, k in enumerate(KEYS):
        rMin, rMax = RANGES[k]
        up_v  = clamp(current[i] * 1.12, rMin, rMax)
        dn_v  = clamp(current[i] * 0.88, rMin, rMax)
        up_in = current[:]; up_in[i] = up_v
        dn_in = current[:]; dn_in[i] = dn_v
        rate_up = model_rate.predict([[*up_in, day]])[0]
        rate_dn = model_rate.predict([[*dn_in, day]])[0]
        h = health_status(k, current[i])
        if rate_up - base_rate > 0.005 and rate_up >= rate_dn:
            tips.append({"key": k, "dir": "UP",   "boost": rate_up - base_rate, "new_val": up_v,       "health": h})
        elif rate_dn - base_rate > 0.005:
            tips.append({"key": k, "dir": "DOWN", "boost": rate_dn - base_rate, "new_val": dn_v,       "health": h})
        else:
            tips.append({"key": k, "dir": "OK",   "boost": 0,                   "new_val": current[i], "health": h})
    tips.sort(key=lambda x: -x["boost"])
    return tips, base_rate

# ============================================================================
# DATA + MODEL (CACHED)
# ============================================================================
@st.cache_data(show_spinner="📡 Fetching data from ThingSpeak...")
def load_data():
    url  = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
    resp = requests.get(url, params={"api_key": READ_API_KEY, "results": 8000})
    df   = pd.DataFrame(resp.json().get("feeds", []))
    df   = df.rename(columns={
        "field1": "Temp", "field2": "Hum", "field3": "TDS",
        "field4": "pH",   "field5": "Growth_Days", "field6": "Growth_Length"
    })
    for col in ["Temp","Hum","TDS","pH","Growth_Days","Growth_Length"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["Growth_Rate"] = df["Growth_Length"] / df["Growth_Days"]
    return df

@st.cache_resource(show_spinner="🧠 Training models...")
def train_models(n_rows):
    df = load_data()
    out = {}
    for target, col in [("length", "Growth_Length"), ("rate", "Growth_Rate")]:
        X = df[["Temp","Hum","TDS","pH","Growth_Days"]]
        y = df[col]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=4, subsample=0.8, random_state=42)
        m.fit(Xtr, ytr)
        yp  = m.predict(Xte)
        mae = mean_absolute_error(yte, yp)
        r2  = r2_score(yte, yp)
        acc = 100 - np.mean(np.abs((yte - yp) / yte)) * 100
        out[target] = {"model": m, "mae": mae, "r2": r2, "acc": acc}
    return out

# ============================================================================
# PAGE SETUP
# ============================================================================
st.set_page_config(page_title="🥬 Lettuce Growth Predictor", layout="wide")
st.title("🥬 Lettuce Growth ML Predictor")
st.caption("Dual-mode predictor — single day point prediction OR full 48-day lifecycle simulation with AI sensor suggestions")

df = load_data()
if df is None or df.empty:
    st.error("❌ No data found in ThingSpeak. Please upload data first.")
    st.stop()

models       = train_models(len(df))
model_length = models["length"]["model"]
model_rate   = models["rate"]["model"]
mae_len      = models["length"]["mae"]
acc_len      = models["length"]["acc"]
acc_rate     = models["rate"]["acc"]

# Banner
st.success(f"✅ **{len(df):,} rows** loaded · Growth cycle confirmed: **Day 1 – Day {MAX_DAYS}**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("📏 Length Model Accuracy", f"{acc_len:.1f}%")
c2.metric("📏 Length MAE",            f"±{mae_len:.2f} cm")
c3.metric("⚡ Rate Model Accuracy",   f"{acc_rate:.1f}%")
c4.metric("🗓️ Full Growth Cycle",     f"{MAX_DAYS} days")
st.divider()

# ============================================================================
# TABS
# ============================================================================
tab1, tab2 = st.tabs(["📏  Mode 1 — Point Prediction", "🌱  Mode 2 — Full Growth Simulation"])

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MODE 1                                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
with tab1:
    st.subheader("📏 Predict Length at a Specific Day")
    st.caption("Choose a growth day and sensor readings → get the predicted length at that point in time.")

    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        st.markdown("**Inputs**")
        day  = st.slider("Growth Day",           1,    MAX_DAYS, 20)
        temp = st.slider("Temperature (°C)",     18.0, 33.5,    28.0, step=0.5)
        hum  = st.slider("Humidity (%)",         50.0, 80.0,    65.0, step=1.0)
        tds  = st.slider("TDS / Nutrients (ppm)",400.0,800.0,   580.0,step=10.0)
        ph   = st.slider("pH",                   6.0,  6.8,     6.4,  step=0.1)
        st.info(f"Phase: **{get_phase(day)}** — Day {day} of {MAX_DAYS}")
        go = st.button("🔮 Predict", use_container_width=True, type="primary")

    with col_r:
        st.markdown("**Results**")
        if go:
            pl = model_length.predict([[temp, hum, tds, ph, day]])[0]
            pr = model_rate.predict([[temp,   hum, tds, ph, day]])[0]
            ra, rb = st.columns(2)
            ra.metric("Predicted Length",   f"{pl:.2f} cm")
            rb.metric("Growth Rate",        f"{pr:.3f} cm/day")
            st.info(f"Confidence range: **{pl-mae_len:.2f} – {pl+mae_len:.2f} cm**")

            # Sensor health
            st.markdown("**Sensor Health**")
            icons = {"good": "🟢 Optimal", "warn": "🟠 Suboptimal", "bad": "🔴 Off-range"}
            hcols = st.columns(4)
            for col, k, v in zip(hcols, KEYS, [temp, hum, tds, ph]):
                col.metric(k, f"{v}{UNITS[k]}", icons[health_status(k, v)])

            # 5-day forecast table
            st.markdown("**5-Day Forecast**")
            fc = [{"Day": d, "Phase": get_phase(d),
                   "Length (cm)": round(model_length.predict([[temp,hum,tds,ph,d]])[0], 2),
                   "Rate (cm/day)": round(model_rate.predict([[temp,hum,tds,ph,d]])[0], 3)}
                  for d in range(day, min(day+6, MAX_DAYS+1))]
            st.dataframe(pd.DataFrame(fc), use_container_width=True, hide_index=True)

            # Top suggestion
            tips, _ = get_suggestions(model_rate, day, temp, hum, tds, ph)
            top = [t for t in tips if t["dir"] != "OK"]
            if top:
                t   = top[0]
                fmt = f"{t['new_val']:.1f}" if t['key'] in ("Temp","pH") else f"{int(t['new_val'])}"
                arr = "⬆️ increase" if t["dir"] == "UP" else "⬇️ decrease"
                st.warning(f"💡 **Top tip:** {arr} **{t['key']}** → {fmt}{UNITS[t['key']]} to gain +{t['boost']:.4f} cm/day")
            else:
                st.success("✅ All sensors are well-optimised for this growth stage!")
        else:
            st.info("👈 Set inputs and click **Predict**")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MODE 2 — SIMULATION                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
with tab2:
    st.subheader("🌱 Full 48-Day Growth Simulation")
    st.caption("""
    The simulation runs the **complete 48-day lettuce lifecycle**.
    At every break point the AI analyses sensor conditions and suggests
    which values to adjust to maximise growth rate for the next phase.
    The top suggestion is automatically applied in the simulation.
    """)

    # Settings
    st.markdown("#### ⚙️ Settings")
    set1, set2 = st.columns([1, 2])
    with set1:
        break_interval = st.selectbox(
            "AI checks in every...",
            [2, 3, 4, 5, 7, 8, 10, 12, 16],
            index=2,
            format_func=lambda x: f"Every {x} days ({MAX_DAYS//x} check-ins)"
        )

    with set2:
        st.markdown("**Break points in this simulation:**")
        break_days_preview = list(range(break_interval, MAX_DAYS, break_interval))
        parts = []
        for d in range(1, MAX_DAYS + 1):
            if d in break_days_preview:
                parts.append(f"**🔔d{d}**")
            elif d in (1, MAX_DAYS):
                parts.append(f"`d{d}`")
        st.markdown("  →  ".join(parts))
        st.caption("🔔 = AI suggestion break  ·  🌱 d1–16 Seedling  ·  🌿 d17–32 Vegetative  ·  🥬 d33–48 Mature")

    st.divider()

    # Starting sensor values
    st.markdown("#### 🌡️ Starting Sensor Values (Day 1)")
    s1, s2, s3, s4 = st.columns(4)
    init_temp = s1.number_input("Temp (°C)",    18.0, 33.5, 28.0, step=0.5, help="Optimal: 27–30°C")
    init_hum  = s2.number_input("Humidity (%)", 50.0, 80.0, 65.0, step=1.0, help="Optimal: 62–72%")
    init_tds  = s3.number_input("TDS (ppm)",    400.0,800.0,580.0,step=10.0,help="Optimal: 560–680 ppm")
    init_ph   = s4.number_input("pH",           6.0,  6.8,  6.4,  step=0.1, help="Optimal: 6.2–6.5")

    run_btn = st.button("▶️ Run Full 48-Day Simulation", use_container_width=True, type="primary")

    if run_btn:
        temp, hum, tds, ph = init_temp, init_hum, init_tds, init_ph
        sim_log, sim_breaks = [], []

        for day in range(1, MAX_DAYS + 1):
            pl = model_length.predict([[temp, hum, tds, ph, day]])[0]
            pr = model_rate.predict([[temp,   hum, tds, ph, day]])[0]
            sim_log.append({"day": day, "length": round(pl, 3), "rate": round(pr, 4),
                             "temp": round(temp,1), "hum": round(hum,1),
                             "tds":  round(tds, 1), "ph":  round(ph, 2)})

            if day % break_interval == 0 and day < MAX_DAYS:
                tips, base_rate = get_suggestions(model_rate, day, temp, hum, tds, ph)
                sim_breaks.append({
                    "day": day, "phase": get_phase(day),
                    "pred_len": round(pl, 2), "base_rate": round(base_rate, 4),
                    "tips": tips,
                    "sensors": {"Temp": temp, "Hum": hum, "TDS": tds, "pH": ph}
                })
                # Auto-apply top actionable suggestion
                for t in tips:
                    if t["dir"] != "OK":
                        if   t["key"] == "Temp": temp = t["new_val"]
                        elif t["key"] == "Hum":  hum  = t["new_val"]
                        elif t["key"] == "TDS":  tds  = t["new_val"]
                        elif t["key"] == "pH":   ph   = t["new_val"]
                        break

        sim_df = pd.DataFrame(sim_log)
        final  = sim_df.iloc[-1]

        # Summary
        st.success(f"✅ Simulation complete! **Final predicted length on Day 48: {final['length']:.2f} cm**")
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Final Length",      f"{final['length']:.2f} cm")
        sm2.metric("Peak Rate",         f"{sim_df['rate'].max():.3f} cm/day",
                   f"Day {int(sim_df.loc[sim_df['rate'].idxmax(),'day'])}")
        sm3.metric("Average Rate",      f"{sim_df['rate'].mean():.3f} cm/day")
        sm4.metric("AI Check-ins Done", len(sim_breaks))

        st.divider()

        # Charts
        st.markdown("#### 📈 Growth Curves")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
        fig.patch.set_facecolor("#fafffe")

        phases = [(1,16,CLR_PHASE[0],"Seedling d1–16"),
                  (17,32,CLR_PHASE[1],"Vegetative d17–32"),
                  (33,48,CLR_PHASE[2],"Mature d33–48")]

        bdays = [b["day"] for b in sim_breaks]
        blens = [sim_df.loc[sim_df["day"]==d,"length"].values[0] for d in bdays]
        brate = [sim_df.loc[sim_df["day"]==d,"rate"].values[0]   for d in bdays]

        for lo, hi, clr, lbl in phases:
            ax1.axvspan(lo, hi, alpha=0.09, color=clr, label=lbl)
            ax2.axvspan(lo, hi, alpha=0.09, color=clr)

        ax1.plot(sim_df["day"], sim_df["length"], color="#1D9E75", lw=2.5, zorder=3)
        ax1.fill_between(sim_df["day"], sim_df["length"], alpha=0.1, color="#1D9E75")
        ax1.scatter(bdays, blens, color="orange", s=70, zorder=5, label="AI check-in")
        for d, l in zip(bdays, blens):
            ax1.annotate(f"d{d}", (d,l), xytext=(2,6), textcoords="offset points",
                         fontsize=7, color="darkorange")
        ax1.set_xlabel("Growth Day"); ax1.set_ylabel("Length (cm)")
        ax1.set_title("Predicted Growth Length — Full Lifecycle", fontweight="bold")
        ax1.set_xlim(1, MAX_DAYS); ax1.grid(axis="y", alpha=0.2)
        handles = [mpatches.Patch(color=c, alpha=0.5, label=l) for _,_,c,l in phases]
        handles += [plt.Line2D([],[],marker="o",color="orange",ls="none",label="AI check-in",markersize=7)]
        ax1.legend(handles=handles, fontsize=7, loc="upper left")

        ax2.fill_between(sim_df["day"], sim_df["rate"], alpha=0.2, color="#5DCAA5")
        ax2.plot(sim_df["day"], sim_df["rate"], color="#1D9E75", lw=2.5)
        ax2.scatter(bdays, brate, color="orange", s=70, zorder=5)
        ax2.axhline(sim_df["rate"].mean(), color="gray", ls=":", lw=1.2,
                    label=f"Avg {sim_df['rate'].mean():.3f} cm/day")
        ax2.set_xlabel("Growth Day"); ax2.set_ylabel("Growth Rate (cm/day)")
        ax2.set_title("Growth Rate Over Lifecycle", fontweight="bold")
        ax2.set_xlim(1, MAX_DAYS); ax2.grid(axis="y", alpha=0.2); ax2.legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # AI Suggestion Breaks
        st.divider()
        st.markdown("#### 🤖 AI Suggestions at Each Break")
        st.caption("At each orange dot above the AI analysed conditions and recommended sensor changes. The best suggestion was automatically applied to the next phase of the simulation.")

        h_icon  = {"good": "🟢", "warn": "🟠", "bad": "🔴"}
        h_label = {"good": "Optimal", "warn": "Suboptimal", "bad": "Off-range"}

        for b in sim_breaks:
            with st.expander(
                f"{b['phase']}  ·  📍 Day {b['day']}  ·  "
                f"Length: **{b['pred_len']:.2f} cm**  ·  "
                f"Rate: {b['base_rate']:.4f} cm/day",
                expanded=False
            ):
                # Current sensors at this break
                st.markdown("**Sensor readings at this break:**")
                sc = st.columns(4)
                for col, k in zip(sc, KEYS):
                    v = b["sensors"][k]
                    h = health_status(k, v)
                    col.metric(f"{h_icon[h]} {k}", f"{v}{UNITS[k]}", h_label[h])

                st.markdown("**What AI recommends for next phase:**")
                tc = st.columns(4)
                for col, t in zip(tc, b["tips"]):
                    k   = t["key"]
                    nv  = t["new_val"]
                    fmt = f"{nv:.1f}" if k in ("Temp","pH") else f"{int(nv)}"
                    if t["dir"] == "OK":
                        col.success(f"**{k}** ✅\n\nKeep at **{fmt}{UNITS[k]}**\n\nAlready optimal")
                    elif t["dir"] == "UP":
                        col.warning(f"**{k}** ⬆️ Increase\n\nTarget: **{fmt}{UNITS[k]}**\n\nGain: +{t['boost']:.4f} cm/day")
                    else:
                        col.info(f"**{k}** ⬇️ Decrease\n\nTarget: **{fmt}{UNITS[k]}**\n\nGain: +{t['boost']:.4f} cm/day")

        # Full log
        st.divider()
        with st.expander("📋 View full day-by-day simulation log", expanded=False):
            disp = sim_df.copy()
            disp.insert(1, "Phase", disp["day"].apply(get_phase))
            st.dataframe(disp.rename(columns={
                "day":"Day","length":"Length (cm)","rate":"Rate (cm/day)",
                "temp":"Temp (°C)","hum":"Hum (%)","tds":"TDS (ppm)","ph":"pH"
            }), use_container_width=True, hide_index=True)

# Sidebar
with st.sidebar:
    st.header("📊 Dataset")
    st.metric("Rows loaded",     f"{len(df):,}")
    st.metric("Growth cycle",    "Day 1 – 48")
    st.metric("Max length",      f"{df['Growth_Length'].max():.1f} cm")
    st.metric("Avg growth rate", f"{df['Growth_Rate'].mean():.3f} cm/day")
    st.divider()
    st.markdown("""
**📏 Mode 1 — Point Prediction**  
Pick any day (1–48) + sensor values → predicted length at that moment + 5-day forecast

**🌱 Mode 2 — Simulation**  
1. Choose break interval (e.g. every 3 days)  
2. Set starting sensor values  
3. Click **Run Simulation**  
4. See full 48-day curve  
5. Expand each break to see AI suggestions  

---
**Growth Phases**  
🌱 Seedling — Day 1–16  
🌿 Vegetative — Day 17–32  
🥬 Mature — Day 33–48  
    """)
