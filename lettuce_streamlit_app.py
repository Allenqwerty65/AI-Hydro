import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

CHANNEL_ID = "3179322"
READ_API_KEY = "6VGSBPTD8VN1RY7T"

st.set_page_config(layout="wide")
st.title("Lettuce Growth ML Predictor")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

df = None
model = None

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin-1")
    df.columns = ["Plant_ID", "Date", "Temp", "Hum", "TDS", "pH", "Growth_Days", "Growth_Length"]
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

if df is not None:
    if st.button("Train Model"):
        X = df[["Temp", "Hum", "TDS", "pH"]]
        y = df["Growth_Length"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        acc = 100 - mape

        st.success("Model trained!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.2f}%")
        col2.metric("MAE", f"{mae:.3f}")
        col3.metric("R² Score", f"{r2:.4f}")

        st.session_state["model"] = model
        st.session_state["mae"] = mae
        st.session_state["X"] = X
        st.session_state["y"] = y

st.subheader("Sensor Inputs")

temp = st.number_input("Temperature (°C)", value=35.0)
hum = st.number_input("Humidity (%)", value=60.0)
tds = st.number_input("TDS / Nutrients (ppm)", value=680.0)
ph = st.number_input("pH", value=7.0)

if st.button("Predict Growth Length"):
    if "model" not in st.session_state:
        st.warning("Please train the model first.")
    else:
        model = st.session_state["model"]
        mae = st.session_state["mae"]

        input_data = np.array([[temp, hum, tds, ph]])
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.success(f"Predicted Growth Length: {prediction:.2f} cm")
        st.info(f"Confidence Range: {prediction - mae:.2f} – {prediction + mae:.2f} cm")

if df is not None and "model" in st.session_state:
    model = st.session_state["model"]
    X = st.session_state["X"]
    y = st.session_state["y"]

    st.subheader("Analysis Plots")

    tab1, tab2, tab3 = st.tabs(["Predicted vs Actual", "Feature Importance", "Correlation Heatmap"])

    with tab1:
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        st.pyplot(fig)

    with tab2:
        importances = model.feature_importances_
        features = ["Temp", "Hum", "TDS", "pH"]

        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=features, ax=ax)
        st.pyplot(fig)

    with tab3:
        corr = df[["Temp", "Hum", "TDS", "pH", "Growth_Days", "Growth_Length"]].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)
