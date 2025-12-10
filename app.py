
import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Liver Cirrhosis Stage Predictor", page_icon="🩺", layout="wide")

project_root = Path(".")

@st.cache_resource
def load_artifacts():
    model = joblib.load(project_root / "model.pkl")
    with open(project_root / "features.json") as f:
        features = json.load(f)
    return model, features

def prediction_form(model, features):
    st.subheader("Patient Details")
    cols = st.columns(3)
    inputs = {}
    cat_cols = set(features.get("categorical_columns", []))
    num_cols = set(features.get("numeric_columns", []))

    for i, col in enumerate(features.get("feature_columns", [])):
        with cols[i % 3]:
            if col in cat_cols:
                inputs[col] = st.text_input(col, value="")
            elif col in num_cols:
                inputs[col] = st.number_input(col, value=0.0, step=0.1, format="%.4f")
            else:
                inputs[col] = st.text_input(col, value="")
    if st.button("Predict Stage", type="primary"):
        X = pd.DataFrame([inputs])
        pred = model.predict(X)[0]
        st.success(f"Predicted Stage: **{pred}**")
        try:
            proba = model.predict_proba(X)[0]
            st.write("Probabilities:")
            st.json({cls: float(p) for cls, p in zip(model.classes_, proba)})
        except Exception:
            pass

def dashboard():
    st.subheader("Dataset Insights")
    data_path = project_root / "data" / "liver_cirrhosis.csv"
    if not data_path.exists():
        st.warning("Dataset not found at data/liver_cirrhosis.csv")
        return
    df = pd.read_csv(data_path)
    st.dataframe(df.head(50), use_container_width=True)
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if num_cols:
        col = st.selectbox("Numeric column for histogram", num_cols, index=0)
        fig = plt.figure()
        df[col].dropna().hist(bins=30)
        st.pyplot(fig)
    if "Stage" in df.columns:
        fig2 = plt.figure()
        df["Stage"].value_counts().plot(kind="bar")
        plt.title("Stage Distribution")
        st.pyplot(fig2)

def main():
    st.title("🩺 Liver Cirrhosis Stage Prediction")
    tab1, tab2 = st.tabs(["Predict", "Dashboard"])
    with tab1:
        try:
            model, features = load_artifacts()
        except Exception as e:
            st.error(f"Failed to load model/features: {e}")
            st.stop()
        prediction_form(model, features)
    with tab2:
        dashboard()

if __name__ == "__main__":
    main()
