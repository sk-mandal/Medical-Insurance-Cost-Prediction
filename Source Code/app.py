import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load saved model and scaler
model = pickle.load(open("best_insurance_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Get feature order from scaler
feature_order = list(scaler.feature_names_in_)

# ----------- Page Styling -----------

st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")

# Custom CSS for cleaner UI
st.markdown("""
<style>
    .main {background-color: #F8F9FA;}
    div.stButton > button {
        width: 100%;
        background-color:#4CAF50;
        color:white;
        font-size:18px;
        padding:10px;
        border-radius:8px;
    }
    .result-box {
        background:#f0fff4;
        padding:15px;
        border-radius:10px;
        border-left:6px solid #4CAF50;
        font-size:20px;
        font-weight:bold;
    }
</style>
""", unsafe_allow_html=True)


# ----------- Header -----------
st.title("🏥 Medical Insurance Cost Prediction")
st.write("Enter patient details below to estimate the potential medical insurance cost.")


# ----------- Two Column Input Layout -----------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])


# ----------- Preprocessing -----------
def preprocess_input():

    data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": 1 if sex == "male" else 0,
        "smoker": 1 if smoker == "yes" else 0,
        "region_northeast": 1 if region == "northeast" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0
    }

    df = pd.DataFrame([data])

    # Ensure missing columns are added (if any)
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_order]
    return scaler.transform(df)


# ----------- Prediction Button -----------
predict_button = st.button("Predict Insurance Cost")

if predict_button:

    try:
        processed = preprocess_input()
        prediction = model.predict(processed)[0]

        # Reverse log transform if needed
        try:
            prediction = np.expm1(prediction)
        except:
            pass

        st.markdown(f"<div class='result-box'>Estimated Cost: $ {round(prediction, 2)}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Failed: {e}")


# Footer
st.write("-----------")
st.caption("Powered by Machine Learning | Model: Gradient Boosting Regressor | Made By: Subhash Kumar Mandal")
