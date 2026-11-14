
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes AI Dashboard",
    page_icon="🧬",
    layout="wide"
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
DATASET_PATH = BASE_DIR / "data" / "diabetes.csv"

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #ffffff;
}

.main-title {
    font-size: 38px;
    font-weight: bold;
    color: #38bdf8;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.metric-card {
    background: #172554;
    padding: 22px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #334155;
}

.metric-title {
    color: #94a3b8;
    font-size: 15px;
}

.metric-value {
    color: #38bdf8;
    font-size: 27px;
    font-weight: bold;
}

.section-title {
    color: #38bdf8;
    font-size: 23px;
    font-weight: bold;
    margin-top: 25px;
}

div.stButton > button {
    background-color: #0891b2;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 22px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #0e7490;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        return None

    return pd.read_csv(DATASET_PATH)


data = load_dataset()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🧬 Diabetes AI Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Diabetes Risk Classification Using AI & Machine Learning</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# TOP DASHBOARD CARDS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">AI Model</div>
        <div class="metric-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Dataset</div>
        <div class="metric-value">768 Records</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Task</div>
        <div class="metric-value">Classification</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

st.subheader("Accuracy Comparison")

col1, col2 = st.columns(2)

with col1:
    st.metric("Logistic Regression", "70.78%")

with col2:
    st.metric("Random Forest", "74.03%")

st.subheader("Random Forest Evaluation Metrics")

m1, m2, m3 = st.columns(3)

m1.metric("Precision", "65%")
m2.metric("Recall", "56%")
m3.metric("F1 Score", "60%")

st.subheader("Confusion Matrix")

cm1, cm2 = st.columns(2)

with cm1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">True Negative</div>
        <div class="metric-value">84</div>
    </div>
    """, unsafe_allow_html=True)

with cm2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">False Positive</div>
        <div class="metric-value">16</div>
    </div>
    """, unsafe_allow_html=True)

cm3, cm4 = st.columns(2)

with cm3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">False Negative</div>
        <div class="metric-value">24</div>
    </div>
    """, unsafe_allow_html=True)

with cm4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">True Positive</div>
        <div class="metric-value">30</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# CUSTOM BAR CHART FUNCTION
# --------------------------------------------------

def show_bar_chart(labels, values, colors=None):
    if colors is None:
        colors = ["#38bdf8"] * len(labels)

    maximum = max(values) if values else 1

    for label, value, color in zip(labels, values, colors):
        percentage = (value / maximum) * 100 if maximum else 0

        st.markdown(f"""
        <div style="margin-bottom:18px;">
            <div style="display:flex; justify-content:space-between;">
                <span>{label}</span>
                <b>{value:.2f}</b>
            </div>
            <div style="
                background:#334155;
                border-radius:10px;
                height:13px;
                margin-top:6px;
            ">
                <div style="
                    background:{color};
                    width:{percentage:.2f}%;
                    height:13px;
                    border-radius:10px;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# --------------------------------------------------
# DATASET ANALYSIS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📈 Dataset Analysis</div>',
    unsafe_allow_html=True
)

if data is not None:

    total_records = len(data)
    positive = int((data["Outcome"] == 1).sum())
    negative = int((data["Outcome"] == 0).sum())

    d1, d2, d3 = st.columns(3)

    d1.metric("Total Records", total_records)
    d2.metric("Diabetes Positive", positive)
    d3.metric("Diabetes Negative", negative)

    # Outcome Distribution

    st.subheader("Diabetes Outcome Distribution")

    show_bar_chart(
        ["Positive", "Negative"],
        [positive, negative],
        ["#f87171", "#38bdf8"]
    )

    # Replace invalid zero values for analysis

    analysis_data = data.copy()

    columns_with_invalid_zero = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in columns_with_invalid_zero:
        analysis_data[column] = analysis_data[column].replace(
            0, float("nan")
        )

    # Average Glucose

    st.subheader("Average Glucose Level Analysis")

    glucose_avg = analysis_data.groupby(
        "Outcome"
    )["Glucose"].mean()

    if 0 in glucose_avg.index and 1 in glucose_avg.index:
        show_bar_chart(
            ["No Diabetes", "Diabetes"],
            [glucose_avg.loc[0], glucose_avg.loc[1]],
            ["#38bdf8", "#f87171"]
        )

    # Average BMI

    st.subheader("Average BMI Analysis")

    bmi_avg = analysis_data.groupby(
        "Outcome"
    )["BMI"].mean()

    if 0 in bmi_avg.index and 1 in bmi_avg.index:
        show_bar_chart(
            ["No Diabetes", "Diabetes"],
            [bmi_avg.loc[0], bmi_avg.loc[1]],
            ["#38bdf8", "#f87171"]
        )

else:
    st.error("Dataset not found. Check the data folder.")

# --------------------------------------------------
# PATIENT INPUT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🧑‍⚕️ Diabetes Risk Prediction</div>',
    unsafe_allow_html=True
)

st.write("Enter the following values to demonstrate the model prediction.")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=0
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=300.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )

    with col2:
        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=900.0,
            value=80.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=25.0
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25
        )

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes Risk"
    )

# --------------------------------------------------
# PREDICTION RESULT
# --------------------------------------------------

if submitted:

    if model is None:
        st.error("Model file not found. Please train the model first.")

    else:

        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]],
            columns=[
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
            ]
        )

        prediction = model.predict(input_data)[0]

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            probability = float(probabilities[1]) * 100

        st.markdown(
            '<div class="section-title">📋 Prediction Result</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.error(
                "Model result: Diabetes-positive classification"
            )

        else:

            st.success(
                "Model result: Diabetes-negative classification"
            )

        if probability is not None:
            st.metric(
                "Model Estimated Positive-Class Probability",
                f"{probability:.2f}%"
            )

        # Save prediction history

        record = {
            "Prediction": (
                "Positive" if prediction == 1 else "Negative"
            ),
            "Probability (%)": (
                round(probability, 2)
                if probability is not None else "Not available"
            ),
            "Age": age,
            "Glucose": glucose,
            "BMI": bmi
        }

        st.session_state.prediction_history.append(record)

# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📜 Prediction History</div>',
    unsafe_allow_html=True
)

if st.session_state.prediction_history:

    for number, record in enumerate(
        st.session_state.prediction_history,
        start=1
    ):

        st.markdown(f"""
        **Prediction {number}**

        - Result: {record['Prediction']}
        - Probability: {record['Probability (%)']}%
        - Age: {record['Age']}
        - Glucose: {record['Glucose']}
        - BMI: {record['BMI']}
        """)

        st.divider()

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()

else:
    st.info("No predictions have been made yet.")

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.warning("""
**Educational Disclaimer:** This project is developed for educational
and machine-learning demonstration purposes only. Predictions are not
medical diagnoses and must not be used to make healthcare decisions.
""")

st.markdown("""
<div style="text-align:center; color:#94a3b8; padding:20px;">
    Diabetes Risk Classification Using AI & Machine Learning
</div>
""", unsafe_allow_html=True)
