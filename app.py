import streamlit as st
import os
from src.nlp_engine import NLPExtractor
from src.ml_engine import AuthML
from src.data_generator import generate_synthetic_data

st.set_page_config(page_title="AuthAI", layout="wide")

st.title("🛡️ AuthAI: Smart Prior Authorization")

# Auto-train model if not exists
if not os.path.exists("model/classifier.pkl"):
    st.warning("Model not found. Training automatically...")
    df = generate_synthetic_data(1000)
    ml_temp = AuthML()
    ml_temp.train_on_synthetic(df)
    st.success("Model trained!")

col1, col2 = st.columns(2)

with col1:
    patient_note = st.text_area(
        "Paste Patient Progress Note:",
        height=200,
        placeholder="Patient has chronic back pain for 6 months. Severity is moderate. Completed physical therapy..."
    )

if st.button("Analyze Authorization Request"):

    extractor = NLPExtractor()
    clin_features = extractor.extract_features(patient_note)

    policy_match_score = 0.9 if clin_features['duration_months'] >= 3 else 0.4

    ml = AuthML()

    input_vector = [
        clin_features['duration_months'],
        clin_features['severity_score'],
        int(clin_features['previous_therapy']),
        policy_match_score
    ]

    prob, importance = ml.predict(input_vector)

    with col2:
        st.metric("Approval Probability", f"{prob*100:.1f}%")
        st.progress(prob)

        st.write("Extracted Features:")
        st.json(clin_features)

        status = "APPROVED" if prob > 0.7 else "DENIED"
        st.success(f"Final Decision: {status}")
