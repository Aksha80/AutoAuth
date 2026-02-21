import streamlit as st
import os
from src.nlp_engine import NLPExtractor
from src.ml_engine import AuthML
from src.data_generator import generate_synthetic_data

st.set_page_config(page_title="AuthAI", layout="wide")

st.title("🛡️ AuthAI: Smart Prior Authorization")
st.caption("AI-driven Prior Authorization Decision Support System")

# Auto-train if model missing
if not os.path.exists("model/classifier.pkl"):
    st.warning("Model not found. Training automatically...")
    df = generate_synthetic_data(1500)
    ml_temp = AuthML()
    ml_temp.train_on_synthetic(df)
    st.success("Model trained successfully!")

col1, col2 = st.columns(2)

with col1:
    patient_note = st.text_area(
        "Paste Patient Progress Note:",
        height=200,
        placeholder="Patient has chronic back pain for 6 months. Severity is moderate. Completed physical therapy..."
    )

if st.button("Analyze Authorization Request") and patient_note:

    extractor = NLPExtractor()
    clin_features = extractor.extract_features(patient_note)

    # Policy logic
    policy_match_score = 0.6 if clin_features['duration_months'] >= 3 else 0.3

    ml = AuthML()

    input_vector = [
        clin_features['duration_months'],
        clin_features['severity_score'],
        int(clin_features['previous_therapy']),
        policy_match_score
    ]

    prob = ml.predict(input_vector)

    with col2:

        # Probability Display
        st.metric("Approval Probability", f"{prob*100:.1f}%")
        st.progress(prob)

        # Human Explanation Instead of JSON
        st.subheader("🔎 Extracted Clinical Summary")

        severity_map = {1: "Mild", 2: "Moderate", 3: "Severe"}

        st.write(f"""
        • **Duration Identified:** {clin_features['duration_months']} month(s)  
        • **Severity Level:** {severity_map.get(clin_features['severity_score'], "Mild")}  
        • **Previous Therapy Attempted:** {"Yes" if clin_features['previous_therapy'] else "No"}  
        """)

        # Decision Band
        st.subheader("📌 AutoAuth Decision")

        if prob >= 0.80:
            status = "APPROVED"
            st.success("✅ Final Decision: APPROVED")
            st.write("""
            Based on the clinical duration, severity level, and prior therapy attempts,  
            the case strongly aligns with policy requirements.  
            **AutoAuth has automatically approved this request.**
            """)

        elif prob >= 0.55:
            status = "REQUIRES REVIEW"
            st.warning("⚠ Final Decision: REQUIRES MANUAL REVIEW")
            st.write("""
            The case partially meets authorization criteria.  
            AutoAuth recommends manual review by a clinical specialist  
            before final approval.
            """)

        else:
            status = "DENIED"
            st.error("❌ Final Decision: DENIED")
            st.write("""
            Based on current extracted information,  
            the request does not sufficiently meet policy criteria.  
            AutoAuth recommends denial unless additional documentation is provided.
            """)
    
