import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load('XGBoost.pkl')

scaler_means = {
    "AST": 233.68,
    "LDH": 828.67,
    "U": 6.67,
    "L": 0.82
}

scaler_stds = {
    "AST": 280.91,
    "LDH": 715.69,
    "U": 4.26,
    "L": 0.63
}

cons_options = {
    0: 'No change in consciousness (0)',
    1: 'Change in consciousness (1)'
}

mv_options = {
    0: 'Not applied (0)',
    1: 'Applied (1)'
}

crrt_options = {
    0: 'Not applied (0)',
    1: 'Applied (1)'
}

feature_names = ['Consciousness', 'LDH', 'MV', 'AST', 'CRRT', 'U', 'L']

st.title("Bunyavirus Prognosis")

consciousness = st.selectbox("Consciousness Status (Consciousness):", options=list(cons_options.keys()), format_func=lambda x: cons_options[x])
ldh = st.number_input("Lactate Dehydrogenase (LDH, U/L):", min_value=0, max_value=5000, value=200)
mv = st.selectbox("Mechanical Ventilation (MV):", options=list(mv_options.keys()), format_func=lambda x: mv_options[x])
ast = st.number_input("Aspartate Aminotransferase (AST, U/L):", min_value=0, max_value=5000, value=30)
crrt = st.selectbox("Continuous Renal Replacement Therapy (CRRT):", options=list(crrt_options.keys()), format_func=lambda x: crrt_options[x])
u = st.number_input("Urea (mmol/L):", min_value=0.0, max_value=200.0, value=5.0)
l = st.number_input("Lymphocyte Percentage (%):", min_value=0.0, max_value=100.0, value=20.0)

ldh_standardized = (ldh - scaler_means["LDH"]) / scaler_stds["LDH"]
ast_standardized = (ast - scaler_means["AST"]) / scaler_stds["AST"]
u_standardized = (u - scaler_means["U"]) / scaler_stds["U"]
l_standardized = (l - scaler_means["L"]) / scaler_stds["L"]

feature_values = [consciousness, ldh_standardized, mv, ast_standardized, crrt, u_standardized, l_standardized]
features = np.array([feature_values])

if st.button("Predict"):
    predicted_probabilities = model.predict_proba(features)[0]

    st.write(f"**Probability of Mortality:** {predicted_probabilities[1]:.2f}")
    st.write(f"**Decision threshold:** 0.22")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame([feature_values], columns=feature_names))

    shap_values_single = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=np.array([consciousness, ldh, mv, ast, crrt, u, l]),
        feature_names=feature_names
    )

    plt.figure()
    shap.plots.waterfall(shap_values_single, max_display=10)
    st.pyplot(plt)
    plt.close()
