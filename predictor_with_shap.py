import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the pre-trained model
model = joblib.load('XGBoost.pkl')  # Please replace 'XGBoost.pkl' with your model filename

# Define means and standard deviations used in the normalization
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

# Define options for categorical variables
cons_options = {
    0: 'No change in consciousness (CONS: 0)',
    1: 'Change in consciousness (CONS: 1)'
}

mv_options = {
    0: 'Not applied (MV: 0)',
    1: 'Applied (MV: 1)'
}

crrt_options = {
    0: 'Not applied (CRRT: 0)',
    1: 'Applied (CRRT: 1)'
}

# Feature names
feature_names = ['CONS', 'LDH', 'MV', 'AST', 'CRRT', 'U', 'L']

# Streamlit UI
st.title("Bunyavirus Prognosis")

# User inputs
cons = st.selectbox("Consciousness State (CONS):", options=list(cons_options.keys()), format_func=lambda x: cons_options[x])
ldh = st.number_input("Lactate Dehydrogenase (LDH) (U/L):", min_value=0, max_value=5000, value=200)
mv = st.selectbox("Mechanical Ventilation (MV):", options=list(mv_options.keys()), format_func=lambda x: mv_options[x])
ast = st.number_input("Aspartate Aminotransferase (AST) (U/L):", min_value=0, max_value=5000, value=30)
crrt = st.selectbox("Continuous Renal Replacement Therapy (CRRT):", options=list(crrt_options.keys()), format_func=lambda x: crrt_options[x])
u = st.number_input("Urea (U) (mmol/L):", min_value=0.0, max_value=200.0, value=5.0)
l = st.number_input("Lymphocyte Percentage (L) (%):", min_value=0.0, max_value=100.0, value=20.0)

# Standardize continuous variables
ldh_standardized = (ldh - scaler_means["LDH"]) / scaler_stds["LDH"]
ast_standardized = (ast - scaler_means["AST"]) / scaler_stds["AST"]
u_standardized = (u - scaler_means["U"]) / scaler_stds["U"]
l_standardized = (l - scaler_means["L"]) / scaler_stds["L"]

# Create standardized feature array
feature_values = [cons, ldh_standardized, mv, ast_standardized, crrt, u_standardized, l_standardized]
features = np.array([feature_values])

# Predict button
if st.button("Predict"):
    # Get model predictions
    predicted_probabilities = model.predict_proba(features)[0]
    predicted_class = model.predict(features)[0]
    
    # Convert prediction result to 'Survival' or 'Mortality'
    predicted_class_str = 'Survival' if predicted_class == 0 else 'Mortality'

    # Show prediction probabilities
    st.write(f"**Prediction: {predicted_class_str}**")
    st.write(f"**Probability of Mortality:** {predicted_probabilities[1]:.2f}")
    st.write(f"**Probability of Survival:** {predicted_probabilities[0]:.2f}")
    st.write(f"**Decision threshold: 0.22**")

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame([feature_values], columns=feature_names))

    # Plot SHAP waterfall chart
    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=10)

    # Show original input values in the SHAP plot (not standardized values)
    st.write("SHAP values based on original inputs:")
    st.write(f"**LDH:** {ldh} U/L, **AST:** {ast} U/L, **Urea:** {u} mmol/L, **Lymphocyte Percentage:** {l}%")
    
    # Save and display the plot in Streamlit
    st.pyplot(plt)
