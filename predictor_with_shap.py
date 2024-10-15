import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载预训练的模型
model = joblib.load('XGBoost.pkl')  # 请将 'XGBoost.pkl' 替换为你的模型文件名

# 定义标准化时使用的均值和标准差
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

# 定义分类变量的选项
cons_options = {
    0: 'No change in consciousness (0)',  # 意识没有改变
    1: 'Change in consciousness (1)'      # 意识改变
}

mv_options = {
    0: 'Not applied (0)',  # 未应用
    1: 'Applied (1)'        # 应用
}

crrt_options = {
    0: 'Not applied (0)',  # 未应用
    1: 'Applied (1)'        # 应用
}

# 定义特征名称
feature_names = ['CONS', 'LDH', 'MV', 'AST', 'CRRT', 'U', 'L']

# Streamlit 用户界面
st.title("Bunyavirus Prognosis")

# 获取用户输入
cons = st.selectbox("Consciousness Status (CONS):", options=list(cons_options.keys()), format_func=lambda x: cons_options[x])
ldh = st.number_input("Lactate Dehydrogenase (LDH, U/L):", min_value=0, max_value=5000, value=200)
mv = st.selectbox("Mechanical Ventilation (MV):", options=list(mv_options.keys()), format_func=lambda x: mv_options[x])
ast = st.number_input("Aspartate Aminotransferase (AST, U/L):", min_value=0, max_value=5000, value=30)
crrt = st.selectbox("Continuous Renal Replacement Therapy (CRRT):", options=list(crrt_options.keys()), format_func=lambda x: crrt_options[x])
u = st.number_input("Urea (mmol/L):", min_value=0.0, max_value=200.0, value=5.0)
l = st.number_input("Lymphocyte Percentage (%):", min_value=0.0, max_value=100.0, value=20.0)

# 将用户输入的变量转换为模型输入格式
# 首先对连续变量进行标准化
ldh_standardized = (ldh - scaler_means["LDH"]) / scaler_stds["LDH"]
ast_standardized = (ast - scaler_means["AST"]) / scaler_stds["AST"]
u_standardized = (u - scaler_means["U"]) / scaler_stds["U"]
l_standardized = (l - scaler_means["L"]) / scaler_stds["L"]

# 创建标准化后的特征数组
feature_values = [cons, ldh_standardized, mv, ast_standardized, crrt, u_standardized, l_standardized]
features = np.array([feature_values])

# 当用户点击“预测”按钮时执行预测
if st.button("Predict"):
    # 使用模型进行预测
    predicted_probabilities = model.predict_proba(features)[0]

    # 显示预测结果（只输出死亡概率）
    st.write(f"**Probability of Mortality:** {predicted_probabilities[1]:.2f}")
    st.write(f"**Decision threshold:** 0.22")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame([feature_values], columns=feature_names))

    # 使用 shap.plots.waterfall 创建 SHAP 瀑布图，并且显示原始值
    plt.figure()
    shap_values_original = pd.DataFrame([[cons, ldh, mv, ast, crrt, u, l]], columns=feature_names)  # 原始值
    shap.plots.waterfall(shap_values[0], feature_names=feature_names)

    # 保存图像并显示在 Streamlit 中
    st.pyplot(plt)
