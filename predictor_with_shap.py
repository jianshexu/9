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
    0: '意识没有改变 (0)',  # No change in consciousness
    1: '意识改变 (1)'       # Change in consciousness
}

mv_options = {
    0: '未应用 (0)',  # Not applied
    1: '应用 (1)'     # Applied
}

crrt_options = {
    0: '未应用 (0)',  # Not applied
    1: '应用 (1)'     # Applied
}

# 定义特征名称
feature_names = ['CONS', 'LDH', 'MV', 'AST', 'CRRT', 'U', 'L']

# Streamlit 用户界面
st.title("布尼亚预后")

# 获取用户输入
cons = st.selectbox("意识状态 (CONS):", options=list(cons_options.keys()), format_func=lambda x: cons_options[x])
ldh = st.number_input("乳酸脱氢酶 (LDH):", min_value=0, max_value=5000, value=200)
mv = st.selectbox("机械通气 (MV):", options=list(mv_options.keys()), format_func=lambda x: mv_options[x])
ast = st.number_input("天冬氨酸转氨酶 (AST):", min_value=0, max_value=5000, value=30)
crrt = st.selectbox("持续性肾脏替代治疗 (CRRT):", options=list(crrt_options.keys()), format_func=lambda x: crrt_options[x])
u = st.number_input("尿素 (U):", min_value=0.0, max_value=200.0, value=5.0)
l = st.number_input("淋巴细胞百分比 (L):", min_value=0.0, max_value=100.0, value=20.0)

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
if st.button("预测"):
    # 使用模型进行预测
    predicted_probabilities = model.predict_proba(features)[0]
    predicted_class = model.predict(features)[0]

    # 显示预测结果（输出概率）
    st.write(f"**预测类别:** {predicted_class}")
    st.write(f"**预测概率 (阳性):** {predicted_probabilities[1]:.2f}")
    st.write(f"**预测概率 (阴性):** {predicted_probabilities[0]:.2f}")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame([feature_values], columns=feature_names))

    # 使用 shap.plots.waterfall 创建 SHAP 瀑布图
    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=10)
    
    # 保存图像并显示在 Streamlit 中
    st.pyplot(plt)
