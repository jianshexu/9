import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import streamlit.components.v1 as components

# 加载预训练的模型（确保模型是用最新的 XGBoost 格式保存的）
model = xgb.Booster()
model.load_model('model.json')  # 使用 XGBoost 格式保存并加载模型

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
feature_values = [cons, ldh, mv, ast, crrt, u, l]
features = np.array([feature_values])

# 当用户点击“预测”按钮时执行预测
if st.button("预测"):
    # 使用模型进行预测
    dmatrix = xgb.DMatrix(features, feature_names=feature_names)
    predicted_class = model.predict(dmatrix)[0]
    
    # 显示预测结果
    st.write(f"**预测结果:** {predicted_class}")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 使用 shap.plots.force 创建 SHAP 图表
    shap_plot = shap.plots.force(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names))

    # 使用 Streamlit 组件在 Streamlit 中显示 SHAP 图
    shap_html = f"<head>{shap.getjs()}</head><body>{shap_plot.html()}</body>"
    components.html(shap_html, height=500)

    # 使用 st_shap 函数在 Streamlit 中显示 SHAP 图
    st.pyplot(shap_plot, clear_figure=True)
