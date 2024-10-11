import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components

# 加载预训练的模型
model = joblib.load('XGBoost.pkl')  # 请确保模型文件名和路径正确

# 定义特征名称
selected_features = ['CONS', 'LDH', 'MV', 'AST', 'CRRT', 'U', 'L']

# 创建 SHAP TreeExplainer
explainer = shap.TreeExplainer(model)

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

# 将用户输入的变量转换为模型输入格式，并确保所有数据类型为浮点数或整数
feature_values = np.array([[cons, ldh, mv, ast, crrt, u, l]], dtype=float)
features = pd.DataFrame(feature_values, columns=selected_features)

# 当用户点击“预测”按钮时执行预测
if st.button("预测"):
    try:
        # 使用模型进行预测
        predicted_proba = model.predict_proba(features)[0]
        predicted_class = np.argmax(predicted_proba)
        predicted_probability = predicted_proba[predicted_class]
        
        # 显示预测结果和概率
        st.write(f"**预测结果:** {predicted_class}")
        st.write(f"**预测概率:** {predicted_probability:.2%}")

        # 计算 SHAP 值
        shap_values = explainer.shap_values(features)

        # 提取正类（类别1）的 SHAP 值
        shap_values_positive_class = shap_values[1]

        # 获取基准值（base value）用于正类
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value

        # 显示 SHAP 力图，解释正类的概率
        shap_plot = shap.force_plot(base_value, shap_values_positive_class[0], features.iloc[0], show=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{shap_plot.html()}</body>"

        # 在 Streamlit 中显示 SHAP 力图
        components.html(shap_html, height=600)  # 调整高度以适应图表大小
    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")
