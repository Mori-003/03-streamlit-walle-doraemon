import streamlit as st
import sys
import pathlib
from fastai.vision.all import *
from fastai.learner import Learner

# Python 版本检查
if sys.version_info >= (3, 13):
    st.error("⚠️ 当前 Python 版本为 3.13+，可能与 fastai 不兼容。建议使用 Python 3.11 或更低版本。")
    st.stop()

@st.cache_data
def load_model():
    """加载并缓存模型"""
    # Windows 路径兼容性处理
    temp = None
    if sys.platform == "win32":
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    
    try:
        # 使用 Learner.load 加载模型
        model_path = pathlib.Path(__file__).parent / "doraemon_walle_model.pkl"
        learn = Learner.load(model_path.stem)  # 使用 stem 来获取文件名作为 Learner 加载的名称
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        st.stop()
    finally:
        # 恢复原始设置
        if sys.platform == "win32" and temp is not None:
            pathlib.PosixPath = temp
    
    return learn

# 主应用
st.title("图像分类应用")
st.write("上传一张图片，应用将预测对应的标签。")

learn = load_model()  # 加载模型

uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PILImage.create(uploaded_file)
    st.image(image, caption="上传的图片", use_container_width=True)
    
    pred, pred_idx, probs = learn.predict(image)  # 使用 Learner 对象进行预测
    st.write(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}") 
