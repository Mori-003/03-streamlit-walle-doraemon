import streamlit as st
import sys
import pathlib
import os
import torch
import asyncio  # 导入 asyncio 模块
from fastai.vision.all import *

# 修复 PyTorch 类路径问题
torch.classes.__path__ = []  # 防止访问 PyTorch 内部路径

# 确保事件循环在主线程中创建
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())  # 创建一个新的事件循环

# 使用 nest_asyncio 解决异步事件循环问题
import nest_asyncio
nest_asyncio.apply()  # 允许嵌套事件循环

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
        model_path = pathlib.Path(__file__).parent / "doraemon_walle_model.pkl"
        model = load_learner(model_path)
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        st.stop()
    finally:
        # 恢复原始设置
        if sys.platform == "win32" and temp is not None:
            pathlib.PosixPath = temp
    
    return model

# 主应用
st.title("图像分类应用")
st.write("上传一张图片，应用将预测对应的标签。")

model = load_model()

uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PILImage.create(uploaded_file)
    st.image(image, caption="上传的图片", use_container_width=True)
    
    pred, pred_idx, probs = model.predict(image)
    st.write(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}")
