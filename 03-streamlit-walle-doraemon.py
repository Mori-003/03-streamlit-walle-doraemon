import streamlit as st
import sys
import os

# Python 版本检查
if sys.version_info >= (3, 13):
    st.error("⚠️ 当前 Python 版本为 3.13+，可能与 fastai 不兼容。建议使用 Python 3.11。")
    st.stop()

# 设置环境变量，禁用文件监视
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# 针对torch._classes.__path__._path错误的修复
try:
    import torch
    if hasattr(torch, '_classes'):
        class FakePath:
            _path = []
        if not hasattr(torch._classes, '__path__'):
            torch._classes.__path__ = FakePath()
        elif not hasattr(torch._classes.__path__, '_path'):
            torch._classes.__path__._path = []
except ImportError:
    pass

# 禁用不必要的警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 导入pickle模块用于处理UnpicklingError
import pickle

# 添加fastai模块导入
import fastai
from fastai.vision.all import *
import pathlib

@st.cache_resource
def load_model():
    """加载并缓存模型"""
    # Windows 路径兼容性处理
    original_posix_path = None
    if sys.platform == "win32":
        original_posix_path = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    
    try:
        model_path = pathlib.Path(__file__).parent / "doraemon_walle_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        # 使用try-except捕获load_learner可能的错误
        try:
            # 添加详细日志
            st.write(f"尝试加载模型: {model_path}")
            st.write(f"当前 fastai 版本: {fastai.__version__}")
            st.write(f"当前 torch 版本: {torch.__version__}")
            
            model = load_learner(model_path)
            st.write("模型加载成功!")
            return model
        except RuntimeError as e:
            if "__path__._path" in str(e):
                st.error("加载模型时出现torch路径错误。这是一个已知问题，请尝试重新启动应用。")
                st.stop()
            else:
                st.error(f"RuntimeError: {str(e)}")
                raise
        except pickle.UnpicklingError as e:
            st.error(f"Pickle错误: {str(e)}。这可能是由于模型文件与当前环境不兼容。")
            st.info("尝试使用与模型创建时相同的fastai和torch版本。")
            raise
        except Exception as e:
            st.error(f"未知错误: {str(e)}")
            raise
    finally:
        # 恢复原始设置
        if sys.platform == "win32" and original_posix_path is not None:
            pathlib.PosixPath = original_posix_path

# 主应用
st.title("哆啦A梦与WALL-E图像分类应用")
st.write("上传一张图片，应用将预测是哆啦A梦还是WALL-E。")

try:
    with st.spinner("正在加载模型..."):
        model = load_model()
except FileNotFoundError as e:
    st.error(f"错误: {str(e)}")
    st.info("请确保模型文件存在于项目目录中。")
    st.stop()
except Exception as e:
    st.error(f"加载模型时出错: {str(e)}")
    st.stop()

uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        with st.spinner('处理图片中...'):
            image = PILImage.create(uploaded_file)
            st.image(image, caption="上传的图片", use_container_width=True)
            
            pred, pred_idx, probs = model.predict(image)
            
            # 使用进度条显示概率
            st.write(f"预测结果: {pred}")
            st.progress(float(probs[pred_idx]))
            st.write(f"概率: {probs[pred_idx]:.04f}")
    except Exception as e:
        st.error(f"处理图片时出错: {str(e)}")
