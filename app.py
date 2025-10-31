#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG4Spice 主应用

重构后的主应用文件，采用模块化架构，分离UI逻辑和业务逻辑。
使用方法:
    streamlit run app.py
"""

import sys
import streamlit as st
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import settings
from ui.pages import render_main_page
from utils.validators import ValidationError


def main():
    """主函数"""
    try:
        # 配置Streamlit页面
        _configure_streamlit_page()

        # 初始化应用
        _initialize_app()

        # 渲染主页面
        render_main_page()

    except ValidationError as e:
        st.error(f"⚠️ 配置验证失败: {str(e)}")
        st.stop()

    except Exception as e:
        st.error(f"❌ 应用启动失败: {str(e)}")
        st.stop()


def _configure_streamlit_page():
    """配置Streamlit页面设置"""
    st.set_page_config(
        page_title=settings.APP_TITLE.replace("🤖 ", "").replace(" ", "_"),
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def _initialize_app():
    """初始化应用"""
    try:
        # 获取验证状态
        validation_status = settings.get_validation_status()

        # 基本验证，但不在streamlit上下文中显示警告
        # 警告将在UI层面处理

        # 基本检查：只检查Tesseract，API密钥可以通过网页配置
        if validation_status["tesseract"]:
            print("✅ 应用初始化完成")
        else:
            print("⚠️ Tesseract OCR未配置，图片识别功能将不可用")

        # 可选：在这里添加全局初始化逻辑
        # 例如：检查数据库、初始化模型等

    except Exception as e:
        raise ValidationError(f"应用初始化失败: {str(e)}")


if __name__ == "__main__":
    main()