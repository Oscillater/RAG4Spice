"""
PDF文本提取模块

提供从PDF文件中提取文本的功能。
"""

import io
import streamlit as st
from typing import Optional

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


def extract_text_from_pdf(uploaded_file) -> str:
    """
    从上传的PDF文件中提取文本

    Args:
        uploaded_file: Streamlit上传的文件对象

    Returns:
        str: 提取的文本内容

    Raises:
        RuntimeError: PDF文本提取失败
    """
    if not uploaded_file:
        raise ValueError("上传的PDF文件为空")

    try:
        # 读取文件内容
        pdf_bytes = uploaded_file.read()

        # 尝试使用PyPDF2
        if PYPDF2_AVAILABLE:
            return _extract_with_pypdf2(pdf_bytes)

        # 尝试使用pdfplumber
        elif PDFPLUMBER_AVAILABLE:
            return _extract_with_pdfplumber(pdf_bytes)

        else:
            raise RuntimeError(
                "PDF解析库不可用。请安装PyPDF2或pdfplumber：\n"
                "pip install PyPDF2\n"
                "或\n"
                "pip install pdfplumber"
            )

    except Exception as e:
        raise RuntimeError(f"PDF文本提取失败: {str(e)}")


def _extract_with_pypdf2(pdf_bytes: bytes) -> str:
    """使用PyPDF2提取文本"""
    import PyPDF2

    pdf_stream = io.BytesIO(pdf_bytes)
    reader = PyPDF2.PdfReader(pdf_stream)

    text_content = []

    for page_num, page in enumerate(reader.pages, 1):
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_content.append(f"--- 第{page_num}页 ---\n{page_text.strip()}\n")
        except Exception as e:
            st.warning(f"第{page_num}页提取失败: {str(e)}")
            continue

    return "\n".join(text_content)


def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    """使用pdfplumber提取文本"""
    import pdfplumber

    text_content = []

    with io.BytesIO(pdf_bytes) as pdf_stream:
        try:
            with pdfplumber.open(pdf_stream) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_content.append(f"--- 第{page_num}页 ---\n{page_text.strip()}\n")
                    except Exception as e:
                        st.warning(f"第{page_num}页提取失败: {str(e)}")
                        continue
        except Exception as e:
            st.warning(f"pdfplumber处理失败，尝试备用方法: {str(e)}")
            # 如果pdfplumber失败，尝试简单的文本提取
            return _fallback_text_extraction(pdf_bytes)

    return "\n".join(text_content)


def _fallback_text_extraction(pdf_bytes: bytes) -> str:
    """备用文本提取方法"""
    try:
        # 尝试简单的字符串提取（适用于简单的文本PDF）
        text = pdf_bytes.decode('utf-8', errors='ignore')
        if text and len(text) > 50:  # 至少有一些内容
            return f"--- PDF文本内容 ---\n{text}"
        else:
            return ""
    except Exception:
        return ""


def get_pdf_info() -> dict:
    """
    获取PDF解析库的可用信息

    Returns:
        dict: 库的可用状态
    """
    return {
        "pypdf2": PYPDF2_AVAILABLE,
        "pdfplumber": PDFPLUMBER_AVAILABLE,
        "recommendation": "PyPDF2" if PYPDF2_AVAILABLE else "pdfplumber" if PDFPLUMBER_AVAILABLE else "none"
    }