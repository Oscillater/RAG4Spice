"""
PDF解析模块

提供PDF文件文本提取功能，支持处理多页PDF文档。
"""

import io
import PyPDF2
from typing import List, Optional, Union
from pathlib import Path

from config.settings import settings


class PDFParser:
    """PDF解析器"""

    @staticmethod
    def extract_text_from_file(file_path: Union[str, Path]) -> str:
        """
        从PDF文件路径提取文本

        Args:
            file_path: PDF文件路径

        Returns:
            str: 提取的文本

        Raises:
            FileNotFoundError: 文件不存在
            RuntimeError: PDF解析失败
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {file_path}")

        try:
            with open(file_path, 'rb') as file:
                return PDFParser.extract_text_from_file_obj(file)
        except Exception as e:
            raise RuntimeError(f"PDF文件读取失败: {str(e)}")

    @staticmethod
    def extract_text_from_file_obj(file_obj: io.BytesIO) -> str:
        """
        从文件对象提取文本

        Args:
            file_obj: PDF文件对象

        Returns:
            str: 提取的文本

        Raises:
            RuntimeError: PDF解析失败
        """
        try:
            pdf_reader = PyPDF2.PdfReader(file_obj)
            text_parts = []

            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(f"--- Page {page_num} ---\n{page_text}")
                except Exception as e:
                    print(f"警告: 第{page_num}页解析失败: {str(e)}")
                    continue

            return "\n\n".join(text_parts).strip()

        except Exception as e:
            raise RuntimeError(f"PDF解析失败: {str(e)}")

    @staticmethod
    def extract_text_from_uploaded_file(uploaded_file) -> str:
        """
        从Streamlit上传的文件对象提取文本

        Args:
            uploaded_file: Streamlit上传的文件对象

        Returns:
            str: 提取的文本

        Raises:
            RuntimeError: PDF解析失败
        """
        try:
            # 重置文件指针
            uploaded_file.seek(0)
            return PDFParser.extract_text_from_file_obj(uploaded_file)
        except Exception as e:
            raise RuntimeError(f"PDF文本提取失败: {str(e)}")

    @staticmethod
    def get_pdf_info(file_path: Union[str, Path]) -> dict:
        """
        获取PDF文件信息

        Args:
            file_path: PDF文件路径

        Returns:
            dict: PDF文件信息
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {file_path}")

        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                metadata = pdf_reader.metadata or {}

                return {
                    'page_count': len(pdf_reader.pages),
                    'title': metadata.get('/Title', ''),
                    'author': metadata.get('/Author', ''),
                    'subject': metadata.get('/Subject', ''),
                    'creator': metadata.get('/Creator', ''),
                    'producer': metadata.get('/Producer', ''),
                    'creation_date': metadata.get('/CreationDate', ''),
                    'modification_date': metadata.get('/ModDate', ''),
                }
        except Exception as e:
            raise RuntimeError(f"PDF信息获取失败: {str(e)}")

    @staticmethod
    def extract_pages_text(file_path: Union[str, Path], pages: List[int]) -> str:
        """
        提取指定页面的文本

        Args:
            file_path: PDF文件路径
            pages: 页码列表（从1开始）

        Returns:
            str: 提取的文本
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {file_path}")

        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []

                for page_num in pages:
                    if 1 <= page_num <= len(pdf_reader.pages):
                        page = pdf_reader.pages[page_num - 1]
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(f"--- Page {page_num} ---\n{page_text}")
                    else:
                        print(f"警告: 页码{page_num}超出范围")

                return "\n\n".join(text_parts).strip()

        except Exception as e:
            raise RuntimeError(f"指定页面文本提取失败: {str(e)}")


# 创建全局PDF解析器实例
pdf_parser = PDFParser()


def extract_text_from_pdf(file_obj: io.BytesIO) -> str:
    """
    便捷函数：从PDF文件对象提取文本

    Args:
        file_obj: PDF文件对象

    Returns:
        str: 提取的文本
    """
    return pdf_parser.extract_text_from_file_obj(file_obj)