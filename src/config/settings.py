"""
配置管理模块

统一管理所有配置项，包括数据库路径、模型设置、API配置等。
使用环境变量和默认值相结合的方式，提高配置的灵活性。
"""

import os
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Settings:
    """应用配置类"""

    # 数据库配置
    PERSIST_DIRECTORY: str = "hspice_db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # PDF文档配置
    PDF_PATH: str = "resource/hspice_manual.pdf"

    # 文本分割配置
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # 检索配置
    RETRIEVAL_K: int = 3

    # OCR配置
    TESSERACT_CMD: Optional[str] = os.getenv("TESSERACT_CMD", "tesseract")

    # Google API配置
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

    # LLM模型配置
    LLM_PRO_MODEL: str = "gemini-2.5-flash"
    LLM_FLASH_MODEL: str = "gemini-2.5-flash"

    # 应用配置
    APP_TITLE: str = "🤖 HSPICE RAG 代码生成助手"
    APP_CAPTION: str = "上传实验截图，分析任务，生成HSPICE代码"

    # 文件上传配置
    ALLOWED_IMAGE_TYPES: list = ["png", "jpg", "jpeg"]
    ALLOWED_FILE_TYPES: list = ["png", "jpg", "jpeg", "pdf"]

    # 重试配置
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 2  # 秒

    # 超时配置
    API_TIMEOUT: int = 600  # 秒
    TASK_ANALYSIS_TIMEOUT: int = 30000  # 毫秒

    def validate(self) -> bool:
        """
        验证必要的配置项是否设置正确 (实例方法)

        Returns:
            bool: 配置是否有效
        """
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY 环境变量未设置")

        if not self.TESSERACT_CMD:
            raise ValueError("TESSERACT_CMD 环境变量未设置")

        return True

    def get_pdf_path(self) -> str:
        """
        获取PDF文件路径，确保文件存在 (实例方法)

        Returns:
            str: PDF文件路径

        Raises:
            FileNotFoundError: PDF文件不存在
        """
        if not os.path.exists(self.PDF_PATH):
            raise FileNotFoundError(f"PDF文件不存在: {self.PDF_PATH}")
        return self.PDF_PATH

    def ensure_directory(self, directory: str) -> str:
        """
        确保目录存在，如果不存在则创建 (实例方法)

        Args:
            directory: 目录路径

        Returns:
            str: 目录路径
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return directory


# 创建全局配置实例
settings = Settings()


# 常用配置的快捷访问函数
def get_embedding_model() -> str:
    """获取嵌入模型名称"""
    return settings.EMBEDDING_MODEL


def get_persist_directory() -> str:
    """获取数据库持久化目录"""
    return settings.ensure_directory(settings.PERSIST_DIRECTORY)


def get_google_api_key() -> str:
    """获取Google API密钥"""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("Google API密钥未配置")
    return settings.GOOGLE_API_KEY


def get_tesseract_cmd() -> str:
    """获取Tesseract命令路径"""
    if not settings.TESSERACT_CMD:
        raise ValueError("Tesseract命令未配置")
    return settings.TESSERACT_CMD