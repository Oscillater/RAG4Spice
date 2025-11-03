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

    # 多模型配置
    DEFAULT_MODEL: str = "gemini-2.5-flash"  # 默认选择的模型（用户可修改）
    ENABLE_MULTI_MODEL: bool = True  # 是否启用多模型功能

    # LLM模型配置 (保持向后兼容)
    LLM_PRO_MODEL: str = "gemini-2.5-flash"
    LLM_FLASH_MODEL: str = "gemini-2.5-flash"

    # 向后兼容：保留Google API配置（仅用于导入现有配置）
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

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
        不再直接抛出错误，而是返回验证结果

        Returns:
            bool: 配置是否有效
        """
        # 基本验证 - 只检查Tesseract，API密钥不再是必需的
        # 用户可以通过网页界面配置任何模型
        return bool(self.TESSERACT_CMD)

    def _check_any_api_key_available(self) -> bool:
        """检查是否有任何可用的API密钥"""
        try:
            from .models import model_config
            available_models = model_config.get_all_models()

            for model in available_models:
                env_key = model.get_env_key()
                if env_key and os.getenv(env_key):
                    return True
            return False
        except Exception:
            return False

    def get_validation_status(self) -> dict:
        """
        获取详细的验证状态信息

        Returns:
            dict: 验证状态信息
        """
        status = {
            "tesseract": bool(self.TESSERACT_CMD),
            "api_keys": [],
            "multi_model_enabled": self.ENABLE_MULTI_MODEL,
            "has_any_api_key": False,
            "recommendations": []
        }

        # 检查API密钥
        try:
            from .models import model_config
            available_models = model_config.get_all_models()

            for model in available_models:
                env_key = model.get_env_key()
                has_key = env_key and os.getenv(env_key)

                status["api_keys"].append({
                    "model_id": model.model_id,
                    "display_name": model.display_name,
                    "provider": model.provider.value,
                    "env_key": env_key,
                    "has_key": bool(has_key)
                })

                if has_key:
                    status["has_any_api_key"] = True

        except Exception as e:
            status["api_keys_error"] = str(e)

        # 生成建议
        if not status["tesseract"]:
            status["recommendations"].append("请安装Tesseract OCR并设置TESSERACT_CMD环境变量")

        if not status["has_any_api_key"]:
            status["recommendations"].append("请至少配置一个AI模型的API密钥")
            status["recommendations"].append("建议使用Google Gemini 2.5 Flash作为入门模型")

        return status

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


def get_google_api_key() -> Optional[str]:
    """获取Google API密钥（向后兼容函数）"""
    return settings.GOOGLE_API_KEY


def get_tesseract_cmd() -> Optional[str]:
    """获取Tesseract命令路径"""
    return settings.TESSERACT_CMD