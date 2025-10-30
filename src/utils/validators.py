"""
数据验证模块

提供各种数据验证功能，确保输入数据的正确性和安全性。
"""

import os
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class ValidationError(Exception):
    """验证错误异常"""
    pass


class FileValidator:
    """文件验证器"""

    @staticmethod
    def validate_file_exists(file_path: Union[str, Path]) -> Path:
        """
        验证文件是否存在

        Args:
            file_path: 文件路径

        Returns:
            Path: 验证后的Path对象

        Raises:
            ValidationError: 文件不存在
        """
        path = Path(file_path)
        if not path.exists():
            raise ValidationError(f"文件不存在: {file_path}")
        return path

    @staticmethod
    def validate_file_extension(
        file_path: Union[str, Path],
        allowed_extensions: List[str]
    ) -> Path:
        """
        验证文件扩展名

        Args:
            file_path: 文件路径
            allowed_extensions: 允许的扩展名列表

        Returns:
            Path: 验证后的Path对象

        Raises:
            ValidationError: 文件扩展名不被允许
        """
        path = Path(file_path)
        extension = path.suffix.lower().lstrip('.')

        if extension not in [ext.lower() for ext in allowed_extensions]:
            raise ValidationError(
                f"文件扩展名 '{extension}' 不被允许。"
                f"允许的扩展名: {', '.join(allowed_extensions)}"
            )
        return path

    @staticmethod
    def validate_file_size(file_path: Union[str, Path], max_size_mb: int = 50) -> Path:
        """
        验证文件大小

        Args:
            file_path: 文件路径
            max_size_mb: 最大文件大小（MB）

        Returns:
            Path: 验证后的Path对象

        Raises:
            ValidationError: 文件过大
        """
        path = Path(file_path)
        if not path.exists():
            raise ValidationError(f"文件不存在: {file_path}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValidationError(
                f"文件过大: {size_mb:.2f}MB (最大允许: {max_size_mb}MB)"
            )
        return path


class TextValidator:
    """文本验证器"""

    @staticmethod
    def validate_non_empty(text: str, field_name: str = "文本") -> str:
        """
        验证文本非空

        Args:
            text: 要验证的文本
            field_name: 字段名称

        Returns:
            str: 验证后的文本

        Raises:
            ValidationError: 文本为空
        """
        if not text or not text.strip():
            raise ValidationError(f"{field_name}不能为空")
        return text.strip()

    @staticmethod
    def validate_max_length(
        text: str,
        max_length: int,
        field_name: str = "文本"
    ) -> str:
        """
        验证文本最大长度

        Args:
            text: 要验证的文本
            max_length: 最大长度
            field_name: 字段名称

        Returns:
            str: 验证后的文本

        Raises:
            ValidationError: 文本过长
        """
        if len(text) > max_length:
            raise ValidationError(
                f"{field_name}过长: {len(text)} 字符 (最大允许: {max_length} 字符)"
            )
        return text

    @staticmethod
    def validate_min_length(
        text: str,
        min_length: int,
        field_name: str = "文本"
    ) -> str:
        """
        验证文本最小长度

        Args:
            text: 要验证的文本
            min_length: 最小长度
            field_name: 字段名称

        Returns:
            str: 验证后的文本

        Raises:
            ValidationError: 文本过短
        """
        if len(text) < min_length:
            raise ValidationError(
                f"{field_name}过短: {len(text)} 字符 (最小要求: {min_length} 字符)"
            )
        return text


class TaskValidator:
    """任务数据验证器"""

    @staticmethod
    def validate_task_structure(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证任务数据结构

        Args:
            task_data: 任务数据

        Returns:
            Dict: 验证后的任务数据

        Raises:
            ValidationError: 任务结构无效
        """
        if not isinstance(task_data, dict):
            raise ValidationError("任务数据必须是字典类型")

        # 验证必需字段
        required_fields = ['general_description', 'tasks']
        for field in required_fields:
            if field not in task_data:
                raise ValidationError(f"缺少必需字段: {field}")

        # 验证general_description
        if not isinstance(task_data['general_description'], str):
            raise ValidationError("general_description必须是字符串类型")

        # 验证tasks
        if not isinstance(task_data['tasks'], list):
            raise ValidationError("tasks必须是列表类型")

        # 验证每个task
        for i, task in enumerate(task_data['tasks']):
            TaskValidator.validate_single_task(task, i)

        return task_data

    @staticmethod
    def validate_single_task(task: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        """
        验证单个任务

        Args:
            task: 任务数据
            index: 任务索引

        Returns:
            Dict: 验证后的任务数据

        Raises:
            ValidationError: 任务结构无效
        """
        if not isinstance(task, dict):
            raise ValidationError(f"任务{index + 1}必须是字典类型")

        # 验证必需字段
        required_fields = ['id', 'title', 'description']
        for field in required_fields:
            if field not in task:
                raise ValidationError(f"任务{index + 1}缺少必需字段: {field}")

        # 验证id
        if not isinstance(task['id'], int) or task['id'] <= 0:
            raise ValidationError(f"任务{index + 1}的id必须是正整数")

        # 验证title
        if not isinstance(task['title'], str) or not task['title'].strip():
            raise ValidationError(f"任务{index + 1}的title不能为空")

        # 验证description
        if not isinstance(task['description'], str) or not task['description'].strip():
            raise ValidationError(f"任务{index + 1}的description不能为空")

        # 验证文件名格式
        if not task['title'].endswith('.sp'):
            raise ValidationError(f"任务{index + 1}的title必须以.sp结尾")

        return task


class ConfigValidator:
    """配置验证器"""

    @staticmethod
    def validate_api_key(api_key: str, service_name: str = "API") -> str:
        """
        验证API密钥格式

        Args:
            api_key: API密钥
            service_name: 服务名称

        Returns:
            str: 验证后的API密钥

        Raises:
            ValidationError: API密钥格式无效
        """
        if not api_key or not api_key.strip():
            raise ValidationError(f"{service_name}密钥不能为空")

        api_key = api_key.strip()

        # 基本长度检查
        if len(api_key) < 10:
            raise ValidationError(f"{service_name}密钥长度过短")

        return api_key

    @staticmethod
    def validate_model_name(model_name: str, allowed_models: List[str]) -> str:
        """
        验证模型名称

        Args:
            model_name: 模型名称
            allowed_models: 允许的模型列表

        Returns:
            str: 验证后的模型名称

        Raises:
            ValidationError: 模型名称不被允许
        """
        if not model_name or not model_name.strip():
            raise ValidationError("模型名称不能为空")

        model_name = model_name.strip()

        if model_name not in allowed_models:
            raise ValidationError(
                f"不支持的模型: {model_name}。"
                f"支持的模型: {', '.join(allowed_models)}"
            )

        return model_name


def validate_email(email: str) -> str:
    """
    验证邮箱格式

    Args:
        email: 邮箱地址

    Returns:
        str: 验证后的邮箱地址

    Raises:
        ValidationError: 邮箱格式无效
    """
    if not email or not email.strip():
        raise ValidationError("邮箱地址不能为空")

    email = email.strip()
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if not re.match(pattern, email):
        raise ValidationError("邮箱格式无效")

    return email


def validate_url(url: str) -> str:
    """
    验证URL格式

    Args:
        url: URL地址

    Returns:
        str: 验证后的URL地址

    Raises:
        ValidationError: URL格式无效
    """
    if not url or not url.strip():
        raise ValidationError("URL地址不能为空")

    url = url.strip()
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'

    if not re.match(pattern, url):
        raise ValidationError("URL格式无效")

    return url