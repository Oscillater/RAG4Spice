"""
文本处理工具模块

提供文本处理相关的工具函数，包括字符串处理、格式转换等功能。
"""

import re
import json
from typing import Any, Dict, List, Union


def ensure_string(value: Any) -> str:
    """
    确保值是字符串类型，处理各种可能的输入类型

    Args:
        value: 需要转换为字符串的值，可以是任意类型

    Returns:
        str: 转换后的字符串
    """
    if isinstance(value, str):
        return value.strip()
    elif isinstance(value, dict):
        # 如果是字典，尝试提取常见的文本字段
        description = value.get("description", "")
        if description:
            return str(description).strip()
        text = value.get("text", "")
        if text:
            return str(text).strip()
        # 如果没有找到常见字段，将整个字典转换为字符串
        return str(value).strip()
    elif isinstance(value, list):
        # 如果是列表，连接所有元素
        return " ".join(str(item) for item in value).strip()
    else:
        # 其他类型直接转换为字符串
        return str(value).strip()


def clean_text(text: str) -> str:
    """
    清理文本，移除多余的空白字符和特殊字符

    Args:
        text: 需要清理的文本

    Returns:
        str: 清理后的文本
    """
    if not text:
        return ""

    # 移除控制字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 规范化空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_json_from_text(text: str) -> Union[Dict, None]:
    """
    从文本中提取JSON对象

    Args:
        text: 包含JSON的文本

    Returns:
        dict or None: 解析后的JSON对象，如果解析失败返回None
    """
    try:
        # 方法1: 直接解析整个响应（如果是纯JSON）
        cleaned_text = text.strip().replace('\r\n', '\n')
        print(f"尝试方法1: 直接解析整个文本")
        try:
            result = json.loads(cleaned_text)
            print("方法1成功：直接解析JSON")
            return result
        except json.JSONDecodeError as e:
            print(f"方法1失败: {e}")

        # 方法2: 查找JSON代码块
        if "```json" in cleaned_text:
            print("尝试方法2: 查找JSON代码块")
            json_match = re.search(r'```json\s*\n(.*?)\n```', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                print(f"找到JSON代码块: {json_str[:200]}...")
                try:
                    result = json.loads(json_str)
                    print("方法2成功：解析JSON代码块")
                    return result
                except json.JSONDecodeError as e:
                    print(f"方法2失败: {e}")

        # 方法3: 查找通用代码块中的JSON
        if "```" in cleaned_text:
            print("尝试方法3: 查找通用代码块中的JSON")
            json_match = re.search(r'```\s*\n(.*?\{.*?\}.*?)\n```', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                print(f"找到通用代码块: {json_str[:200]}...")
                try:
                    result = json.loads(json_str)
                    print("方法3成功：解析通用代码块")
                    return result
                except json.JSONDecodeError as e:
                    print(f"方法3失败: {e}")

        # 方法4: 查找第一个{到最后一个}
        print("尝试方法4: 查找第一个{到最后一个}")
        json_start = cleaned_text.find('{')
        json_end = cleaned_text.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            json_str = cleaned_text[json_start:json_end]
            print(f"提取的完整JSON内容:")
            print("=" * 50)
            print(json_str)
            print("=" * 50)
            try:
                result = json.loads(json_str)
                print("方法4成功：解析JSON片段")
                return result
            except json.JSONDecodeError as e:
                print(f"方法4失败: {e}")

        print("所有方法都失败了")
        return None

    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        print(f"extract_json_from_text异常: {e}")
        return None


def safe_error_message(error: Exception, max_length: int = 200) -> str:
    """
    生成安全的错误信息，避免包含可能导致问题的字符

    Args:
        error: 异常对象
        max_length: 最大长度限制

    Returns:
        str: 安全的错误信息
    """
    error_msg = str(error)
    # 移除可能导致问题的字符
    error_msg = error_msg.replace('"', "'").replace('\n', ' ').replace('\r', ' ')
    return error_msg[:max_length] + "..." if len(error_msg) > max_length else error_msg


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    将文本分割成指定大小的块

    Args:
        text: 要分割的文本
        chunk_size: 块大小
        chunk_overlap: 重叠大小

    Returns:
        List[str]: 分割后的文本块列表
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # 计算下一个块的起始位置（考虑重叠）
        if end >= len(text):
            break
        start = end - chunk_overlap

    return chunks


def normalize_line_endings(text: str) -> str:
    """
    标准化换行符

    Args:
        text: 需要标准化的文本

    Returns:
        str: 标准化后的文本
    """
    return text.replace('\r\n', '\n').replace('\r', '\n')