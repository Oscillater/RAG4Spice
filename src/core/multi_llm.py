"""
统一LLM客户端

完全基于HTTP API的统一客户端，支持所有AI模型。
官方模型自动配置URL，用户只需输入API密钥。
自定义模型用户可配置URL和API密钥。
"""

import requests
import time
from typing import Dict, Any, Optional
from config.models import model_config
from config.settings import settings
from config.custom_api import custom_api_manager, CustomAPIConfig
from utils.text_processing import extract_json_from_text, safe_error_message, normalize_line_endings


def get_api_key_from_session(model_id: str) -> Optional[str]:
    """
    从session state获取API密钥

    Args:
        model_id: 模型ID

    Returns:
        str: API密钥，如果未找到则返回None
    """
    try:
        import streamlit as st
        # 直接查找模型ID
        if 'api_keys' in st.session_state and model_id in st.session_state.api_keys:
            return st.session_state.api_keys[model_id]
        return None
    except Exception:
        return None


class UnifiedLLMClient:
    """统一的LLM客户端"""

    def __init__(self, model_id: str, api_key: str, base_url: str = None):
        """
        初始化统一客户端

        Args:
            model_id: 模型ID
            api_key: API密钥
            base_url: API基础URL
        """
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url

        # 如果没有提供base_url，尝试从模型配置获取
        if not self.base_url:
            model = model_config.get_model_by_id(model_id)
            if model:
                # 获取自动配置的URL
                auto_config = model_config.get_auto_config_for_model(model_id)
                self.base_url = auto_config.get('base_url') if auto_config else None

        if not self.base_url:
            raise ValueError(f"无法确定模型 {model_id} 的API地址")

    def generate_content(self, prompt: str, **kwargs) -> Any:
        """
        生成内容 - 统一使用OpenAI兼容格式

        Args:
            prompt: 提示词
            **kwargs: 其他参数

        Returns:
            Any: API响应
        """
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 构建请求数据
        data = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', 4096),
            "temperature": kwargs.get('temperature', 0.7)
        }

        # 发送请求
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=data,
            timeout=kwargs.get('timeout', settings.API_TIMEOUT)
        )
        response.raise_for_status()
        return response.json()

    def parse_response(self, response: Any) -> str:
        """
        解析响应 - 统一解析OpenAI格式

        Args:
            response: API响应

        Returns:
            str: 解析后的文本内容
        """
        if isinstance(response, dict) and "choices" in response:
            choices = response["choices"]
            if choices and "message" in choices[0]:
                return choices[0]["message"]["content"]
        return ""


class MultiLLMManager:
    """多模型LLM管理器 - 简化版"""

    def __init__(self):
        """初始化多模型管理器"""
        self.clients = {}

    def get_client(self, model_id: str, api_key: str = None) -> UnifiedLLMClient:
        """
        获取模型客户端

        Args:
            model_id: 模型ID
            api_key: API密钥（可选，如果未提供则从session state获取）

        Returns:
            UnifiedLLMClient: 统一的客户端实例
        """
        # 如果没有提供api_key，尝试从session state获取
        if not api_key:
            api_key = get_api_key_from_session(model_id)

        if not api_key:
            raise ValueError(f"未找到模型 {model_id} 的API密钥，请先在界面中配置")

        cache_key = f"{model_id}:{api_key[:8]}"

        if cache_key not in self.clients:
            # 检查是否为自定义API模型
            if model_id.startswith("custom:"):
                client = self._create_custom_client(model_id, api_key)
            else:
                client = self._create_official_client(model_id, api_key)

            self.clients[cache_key] = client

        return self.clients[cache_key]

    def _create_official_client(self, model_id: str, api_key: str) -> UnifiedLLMClient:
        """
        创建官方模型客户端

        Args:
            model_id: 模型ID
            api_key: API密钥

        Returns:
            UnifiedLLMClient: 客户端实例
        """
        # 获取官方模型的自动配置
        auto_config = model_config.get_auto_config_for_model(model_id)
        if not auto_config:
            raise ValueError(f"不支持的官方模型: {model_id}")

        base_url = auto_config.get('base_url')
        if not base_url:
            raise ValueError(f"官方模型 {model_id} 未配置API地址")

        return UnifiedLLMClient(model_id, api_key, base_url)

    def _create_custom_client(self, model_id: str, api_key: str) -> UnifiedLLMClient:
        """
        创建自定义API客户端

        Args:
            model_id: 模型ID (格式: custom:provider_name:model_name)
            api_key: API密钥

        Returns:
            UnifiedLLMClient: 客户端实例
        """
        # 解析模型ID: custom:provider_name:model_name
        parts = model_id.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"无效的自定义模型ID格式: {model_id}")

        _, provider_name, model_name = parts

        # 获取自定义API配置
        custom_config = custom_api_manager.get_config_by_name(provider_name)
        if not custom_config:
            raise ValueError(f"未找到自定义API配置: {provider_name}")

        if not custom_config.is_active:
            raise ValueError(f"自定义API配置已禁用: {provider_name}")

        # 验证模型是否在支持列表中
        if model_name not in custom_config.models:
            raise ValueError(f"模型 {model_name} 不在 {provider_name} 的支持列表中")

        return UnifiedLLMClient(model_name, api_key, custom_config.base_url)

    def generate_with_retry(
        self,
        model_id: str,
        api_key: str = None,
        prompt: str = "",
        timeout: int = None,
        max_retries: int = None,
        **kwargs
    ) -> str:
        """
        带重试机制的生成

        Args:
            model_id: 模型ID
            api_key: API密钥（可选，如果未提供则从session state获取）
            prompt: 提示词
            timeout: 超时时间
            max_retries: 最大重试次数
            **kwargs: 其他参数

        Returns:
            str: 生成结果

        Raises:
            RuntimeError: 生成失败
        """
        if timeout is None:
            timeout = settings.API_TIMEOUT
        if max_retries is None:
            max_retries = settings.MAX_RETRIES

        retry_delay = settings.RETRY_DELAY
        client = self.get_client(model_id, api_key)

        for attempt in range(max_retries):
            try:
                print(f"尝试第 {attempt + 1}/{max_retries} 次调用API ({model_id})...")
                response = client.generate_content(prompt, timeout=timeout, **kwargs)
                result = client.parse_response(response)

                if not result:
                    raise ValueError("模型返回空响应")

                print("API调用成功")
                return result

            except Exception as e:
                print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")

                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    raise RuntimeError(f"API调用失败，已重试{max_retries}次: {str(e)}")

    def analyze_tasks(self, model_id: str, api_key: str = None, ocr_text: str = "") -> Dict[str, Any]:
        """
        分析任务，将实验要求分解为多个HSPICE文件

        Args:
            model_id: 模型ID
            api_key: API密钥（可选，如果未提供则从session state获取）
            ocr_text: OCR提取的文本

        Returns:
            Dict: 任务分析结果
        """
        if not ocr_text or not ocr_text.strip():
            raise ValueError("OCR文本不能为空")

        # 任务分析提示词
        task_analysis_prompt = f"""
你是一位HSPICE仿真专家，请分析以下实验要求，按照两层架构进行任务分解，并以JSON格式输出。

**重要指令：除非实验极其简单且只有单一分析类型，否则请优先生成多个task（多个HSPICE文件）！**

重要说明：
- 用户将手动提供电路图中LLM无法从视觉上提取的所有信息
- 包括但不限于：MOS管源漏栅极位置、元件连接关系、节点标注、信号流向等
- 因为OCR/AI无法准确识别电路图中的视觉连接细节
- 你的任务是基于文字描述进行合理的任务分解

**两层任务分解架构：**

1. **实验总体描述** (general_description): 包含实验总体概述和所有HSPICE文件共用的环境配置
   - 实验目的和总体要求
   - 公共的仿真环境配置（温度、电源电压、模型文件、全局参数等）
   - 所有HSPICE文件共享的条件和设置

2. **具体任务** (tasks): 按HSPICE文件数量进行任务分组
   - 每个task对应一个独立的HSPICE仿真文件
   - 每个task包含该文件的完整功能描述和测试内容
   - **强烈建议生成多个task而不是1个task**，除非实验确实极其简单
   - 优先考虑将复杂实验分解为多个独立的HSPICE文件
   - 如果需要多个独立文件，则按功能或测试类型分成多个task，尽可能生成多个task而不是1个task

**任务分组原则（基于HSPICE文件）：**

1. **多文件优先原则**
   - **默认情况下应该生成多个task**，每个task对应一个独立的HSPICE文件
   - 即使实验相对简单，如果有多种分析类型，也建议按分析类型分文件
   - 多文件有助于代码管理、调试和模块化设计

2. **单文件例外原则**
   - 只有在实验极其简单且只有一种分析类型时，才考虑创建单个task
   - 单文件task描述包含所有需要执行的仿真分析

3. **多文件适用情况**
   - **不同的分析类型**（DC、AC、TRAN、噪声分析等分别在不同文件中）
   - **不同的测试环境**（如不同温度、不同电源配置）
   - **不同的电路模块**（如放大器、滤波器、偏置电路分别测试）
   - **不同的测试条件**（如不同输入信号幅度、不同负载条件）
   - **复杂的大型仿真**需要分文件管理
   - **参数扫描和优化**可以独立成文件

4. **任务描述要求**
   - 每个task的描述应该完整说明该HSPICE文件需要执行的所有仿真
   - 可以是多种分析类型的组合（DC、AC、TRAN等）
   - 描述要清晰说明该文件的测试目标和预期结果

5. **知识字段要求**
   - 每个task必须包含knowledge字段
   - knowledge字段应从OCR文本中提取具体的技术信息
   - **包括以下内容**：
     * 要求或提示使用的相关hspice语法
     * OCR文件中提到的与description相关的代码教程，方法等
     * 文件中明确提到的HSPICE语法提示（如".PARAM"、".DC"、".AC"等）
     * 特殊的仿真指令或要求（如"步长"、"精度"、"收敛条件"等）
     * 文件中标出的参数格式或数值要求
     * 特殊的模型调用或库文件引用
     * 任何明确的技术约束或条件
   - **如果OCR文本中没有上述技术信息，knowledge字段设为空字符串""，但最好要有**
   - **不要将general description或task description的内容重复到knowledge中**

实验文本：{ocr_text}

请严格按照以下JSON格式输出（每个字段都是必需的）：
{{
  "general_description": "实验总体描述和公共环境配置",
  "tasks": [
    {{
      "id": 1,
      "title": "文件名1.sp",
      "description": "该HSPICE文件的完整功能描述和测试内容，包含所有需要执行的仿真分析",
      "knowledge": "从OCR文本中提取的具体技术信息，如语法提示、参数要求等"
    }},
    {{
      "id": 2,
      "title": "文件名2.sp",
      "description": "该HSPICE文件的完整功能描述和测试内容，包含所有需要执行的仿真分析",
      "knowledge": "从OCR文本中提取的具体技术信息，如语法提示、参数要求等"
    }}
  ]
}}

**⚠️ 重要提醒：每个task对象必须包含以下4个字段，缺一不可！**
- `id`: 数字类型
- `title`: 字符串类型，必须以.sp结尾
- `description`: 字符串类型，详细描述仿真内容
- `knowledge`: 字符串类型，从OCR提取的技术信息（可为空字符串""）

**关键要点：**
- general_description必须包含所有HSPICE文件共用的环境配置
- **优先选择多任务结构，每个task对应一个独立的HSPICE文件**
- **只有在实验极其简单且只有单一分析类型时才使用单任务**
- 每个task的description要完整描述该文件的所有仿真内容
- **每个task必须包含knowledge字段，即使为空也要包含这个字段**
- 多任务结构有助于代码管理、调试和模块化设计
- 考虑按分析类型、测试环境、电路模块等因素合理分解任务
"""

        try:
            print("开始任务分析...")
            response_text = self.generate_with_retry(
                model_id, api_key, task_analysis_prompt,
                timeout=settings.TASK_ANALYSIS_TIMEOUT
            )

            # 解析响应
            task_data = self._parse_task_analysis(response_text)
            print("任务分析完成")
            return task_data

        except Exception as e:
            raise RuntimeError(f"任务分析失败: {str(e)}")

    def _parse_task_analysis(self, response_text: str) -> Dict[str, Any]:
        """解析任务分析结果"""
        try:
            # 标准化文本
            cleaned_text = normalize_line_endings(response_text.strip())
            print(f"清理后的响应文本（前500字符）: {cleaned_text[:500]}")

            # 尝试提取JSON
            task_data = extract_json_from_text(cleaned_text)

            if task_data is None:
                print("无法从响应中提取有效JSON，返回默认结构")
                print("=" * 50)
                print("完整响应文本:")
                print(response_text)
                print("=" * 50)
                return {
                    "general_description": f"无法解析任务分析结果\n原始响应: {response_text[:200]}...",
                    "tasks": []
                }

            # 转换和验证任务格式
            return self._convert_task_format(task_data)

        except Exception as e:
            print(f"任务分析解析失败: {str(e)}")
            safe_error_msg = safe_error_message(e)
            safe_response_text = safe_error_message(Exception(response_text[:200]))

            return {
                "general_description": f"任务分析结果格式错误: {safe_error_msg} 原始响应: {safe_response_text}...",
                "tasks": []
            }

    def _convert_task_format(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """转换任务格式，确保结构正确"""
        if not isinstance(task_data, dict):
            print(f"错误：task_data不是字典类型，而是{type(task_data)}")
            return {
                "general_description": "任务数据格式错误",
                "tasks": []
            }

        # 检查是否包含tasks（新格式）
        if "tasks" in task_data:
            print("检测到tasks格式")
            tasks = task_data["tasks"]

            if not isinstance(tasks, list):
                print(f"错误：tasks不是列表类型，而是{type(tasks)}")
                return {
                    "general_description": str(task_data.get("general_description", "")),
                    "tasks": []
                }

            # 标准化tasks格式
            normalized_tasks = []
            for i, task in enumerate(tasks):
                if isinstance(task, dict):
                    normalized_task = {
                        "id": task.get("id", i + 1),
                        "title": task.get("title", f"任务{i+1}.sp"),
                        "description": str(task.get("description", "")).strip(),
                        "knowledge": str(task.get("knowledge", "")).strip(),
                        "additional_info": str(task.get("additional_info", "")).strip(),
                        "generate_request": task.get("generate_request", False)
                    }
                    print(f"处理任务{i+1}: {normalized_task}")
                else:
                    normalized_task = {
                        "id": i + 1,
                        "title": f"任务{i+1}.sp",
                        "description": str(task).strip(),
                        "knowledge": "",
                        "additional_info": "",
                        "generate_request": False
                    }
                    print(f"处理任务{i+1}（非字典格式）: {normalized_task}")
                normalized_tasks.append(normalized_task)

            return {
                "general_description": str(task_data.get("general_description", "")).strip(),
                "tasks": normalized_tasks
            }

        # 默认结构
        return {
            "general_description": str(task_data.get("general_description", "")).strip(),
            "tasks": []
        }

    def generate_hspice_code(
        self,
        model_id: str,
        api_key: str = None,
        context: str = "",
        requirements: str = "",
        additional_info: str = "",
        task_description: str = "",
        filename: str = "",
        task_knowledge: str = ""
    ) -> tuple[str, str]:
        """
        生成HSPICE代码

        Args:
            model_id: 模型ID
            api_key: API密钥（可选，如果未提供则从session state获取）
            context: 检索到的知识上下文
            requirements: 实验要求
            additional_info: 补充信息（用户添加的任何相关内容）
            task_description: 任务描述
            filename: HSPICE文件名
            task_knowledge: 任务相关知识

        Returns:
            Tuple[str, str]: (分析文本, HSPICE代码)
        """
        # 代码生成提示词
        code_generation_prompt = f"""
你是一位精通HSPICE仿真的资深电路设计工程师。
请根据用户提供的"实验目标"、"补充信息"和"仿真任务描述"，并参考下方提供的"相关HSPICE知识"，完成以下任务：

1.  首先，对仿真思路进行简要分析，解释你将如何实现该HSPICE文件的仿真目标。
2.  然后，生成一份完整、可执行的HSPICE仿真代码。

# 上下文信息
## 相关HSPICE知识:
{context}

## 实验目标:
{requirements}

## 补充信息:
{additional_info}

## 仿真任务描述:
{task_description}

## 任务相关知识:
{task_knowledge}

## HSPICE文件名:
{filename}

# 输出要求
- 你的分析内容应该直接书写。
- 请务必将最终的HSPICE代码包裹在```hspice ... ```中。
- 生成的代码应该是一个完整的、可独立运行的HSPICE文件。
- 考虑代码的可复用性，当使用子模块使得代码思路更清晰时，标出子模块。
- 代码中应该包含所有必要的分析命令来完成指定的仿真任务。
"""

        try:
            print(f"开始生成 {filename} 的HSPICE代码...")
            response_text = self.generate_with_retry(model_id, api_key, code_generation_prompt)

            print(f"{filename} 模型完整响应:")
            print("=" * 50)
            print(response_text)
            print("=" * 50)

            # 解析响应
            analysis, hspice_code = self._parse_llm_output(response_text)

            print(f"{filename} 解析结果:")
            print(f"分析内容长度: {len(analysis)}")
            print(f"代码内容长度: {len(hspice_code)}")

            print("=" * 50)
            print("完整分析内容:")
            print(analysis)
            print("=" * 50)

            if hspice_code:
                print("=" * 50)
                print("完整代码内容:")
                print(hspice_code)
                print("=" * 50)
            else:
                print("警告: 没有提取到代码内容")

            print(f"{filename} 代码生成完成")

            return analysis, hspice_code

        except Exception as e:
            raise RuntimeError(f"HSPICE代码生成失败: {str(e)}")

    def _parse_llm_output(self, response_text: str) -> tuple[str, str]:
        """解析LLM输出，分离分析和代码"""
        analysis = ""
        hspice_code = ""

        cleaned_text = normalize_line_endings(response_text.strip())

        # 使用 ```hspice 作为分割点
        code_delimiter = "```hspice"

        if code_delimiter in cleaned_text:
            parts = cleaned_text.split(code_delimiter, 1)
            analysis = parts[0].strip()

            # 进一步从第二部分中分离代码和可能的后续文本
            code_part = parts[1]
            if "```" in code_part:
                hspice_code = code_part.split("```", 1)[0].strip()
            else:
                hspice_code = code_part.strip()
        else:
            # 如果模型没有按要求输出代码块，则将全部内容视为分析
            analysis = cleaned_text

        return analysis, hspice_code


# 创建全局多模型管理器实例
multi_llm_manager = MultiLLMManager()