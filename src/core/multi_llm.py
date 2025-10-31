"""
多模型LLM客户端

统一的API客户端，支持多种AI模型的调用。
包括中国和国际的主流大语言模型。
"""

import os
import time
import json
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod

import google.generativeai as genai
import openai
from anthropic import Anthropic

from config.models import AIModel, ModelProvider, model_config
from config.settings import settings
from utils.text_processing import extract_json_from_text, safe_error_message, normalize_line_endings


class BaseLLMClient(ABC):
    """LLM客户端基类"""

    def __init__(self, model: AIModel, api_key: str):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def generate_content(self, prompt: str, **kwargs) -> Any:
        """生成内容的抽象方法"""
        pass

    @abstractmethod
    def parse_response(self, response: Any) -> str:
        """解析响应的抽象方法"""
        pass


class GoogleLLMClient(BaseLLMClient):
    """Google Gemini客户端"""

    def __init__(self, model: AIModel, api_key: str):
        super().__init__(model, api_key)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model.model_id)

    def generate_content(self, prompt: str, **kwargs) -> Any:
        """生成内容"""
        timeout = kwargs.get('timeout', settings.API_TIMEOUT)
        return self.client.generate_content(
            prompt,
            request_options={"timeout": timeout}
        )

    def parse_response(self, response: Any) -> str:
        """解析响应"""
        return response.text if response and response.text else ""


class OpenAILLMClient(BaseLLMClient):
    """OpenAI客户端"""

    def __init__(self, model: AIModel, api_key: str):
        super().__init__(model, api_key)
        self.client = openai.OpenAI(api_key=api_key)

    def generate_content(self, prompt: str, **kwargs) -> Any:
        """生成内容"""
        timeout = kwargs.get('timeout', settings.API_TIMEOUT)
        max_tokens = kwargs.get('max_tokens', self.model.max_tokens)
        temperature = kwargs.get('temperature', self.model.temperature)

        return self.client.chat.completions.create(
            model=self.model.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout
        )

    def parse_response(self, response: Any) -> str:
        """解析响应"""
        if response and response.choices:
            return response.choices[0].message.content
        return ""


class AnthropicLLMClient(BaseLLMClient):
    """Anthropic Claude客户端"""

    def __init__(self, model: AIModel, api_key: str):
        super().__init__(model, api_key)
        self.client = Anthropic(api_key=api_key)

    def generate_content(self, prompt: str, **kwargs) -> Any:
        """生成内容"""
        max_tokens = kwargs.get('max_tokens', self.model.max_tokens or 4096)
        temperature = kwargs.get('temperature', self.model.temperature)

        return self.client.messages.create(
            model=self.model.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

    def parse_response(self, response: Any) -> str:
        """解析响应"""
        if response and response.content:
            return response.content[0].text
        return ""


class AlibabaLLMClient(BaseLLMClient):
    """阿里云通义千问客户端"""

    def __init__(self, model: AIModel, api_key: str):
        super().__init__(model, api_key)
        self.base_url = model.base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate_content(self, prompt: str, **kwargs) -> Any:
        """生成内容"""
        data = {
            "model": self.model.model_id,
            "input": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "parameters": {
                "max_tokens": kwargs.get('max_tokens', self.model.max_tokens),
                "temperature": kwargs.get('temperature', self.model.temperature)
            }
        }

        response = requests.post(
            f"{self.base_url}/services/aigc/text-generation/generation",
            headers=self.headers,
            json=data,
            timeout=kwargs.get('timeout', settings.API_TIMEOUT)
        )
        response.raise_for_status()
        return response.json()

    def parse_response(self, response: Any) -> str:
        """解析响应"""
        if isinstance(response, dict):
            output = response.get("output", {})
            if "text" in output:
                return output["text"]
            elif "choices" in output:
                return output["choices"][0]["message"]["content"]
        return ""


class BaiduLLMClient(BaseLLMClient):
    """百度文心一言客户端"""

    def __init__(self, model: AIModel, api_key: str):
        super().__init__(model, api_key)
        self.base_url = model.base_url
        # 百度需要获取access_token
        self.access_token = self._get_access_token()

    def _get_access_token(self) -> str:
        """获取百度API的access_token"""
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": os.getenv("BAIDU_SECRET_KEY", "")
        }
        response = requests.post(url, params=params)
        response.raise_for_status()
        return response.json()["access_token"]

    def generate_content(self, prompt: str, **kwargs) -> Any:
        """生成内容"""
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', self.model.temperature),
            "max_output_tokens": kwargs.get('max_tokens', self.model.max_tokens)
        }

        endpoint_map = {
            "ernie-bot-4": "chat/completions_pro",
            "ernie-bot-turbo": "chat/completions"
        }
        endpoint = endpoint_map.get(self.model.model_id, "chat/completions")

        url = f"{self.base_url}/{endpoint}?access_token={self.access_token}"
        response = requests.post(url, json=data, timeout=kwargs.get('timeout', settings.API_TIMEOUT))
        response.raise_for_status()
        return response.json()

    def parse_response(self, response: Any) -> str:
        """解析响应"""
        if isinstance(response, dict) and "result" in response:
            return response["result"]
        return ""


class ZhipuLLMClient(BaseLLMClient):
    """智谱清言客户端"""

    def __init__(self, model: AIModel, api_key: str):
        super().__init__(model, api_key)
        self.base_url = model.base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate_content(self, prompt: str, **kwargs) -> Any:
        """生成内容"""
        data = {
            "model": self.model.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', self.model.max_tokens),
            "temperature": kwargs.get('temperature', self.model.temperature)
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data,
            timeout=kwargs.get('timeout', settings.API_TIMEOUT)
        )
        response.raise_for_status()
        return response.json()

    def parse_response(self, response: Any) -> str:
        """解析响应"""
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["message"]["content"]
        return ""


class MoonshotLLMClient(BaseLLMClient):
    """月之暗面Kimi客户端"""

    def __init__(self, model: AIModel, api_key: str):
        super().__init__(model, api_key)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=model.base_url
        )

    def generate_content(self, prompt: str, **kwargs) -> Any:
        """生成内容"""
        timeout = kwargs.get('timeout', settings.API_TIMEOUT)
        max_tokens = kwargs.get('max_tokens', self.model.max_tokens)
        temperature = kwargs.get('temperature', self.model.temperature)

        return self.client.chat.completions.create(
            model=self.model.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout
        )

    def parse_response(self, response: Any) -> str:
        """解析响应"""
        if response and response.choices:
            return response.choices[0].message.content
        return ""


class DeepSeekLLMClient(BaseLLMClient):
    """DeepSeek客户端"""

    def __init__(self, model: AIModel, api_key: str):
        super().__init__(model, api_key)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=model.base_url
        )

    def generate_content(self, prompt: str, **kwargs) -> Any:
        """生成内容"""
        timeout = kwargs.get('timeout', settings.API_TIMEOUT)
        max_tokens = kwargs.get('max_tokens', self.model.max_tokens)
        temperature = kwargs.get('temperature', self.model.temperature)

        return self.client.chat.completions.create(
            model=self.model.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout
        )

    def parse_response(self, response: Any) -> str:
        """解析响应"""
        if response and response.choices:
            return response.choices[0].message.content
        return ""


class MistralLLMClient(BaseLLMClient):
    """Mistral AI客户端"""

    def __init__(self, model: AIModel, api_key: str):
        super().__init__(model, api_key)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=model.base_url
        )

    def generate_content(self, prompt: str, **kwargs) -> Any:
        """生成内容"""
        timeout = kwargs.get('timeout', settings.API_TIMEOUT)
        max_tokens = kwargs.get('max_tokens', self.model.max_tokens)
        temperature = kwargs.get('temperature', self.model.temperature)

        return self.client.chat.completions.create(
            model=self.model.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout
        )

    def parse_response(self, response: Any) -> str:
        """解析响应"""
        if response and response.choices:
            return response.choices[0].message.content
        return ""


class CohereLLMClient(BaseLLMClient):
    """Cohere客户端"""

    def __init__(self, model: AIModel, api_key: str):
        super().__init__(model, api_key)
        import cohere
        self.client = cohere.Client(api_key=api_key)

    def generate_content(self, prompt: str, **kwargs) -> Any:
        """生成内容"""
        max_tokens = kwargs.get('max_tokens', self.model.max_tokens)
        temperature = kwargs.get('temperature', self.model.temperature)

        return self.client.generate(
            model=self.model.model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def parse_response(self, response: Any) -> str:
        """解析响应"""
        if hasattr(response, 'generations') and response.generations:
            return response.generations[0].text
        return ""


class MultiLLMManager:
    """多模型LLM管理器"""

    def __init__(self):
        """初始化多模型管理器"""
        self.clients = {}
        self.current_model = None
        self.current_api_key = None

    def get_client(self, model_id: str, api_key: str) -> BaseLLMClient:
        """获取模型客户端"""
        cache_key = f"{model_id}:{api_key[:8]}"

        if cache_key not in self.clients:
            model = model_config.get_model_by_id(model_id)
            if not model:
                raise ValueError(f"不支持的模型: {model_id}")

            client = self._create_client(model, api_key)
            self.clients[cache_key] = client

        return self.clients[cache_key]

    def _create_client(self, model: AIModel, api_key: str) -> BaseLLMClient:
        """创建客户端实例"""
        client_map = {
            ModelProvider.GOOGLE: GoogleLLMClient,
            ModelProvider.OPENAI: OpenAILLMClient,
            ModelProvider.ANTHROPIC: AnthropicLLMClient,
            ModelProvider.ALIBABA: AlibabaLLMClient,
            ModelProvider.BAIDU: BaiduLLMClient,
            ModelProvider.ZHIPU: ZhipuLLMClient,
            ModelProvider.MOONSHOT: MoonshotLLMClient,
            ModelProvider.DEEPSEEK: DeepSeekLLMClient,
            ModelProvider.MISTRAL: MistralLLMClient,
            ModelProvider.COHERE: CohereLLMClient,
        }

        client_class = client_map.get(model.provider)
        if not client_class:
            raise ValueError(f"不支持的模型提供商: {model.provider}")

        return client_class(model, api_key)

    def generate_with_retry(
        self,
        model_id: str,
        api_key: str,
        prompt: str,
        timeout: int = None,
        max_retries: int = None,
        **kwargs
    ) -> str:
        """
        带重试机制的生成

        Args:
            model_id: 模型ID
            api_key: API密钥
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

    def analyze_tasks(self, model_id: str, api_key: str, ocr_text: str) -> Dict[str, Any]:
        """
        分析任务，将实验要求分解为多个HSPICE文件

        Args:
            model_id: 模型ID
            api_key: API密钥
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

实验文本：{ocr_text}

请输出JSON格式：
{{
  "general_description": "实验总体描述和公共环境配置",
  "tasks": [
    {{
      "id": 1,
      "title": "文件名1.sp",
      "description": "该HSPICE文件的完整功能描述和测试内容，包含所有需要执行的仿真分析"
    }},
    {{
      "id": 2,
      "title": "文件名2.sp",
      "description": "该HSPICE文件的完整功能描述和测试内容，包含所有需要执行的仿真分析"
    }}
  ]
}}

**关键要点：**
- general_description必须包含所有HSPICE文件共用的环境配置
- **优先选择多任务结构，每个task对应一个独立的HSPICE文件**
- **只有在实验极其简单且只有单一分析类型时才使用单任务**
- 每个task的description要完整描述该文件的所有仿真内容
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

            # 尝试提取JSON
            task_data = extract_json_from_text(cleaned_text)

            if task_data is None:
                print("无法从响应中提取有效JSON，返回默认结构")
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
                        "description": str(task.get("description", "")).strip()
                    }
                else:
                    normalized_task = {
                        "id": i + 1,
                        "title": f"任务{i+1}.sp",
                        "description": str(task).strip()
                    }
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
        api_key: str,
        context: str,
        requirements: str,
        mos_connections: str,
        task_description: str,
        filename: str
    ) -> Tuple[str, str]:
        """
        生成HSPICE代码

        Args:
            model_id: 模型ID
            api_key: API密钥
            context: 检索到的知识上下文
            requirements: 实验要求
            mos_connections: MOS管连接信息
            task_description: 任务描述
            filename: HSPICE文件名

        Returns:
            Tuple[str, str]: (分析文本, HSPICE代码)
        """
        # 代码生成提示词
        code_generation_prompt = f"""
你是一位精通HSPICE仿真的资深电路设计工程师。
请根据用户提供的"实验目标"、"电路图视觉信息"和"仿真任务描述"，并参考下方提供的"相关HSPICE知识"，完成以下任务：

1.  首先，对仿真思路进行简要分析，解释你将如何实现该HSPICE文件的仿真目标。
2.  然后，生成一份完整、可执行的HSPICE仿真代码。

# 上下文信息
## 相关HSPICE知识:
{context}

## 实验目标:
{requirements}

## 电路图视觉信息:
{mos_connections}

## 仿真任务描述:
{task_description}

## HSPICE文件名:
{filename}

# 输出要求
- 你的分析内容应该直接书写。
- 请务必将最终的HSPICE代码包裹在```hspice ... ```中。
- 生成的代码应该是一个完整的、可独立运行的HSPICE文件。
- 代码中应该包含所有必要的分析命令来完成指定的仿真任务。
"""

        try:
            print(f"开始生成 {filename} 的HSPICE代码...")
            response_text = self.generate_with_retry(model_id, api_key, code_generation_prompt)

            # 解析响应
            analysis, hspice_code = self._parse_llm_output(response_text)
            print(f"{filename} 代码生成完成")

            return analysis, hspice_code

        except Exception as e:
            raise RuntimeError(f"HSPICE代码生成失败: {str(e)}")

    def _parse_llm_output(self, response_text: str) -> Tuple[str, str]:
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