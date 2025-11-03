"""
AI模型配置管理模块

支持多种AI模型的配置和管理，包括中国和国际的主流大语言模型。
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


class ModelProvider(Enum):
    """AI模型提供商枚举"""
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    ALIBABA = "alibaba"
    BAIDU = "baidu"
    ZHIPU = "zhipu"
    MOONSHOT = "moonshot"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    COHERE = "cohere"


@dataclass
class AIModel:
    """AI模型配置类"""
    provider: ModelProvider
    model_id: str
    display_name: str
    api_key_env: str
    base_url: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    supports_streaming: bool = True
    is_chinese: bool = False
    description: str = ""
    # 新增：智能预配置支持
    is_official_model: bool = True  # 是否为官方预配置模型
    auto_config_url: Optional[str] = None  # 自动配置的URL

    def get_env_key(self) -> str:
        """获取环境变量键名"""
        return self.api_key_env

    def get_display_name(self) -> str:
        """获取显示名称"""
        chinese_flag = "🇨🇳" if self.is_chinese else "🌍"
        return f"{chinese_flag} {self.display_name}"

    def get_auto_config_url(self) -> Optional[str]:
        """获取自动配置URL"""
        if self.auto_config_url:
            return self.auto_config_url
        # 为官方模型提供默认URL
        if self.is_official_model:
            return self._get_default_url()
        return self.base_url

    def _get_default_url(self) -> Optional[str]:
        """获取官方模型的默认URL"""
        url_mapping = {
            ModelProvider.GOOGLE: "https://generativelanguage.googleapis.com/v1beta",
            ModelProvider.OPENAI: "https://api.openai.com/v1",
            ModelProvider.ANTHROPIC: "https://api.anthropic.com",
            ModelProvider.ALIBABA: "https://dashscope.aliyuncs.com/api/v1",
            ModelProvider.BAIDU: "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
            ModelProvider.ZHIPU: "https://open.bigmodel.cn/api/paas/v4",
            ModelProvider.MOONSHOT: "https://api.moonshot.cn/v1",
            ModelProvider.DEEPSEEK: "https://api.deepseek.com",
            ModelProvider.MISTRAL: "https://api.mistral.ai/v1",
            ModelProvider.COHERE: "https://api.cohere.com/v1",
        }
        return url_mapping.get(self.provider)


class ModelConfig:
    """模型配置管理器"""

    # 支持的AI模型配置
    SUPPORTED_MODELS = {
        # Google Models
        "gemini-2.5-flash": AIModel(
            provider=ModelProvider.GOOGLE,
            model_id="gemini-2.5-flash",
            display_name="Gemini 2.5 Flash",
            api_key_env="GOOGLE_API_KEY",
            max_tokens=8192,
            temperature=0.7,
            description="Google最新高性能模型，速度快"
        ),
        "gemini-2.5-pro": AIModel(
            provider=ModelProvider.GOOGLE,
            model_id="gemini-2.5-pro",
            display_name="Gemini 2.5 Pro",
            api_key_env="GOOGLE_API_KEY",
            max_tokens=8192,
            temperature=0.7,
            description="Google最新专业模型，质量高"
        ),

        # OpenAI Models
        "gpt-4o": AIModel(
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o",
            display_name="GPT-4o",
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            description="OpenAI最新多模态模型"
        ),
        "gpt-4o-mini": AIModel(
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o-mini",
            display_name="GPT-4o Mini",
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            description="OpenAI经济型模型"
        ),
        "gpt-4-turbo": AIModel(
            provider=ModelProvider.OPENAI,
            model_id="gpt-4-turbo",
            display_name="GPT-4 Turbo",
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            description="OpenAI高性能模型"
        ),

        # Anthropic Claude Models
        "claude-3-5-sonnet-20241022": AIModel(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-3-5-sonnet-20241022",
            display_name="Claude 3.5 Sonnet (Latest)",
            api_key_env="ANTHROPIC_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            description="Anthropic最新高性能模型"
        ),
        "claude-3-haiku-20240307": AIModel(
            provider=ModelProvider.ANTHROPIC,
            model_id="claude-3-haiku-20240307",
            display_name="Claude 3 Haiku",
            api_key_env="ANTHROPIC_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            description="Anthropic快速响应模型"
        ),

        # 阿里巴巴通义千问
        "qwen-turbo": AIModel(
            provider=ModelProvider.ALIBABA,
            model_id="qwen-turbo",
            display_name="通义千问 Turbo",
            api_key_env="DASHSCOPE_API_KEY",
            base_url="https://dashscope.aliyuncs.com/api/v1",
            max_tokens=6000,
            temperature=0.7,
            is_chinese=True,
            description="阿里云通义千问高性能版本"
        ),
        "qwen-plus": AIModel(
            provider=ModelProvider.ALIBABA,
            model_id="qwen-plus",
            display_name="通义千问 Plus",
            api_key_env="DASHSCOPE_API_KEY",
            base_url="https://dashscope.aliyuncs.com/api/v1",
            max_tokens=6000,
            temperature=0.7,
            is_chinese=True,
            description="阿里云通义千问增强版本"
        ),
        "qwen-max": AIModel(
            provider=ModelProvider.ALIBABA,
            model_id="qwen-max",
            display_name="通义千问 Max",
            api_key_env="DASHSCOPE_API_KEY",
            base_url="https://dashscope.aliyuncs.com/api/v1",
            max_tokens=6000,
            temperature=0.7,
            is_chinese=True,
            description="阿里云通义千问顶级版本"
        ),

        # 百度文心一言
        "ernie-bot-4": AIModel(
            provider=ModelProvider.BAIDU,
            model_id="ernie-bot-4",
            display_name="文心一言 4.0",
            api_key_env="BAIDU_API_KEY",
            base_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
            max_tokens=4096,
            temperature=0.7,
            is_chinese=True,
            description="百度文心一言最新版本"
        ),
        "ernie-bot-turbo": AIModel(
            provider=ModelProvider.BAIDU,
            model_id="ernie-bot-turbo",
            display_name="文心一言 Turbo",
            api_key_env="BAIDU_API_KEY",
            base_url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
            max_tokens=4096,
            temperature=0.7,
            is_chinese=True,
            description="百度文心一言快速版本"
        ),

        # 智谱清言
        "glm-4": AIModel(
            provider=ModelProvider.ZHIPU,
            model_id="glm-4",
            display_name="智谱清言 GLM-4",
            api_key_env="ZHIPUAI_API_KEY",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            max_tokens=4096,
            temperature=0.7,
            is_chinese=True,
            description="智谱AI最新模型"
        ),
        "glm-3-turbo": AIModel(
            provider=ModelProvider.ZHIPU,
            model_id="glm-3-turbo",
            display_name="智谱清言 GLM-3 Turbo",
            api_key_env="ZHIPUAI_API_KEY",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            max_tokens=4096,
            temperature=0.7,
            is_chinese=True,
            description="智谱AI快速版本"
        ),

        # 月之暗面 Kimi
        "moonshot-v1-8k": AIModel(
            provider=ModelProvider.MOONSHOT,
            model_id="moonshot-v1-8k",
            display_name="Kimi 8K",
            api_key_env="MOONSHOT_API_KEY",
            base_url="https://api.moonshot.cn/v1",
            max_tokens=7948,
            temperature=0.7,
            is_chinese=True,
            description="月之暗面Kimi 8K上下文版本"
        ),
        "moonshot-v1-32k": AIModel(
            provider=ModelProvider.MOONSHOT,
            model_id="moonshot-v1-32k",
            display_name="Kimi 32K",
            api_key_env="MOONSHOT_API_KEY",
            base_url="https://api.moonshot.cn/v1",
            max_tokens=32768,
            temperature=0.7,
            is_chinese=True,
            description="月之暗面Kimi 32K上下文版本"
        ),

        # DeepSeek
        "deepseek-chat": AIModel(
            provider=ModelProvider.DEEPSEEK,
            model_id="deepseek-chat",
            display_name="DeepSeek Chat",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            max_tokens=4096,
            temperature=0.7,
            is_chinese=True,
            description="深度求索对话模型"
        ),
        "deepseek-coder": AIModel(
            provider=ModelProvider.DEEPSEEK,
            model_id="deepseek-coder",
            display_name="DeepSeek Coder",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            max_tokens=4096,
            temperature=0.7,
            is_chinese=True,
            description="深度求索代码专用模型"
        ),

        # Mistral AI
        "mistral-large-latest": AIModel(
            provider=ModelProvider.MISTRAL,
            model_id="mistral-large-latest",
            display_name="Mistral Large",
            api_key_env="MISTRAL_API_KEY",
            base_url="https://api.mistral.ai/v1",
            max_tokens=4096,
            temperature=0.7,
            description="Mistral AI高性能模型"
        ),
        "mistral-small": AIModel(
            provider=ModelProvider.MISTRAL,
            model_id="mistral-small",
            display_name="Mistral Small",
            api_key_env="MISTRAL_API_KEY",
            base_url="https://api.mistral.ai/v1",
            max_tokens=4096,
            temperature=0.7,
            description="Mistral AI经济型模型"
        ),

        # Cohere
        "command-r-plus": AIModel(
            provider=ModelProvider.COHERE,
            model_id="command-r-plus",
            display_name="Cohere Command R+",
            api_key_env="COHERE_API_KEY",
            base_url="https://api.cohere.com/v1",
            max_tokens=4096,
            temperature=0.7,
            description="Cohere高性能模型"
        ),
        "command": AIModel(
            provider=ModelProvider.COHERE,
            model_id="command",
            display_name="Cohere Command",
            api_key_env="COHERE_API_KEY",
            base_url="https://api.cohere.com/v1",
            max_tokens=4096,
            temperature=0.7,
            description="Cohere标准模型"
        ),
    }

    @classmethod
    def get_all_models(cls) -> Dict[str, AIModel]:
        """获取所有支持的模型"""
        return cls.SUPPORTED_MODELS.copy()

    @classmethod
    def get_model_by_id(cls, model_id: str) -> Optional[AIModel]:
        """根据模型ID获取模型配置"""
        return cls.SUPPORTED_MODELS.get(model_id)

    @classmethod
    def get_models_by_provider(cls, provider: ModelProvider) -> List[AIModel]:
        """根据提供商获取模型列表"""
        return [model for model in cls.SUPPORTED_MODELS.values()
                if model.provider == provider]

    @classmethod
    def get_chinese_models(cls) -> List[AIModel]:
        """获取中国模型列表"""
        return [model for model in cls.SUPPORTED_MODELS.values()
                if model.is_chinese]

    @classmethod
    def get_international_models(cls) -> List[AIModel]:
        """获取国际模型列表"""
        return [model for model in cls.SUPPORTED_MODELS.values()
                if not model.is_chinese]

    @classmethod
    def get_model_choices_for_ui(cls) -> Dict[str, str]:
        """获取UI用的模型选择列表"""
        choices = {}
        # 按提供商分组
        providers = {}
        for model_id, model in cls.SUPPORTED_MODELS.items():
            provider_name = model.provider.value.upper()
            if provider_name not in providers:
                providers[provider_name] = []
            providers[provider_name].append((model_id, model.get_display_name()))

        # 生成选择列表
        for provider_name, models in sorted(providers.items()):
            for model_id, display_name in models:
                choices[model_id] = f"{provider_name}: {display_name}"

        return choices

    @classmethod
    def get_recommended_models(cls) -> List[str]:
        """获取推荐模型列表"""
        return [
            "gemini-2.5-flash",  # 已配置的Google模型
            "gpt-4o-mini",        # OpenAI经济型
            "claude-3-haiku-20240307",  # Anthropic快速版
            "qwen-turbo",         # 阿里通义千问
            "glm-3-turbo",        # 智谱清言
            "moonshot-v1-8k",     # Kimi
            "deepseek-chat",      # DeepSeek
        ]

    @classmethod
    def is_official_model(cls, model_id: str) -> bool:
        """判断是否为官方预配置模型"""
        model = cls.get_model_by_id(model_id)
        return model.is_official_model if model else False

    @classmethod
    def get_auto_config_for_model(cls, model_id: str) -> Dict[str, Any]:
        """获取模型的自动配置信息"""
        model = cls.get_model_by_id(model_id)
        if not model or not model.is_official_model:
            return {}

        return {
            "model_id": model.model_id,
            "display_name": model.display_name,
            "provider": model.provider.value,
            "base_url": model.get_auto_config_url(),
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "supports_streaming": model.supports_streaming,
            "description": model.description,
            "is_chinese": model.is_chinese
        }

    @classmethod
    def get_official_models_list(cls) -> List[Dict[str, Any]]:
        """获取所有官方模型的配置信息列表"""
        official_models = []
        for model_id, model in cls.SUPPORTED_MODELS.items():
            if model.is_official_model:
                official_models.append(cls.get_auto_config_for_model(model_id))
        return official_models


# 创建全局配置实例
model_config = ModelConfig()