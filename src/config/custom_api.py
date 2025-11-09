"""
自定义API配置管理模块

支持用户自定义大模型API配置，类似Cherry Studio的功能。
包括：
1. 自定义API提供商配置
2. 动态模型发现
3. API连接测试
4. 配置持久化存储
"""

import json
import os
import base64
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from config.settings import settings


@dataclass
class CustomAPIConfig:
    """自定义API配置类"""
    provider_name: str           # 用户自定义提供商名称
    base_url: str               # API基础URL
    api_key: str               # API密钥（加密存储）
    models: List[str]          # 支持的模型列表
    is_active: bool = True     # 是否启用
    created_at: Optional[str] = None
    last_tested: Optional[str] = None
    test_status: str = ""      # 测试状态
    description: str = ""      # 用户描述

    def __post_init__(self):
        """初始化后处理"""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def get_display_info(self) -> str:
        """获取显示信息"""
        status = "✅" if self.is_active else "❌"
        test_status = "🟢" if self.test_status == "success" else "🔴" if self.test_status == "failed" else "🟡"
        return f"{status} {test_status} {self.provider_name}"

    def mask_api_key(self) -> str:
        """掩码显示API密钥"""
        if not self.api_key:
            return ""
        if len(self.api_key) <= 8:
            return "*" * len(self.api_key)
        return self.api_key[:4] + "*" * (len(self.api_key) - 8) + self.api_key[-4:]


class CustomAPIManager:
    """自定义API管理器"""

    def __init__(self, config_file: Optional[str] = None):
        """初始化管理器"""
        if config_file is None:
            config_file = os.path.join(settings.ensure_directory("config"), "custom_apis.json")
        self.config_file = config_file
        self._ensure_config_dir()
        self._configs_cache: Optional[List[CustomAPIConfig]] = None

    def _ensure_config_dir(self):
        """确保配置目录存在"""
        config_dir = os.path.dirname(self.config_file)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)

    def _simple_encrypt(self, text: str) -> str:
        """简单加密（用于API密钥存储）"""
        if not text:
            return ""
        # 使用base64简单编码，实际项目中应使用更安全的加密
        encoded = base64.b64encode(text.encode()).decode()
        # 添加简单混淆
        return encoded[::-1]

    def _simple_decrypt(self, encrypted_text: str) -> str:
        """简单解密"""
        if not encrypted_text:
            return ""
        try:
            # 反转混淆并解码
            reversed_text = encrypted_text[::-1]
            decoded = base64.b64decode(reversed_text.encode()).decode()
            return decoded
        except Exception:
            return ""

    def _load_configs(self) -> List[CustomAPIConfig]:
        """加载配置"""
        if self._configs_cache is not None:
            return self._configs_cache

        configs = []
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for config_data in data.get('custom_apis', []):
                    # 解密API密钥
                    if 'api_key' in config_data:
                        config_data['api_key'] = self._simple_decrypt(config_data['api_key'])

                    config = CustomAPIConfig(**config_data)
                    configs.append(config)

            except Exception as e:
                print(f"加载自定义API配置失败: {e}")

        self._configs_cache = configs
        return configs

    def _save_configs(self, configs: List[CustomAPIConfig]):
        """保存配置"""
        try:
            # 准备保存数据
            save_data = {
                'custom_apis': []
            }

            for config in configs:
                config_dict = asdict(config)
                # 加密API密钥
                if 'api_key' in config_dict and config_dict['api_key']:
                    config_dict['api_key'] = self._simple_encrypt(config_dict['api_key'])
                save_data['custom_apis'].append(config_dict)

            # 保存到文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            # 更新缓存
            self._configs_cache = configs

        except Exception as e:
            raise Exception(f"保存自定义API配置失败: {e}")

    def add_custom_api(self, config: CustomAPIConfig) -> bool:
        """
        添加自定义API配置

        Args:
            config: 自定义API配置

        Returns:
            bool: 是否添加成功
        """
        try:
            # 检查名称是否已存在
            configs = self._load_configs()
            for existing_config in configs:
                if existing_config.provider_name == config.provider_name:
                    raise Exception(f"提供商名称 '{config.provider_name}' 已存在")

            # 添加新配置
            configs.append(config)
            self._save_configs(configs)
            return True

        except Exception as e:
            print(f"添加自定义API失败: {e}")
            return False

    def get_all_configs(self) -> List[CustomAPIConfig]:
        """获取所有自定义API配置"""
        return self._load_configs()

    def get_config_by_name(self, provider_name: str) -> Optional[CustomAPIConfig]:
        """根据提供商名称获取配置"""
        configs = self._load_configs()
        for config in configs:
            if config.provider_name == provider_name:
                return config
        return None

    def update_config(self, provider_name: str, updates: Dict[str, Any]) -> bool:
        """
        更新配置

        Args:
            provider_name: 提供商名称
            updates: 更新字段

        Returns:
            bool: 是否更新成功
        """
        try:
            configs = self._load_configs()
            for i, config in enumerate(configs):
                if config.provider_name == provider_name:
                    # 更新字段
                    for key, value in updates.items():
                        if hasattr(config, key):
                            setattr(config, key, value)

                    configs[i] = config
                    self._save_configs(configs)
                    return True

            return False

        except Exception as e:
            print(f"更新自定义API配置失败: {e}")
            return False

    def delete_config(self, provider_name: str) -> bool:
        """
        删除配置

        Args:
            provider_name: 提供商名称

        Returns:
            bool: 是否删除成功
        """
        try:
            configs = self._load_configs()
            configs = [config for config in configs if config.provider_name != provider_name]
            self._save_configs(configs)
            return True

        except Exception as e:
            print(f"删除自定义API配置失败: {e}")
            return False

    def get_active_configs(self) -> List[CustomAPIConfig]:
        """获取所有启用的配置"""
        configs = self._load_configs()
        return [config for config in configs if config.is_active]

    def test_api_connection(self, config: CustomAPIConfig) -> Tuple[bool, str]:
        """
        测试API连接

        Args:
            config: API配置

        Returns:
            Tuple[bool, str]: (是否成功, 错误信息)
        """
        try:
            import requests

            # 构造测试请求
            headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            }

            # 尝试获取模型列表
            models_url = f"{config.base_url.rstrip('/')}/models"
            response = requests.get(models_url, headers=headers, timeout=10)

            if response.status_code == 200:
                # 更新模型列表
                models_data = response.json()
                if 'data' in models_data:
                    config.models = [model['id'] for model in models_data['data']]

                # 更新测试状态
                self.update_config(config.provider_name, {
                    'last_tested': datetime.now().isoformat(),
                    'test_status': 'success'
                })

                return True, "连接成功"
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"

                # 更新测试状态
                self.update_config(config.provider_name, {
                    'last_tested': datetime.now().isoformat(),
                    'test_status': 'failed'
                })

                return False, error_msg

        except Exception as e:
            error_msg = str(e)

            # 更新测试状态
            self.update_config(config.provider_name, {
                'last_tested': datetime.now().isoformat(),
                'test_status': 'failed'
            })

            return False, error_msg

    def discover_models(self, config: CustomAPIConfig) -> List[str]:
        """
        发现API支持的模型

        Args:
            config: API配置

        Returns:
            List[str]: 模型列表
        """
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            }

            models_url = f"{config.base_url.rstrip('/')}/models"
            response = requests.get(models_url, headers=headers, timeout=10)

            if response.status_code == 200:
                models_data = response.json()
                if 'data' in models_data:
                    models = [model['id'] for model in models_data['data']]

                    # 更新配置中的模型列表
                    self.update_config(config.provider_name, {'models': models})

                    return models

            return []

        except Exception as e:
            print(f"发现模型失败: {e}")
            return []

    def get_all_available_models(self) -> Dict[str, List[str]]:
        """获取所有自定义API的可用模型"""
        result = {}
        active_configs = self.get_active_configs()

        for config in active_configs:
            if config.models:
                result[config.provider_name] = config.models

        return result

    def create_config_from_url(self, provider_name: str, base_url: str, api_key: str, description: str = "") -> bool:
        """
        从URL创建配置（自动发现模型）

        Args:
            provider_name: 提供商名称
            base_url: API基础URL
            api_key: API密钥
            description: 描述

        Returns:
            bool: 是否创建成功
        """
        try:
            # 创建临时配置用于测试
            temp_config = CustomAPIConfig(
                provider_name=provider_name,
                base_url=base_url,
                api_key=api_key,
                models=[],
                description=description
            )

            # 测试连接并发现模型
            success, error_msg = self.test_api_connection(temp_config)
            if not success:
                raise Exception(f"API连接测试失败: {error_msg}")

            # 发现模型
            models = self.discover_models(temp_config)

            # 创建最终配置
            final_config = CustomAPIConfig(
                provider_name=provider_name,
                base_url=base_url,
                api_key=api_key,
                models=models,
                description=description,
                test_status='success',
                last_tested=datetime.now().isoformat()
            )

            return self.add_custom_api(final_config)

        except Exception as e:
            print(f"从URL创建配置失败: {e}")
            return False


# 创建全局管理器实例
custom_api_manager = CustomAPIManager()