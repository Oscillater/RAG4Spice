
"""
AI模型配置流程组件

提供清晰的AI模型配置界面：
1. 先选择API配置方式（环境变量 vs 网页设置）
2. 再进行模型选择
3. API连接测试
4. 在主界面显示连接状态
"""

import os
import streamlit as st
from typing import Optional, Tuple, Dict, List
from enum import Enum

from config.models import model_config, AIModel, ModelProvider
from config.settings import settings
from config.custom_api import custom_api_manager
from core.multi_llm import multi_llm_manager
from ui.custom_api_config import custom_api_config_ui


class APIConfigMethod(Enum):
    """API配置方式枚举"""
    ENVIRONMENT = "environment"  # 环境变量方式
    WEB_INPUT = "web_input"      # 网页输入方式


class ModelConfigFlow:
    """AI模型配置流程组件"""

    def __init__(self):
        """初始化模型配置流程组件"""
        self._init_session_state()

    def _init_session_state(self):
        """初始化会话状态"""
        # API配置方式选择
        if 'api_config_method' not in st.session_state:
            st.session_state.api_config_method = APIConfigMethod.ENVIRONMENT

        # 模型配置状态
        if 'selected_analysis_model' not in st.session_state:
            st.session_state.selected_analysis_model = settings.DEFAULT_MODEL
        if 'selected_generation_model' not in st.session_state:
            st.session_state.selected_generation_model = settings.DEFAULT_MODEL

        # API连接状态 - 确保在任何地方都能访问到
        if 'api_connection_status' not in st.session_state:
            st.session_state.api_connection_status = {
                'analysis': {'connected': False, 'model': '', 'error': ''},
                'generation': {'connected': False, 'model': '', 'error': ''}
            }

        # 会话存储的API密钥
        if 'session_api_keys' not in st.session_state:
            st.session_state.session_api_keys = {}

    def render_config_flow(self) -> bool:
        """
        渲染完整的配置流程

        Returns:
            bool: 是否配置完成且连接测试通过
        """
        st.subheader("🤖 AI模型配置")

        # 第一步：选择API配置方式
        config_method = self._render_api_config_method_selection()

        if config_method == APIConfigMethod.ENVIRONMENT:
            return self._render_environment_config_flow()
        else:
            return self._render_web_input_config_flow()

    def _render_api_config_method_selection(self) -> APIConfigMethod:
        """渲染API配置方式选择"""
        st.write("### 1️⃣ 选择API配置方式")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "🔧 环境变量配置",
                help="通过系统环境变量配置API密钥，推荐用于生产环境",
                use_container_width=True,
                type="primary" if st.session_state.api_config_method == APIConfigMethod.ENVIRONMENT else "secondary",
                key="config_method_env"
            ):
                st.session_state.api_config_method = APIConfigMethod.ENVIRONMENT
                st.rerun()

        with col2:
            if st.button(
                "💻 网页输入配置",
                help="直接在网页中输入API密钥，方便测试和临时使用",
                use_container_width=True,
                type="primary" if st.session_state.api_config_method == APIConfigMethod.WEB_INPUT else "secondary",
                key="config_method_web"
            ):
                st.session_state.api_config_method = APIConfigMethod.WEB_INPUT
                st.rerun()

        # 显示当前选择和说明
        if st.session_state.api_config_method == APIConfigMethod.ENVIRONMENT:
            st.info("✅ **已选择：环境变量配置**\n\n系统将从环境变量中读取API密钥。请确保已正确设置相应的环境变量。")
        else:
            st.info("✅ **已选择：网页输入配置**\n\n您可以直接在网页中输入API密钥，密钥将保存在当前会话中。")

        return st.session_state.api_config_method

    def _render_environment_config_flow(self) -> bool:
        """渲染环境变量配置流程"""
        st.write("### 2️⃣ 检查环境变量中的API密钥")

        # 获取所有支持的模型的环境变量状态
        env_status = self._get_environment_status()

        # 显示环境变量状态
        self._display_environment_status(env_status)

        # 第二步：模型选择
        st.write("### 3️⃣ 选择AI模型")
        return self._render_model_selection_and_test(APIConfigMethod.ENVIRONMENT)

    def _render_web_input_config_flow(self) -> bool:
        """渲染网页输入配置流程"""
        st.write("### 2️⃣ 输入API密钥并选择模型")

        # 先选择模型，再输入对应的API密钥
        return self._render_model_selection_and_test(APIConfigMethod.WEB_INPUT)

    def _get_environment_status(self) -> Dict[str, Dict]:
        """获取环境变量状态"""
        env_status = {}
        for model_id, model in model_config.get_all_models().items():
            env_key = model.get_env_key()
            has_key = bool(os.getenv(env_key))

            env_status[model_id] = {
                'model': model,
                'env_key': env_key,
                'has_key': has_key,
                'provider': model.provider.value.upper()
            }

        return env_status

    def _display_environment_status(self, env_status: Dict[str, Dict]):
        """显示环境变量状态"""
        # 按提供商分组
        providers = {}
        for model_id, status in env_status.items():
            provider = status['provider']
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(status)

        # 显示状态
        total_models = len(env_status)
        configured_models = sum(1 for status in env_status.values() if status['has_key'])

        if configured_models == 0:
            st.error("❌ 未检测到任何API密钥环境变量")
            st.info("💡 请设置环境变量后重新运行应用，或选择'网页输入配置'方式")
        else:
            st.success(f"✅ 已检测到 {configured_models}/{total_models} 个API密钥")

        # 按提供商显示详细状态
        for provider, models in sorted(providers.items()):
            with st.expander(f"🏢 {provider}", expanded=False):
                for model_status in models:
                    model = model_status['model']
                    has_key = model_status['has_key']
                    env_key = model_status['env_key']

                    status_icon = "✅" if has_key else "❌"
                    status_text = "已配置" if has_key else "未配置"

                    st.write(f"{status_icon} **{model.display_name}**")
                    st.code(f"环境变量: {env_key}")
                    st.write(f"状态: {status_text}")
                    st.divider()

    def _render_model_selection_and_test(self, config_method: APIConfigMethod) -> bool:
        """渲染模型选择和测试"""
        # 获取所有模型选择（包括自定义API）
        model_choices = model_config.get_model_choices_for_ui()
        custom_choices = custom_api_config_ui.get_model_choices_for_ui()

        # 合并选择列表
        all_choices = {**model_choices, **custom_choices}

        # 任务分析模型选择
        st.write("#### 📊 任务分析模型")
        analysis_model_id = st.selectbox(
            "选择用于分析实验要求的模型",
            options=list(all_choices.keys()),
            format_func=lambda x: all_choices[x],
            index=self._get_model_index(st.session_state.selected_analysis_model, all_choices),
            key=f"analysis_model_select_{config_method.value}"  # 添加配置方法到key
        )
        st.session_state.selected_analysis_model = analysis_model_id

        # 显示模型信息
        self._display_model_info(analysis_model_id, config_method)  # 传递config_method避免冲突

        # API密钥配置/显示
        analysis_api_key = self._handle_api_key(analysis_model_id, "analysis", config_method)

        # 代码生成模型选择
        st.write("#### 💻 代码生成模型")

        # 是否使用相同模型的选项
        use_same_model = st.checkbox("🔗 使用相同模型进行代码生成", value=True,
                                  key=f"use_same_model_{config_method.value}")  # 添加配置方法到key

        if use_same_model:
            generation_model_id = analysis_model_id
            generation_api_key = analysis_api_key
            st.session_state.selected_generation_model = generation_model_id
        else:
            generation_model_id = st.selectbox(
                "选择用于生成HSPICE代码的模型",
                options=list(all_choices.keys()),
                format_func=lambda x: all_choices[x],
                index=self._get_model_index(st.session_state.selected_generation_model, all_choices),
                key=f"generation_model_select_{config_method.value}"  # 添加配置方法到key
            )
            st.session_state.selected_generation_model = generation_model_id

            # 显示模型信息
            self._display_model_info(generation_model_id, config_method)  # 传递config_method避免冲突

            # API密钥配置/显示
            generation_api_key = self._handle_api_key(generation_model_id, "generation", config_method)

        # API连接测试
        st.write("### 4️⃣ API连接测试")
        return self._render_api_connection_test(
            analysis_model_id, analysis_api_key,
            generation_model_id, generation_api_key
        )

    def _get_model_index(self, model_id: str, model_choices: Dict[str, str]) -> int:
        """获取模型在选择列表中的索引"""
        try:
            return list(model_choices.keys()).index(model_id)
        except ValueError:
            return 0

    def _display_model_info(self, model_id: str, config_method: APIConfigMethod):
        """显示模型信息"""
        # 检查是否为自定义API模型
        if model_id.startswith("custom:"):
            self._display_custom_model_info(model_id)
            return

        model = model_config.get_model_by_id(model_id)
        if not model:
            return

        with st.expander(f"📋 {model.get_display_name()} 详细信息", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**基本信息**")
                st.write(f"- **提供商**: {model.provider.value.upper()}")
                st.write(f"- **模型ID**: `{model.model_id}`")
                st.write(f"- **支持流式**: {'✅' if model.supports_streaming else '❌'}")

            with col2:
                st.write("**参数配置**")
                st.write(f"- **最大Token**: {model.max_tokens}")
                st.write(f"- **温度**: {model.temperature}")
                st.write(f"- **中文优化**: {'✅' if model.is_chinese else '❌'}")

            if model.description:
                st.write("**描述**")
                st.info(model.description)

    def _display_custom_model_info(self, model_id: str):
        """显示自定义模型信息"""
        # 解析模型ID: custom:provider_name:model_name
        parts = model_id.split(":", 2)
        if len(parts) != 3:
            return

        _, provider_name, model_name = parts

        # 获取自定义API配置
        custom_config = custom_api_manager.get_config_by_name(provider_name)
        if not custom_config:
            return

        with st.expander(f"🔧 {provider_name}: {model_name} 详细信息", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**基本信息**")
                st.write(f"- **提供商**: {provider_name}")
                st.write(f"- **模型ID**: `{model_name}`")
                st.write(f"- **API地址**: `{custom_config.base_url}`")
                st.write(f"- **支持流式**: ✅")

            with col2:
                st.write("**参数配置**")
                st.write(f"- **最大Token**: 4096")
                st.write(f"- **温度**: 0.7")
                st.write(f"- **状态**: {'🟢 启用' if custom_config.is_active else '🔴 禁用'}")

            if custom_config.description:
                st.write("**描述**")
                st.info(custom_config.description)

            if custom_config.last_tested:
                status_icon = "✅" if custom_config.test_status == "success" else "❌" if custom_config.test_status == "failed" else "❓"
                st.write(f"**测试状态**: {status_icon} {custom_config.test_status}")
                st.write(f"**最后测试**: {custom_config.last_tested}")

    def _handle_api_key(self, model_id: str, config_type: str, config_method: APIConfigMethod) -> str:
        """处理API密钥"""
        # 检查是否为自定义API模型
        if model_id.startswith("custom:"):
            return self._handle_custom_api_key(model_id, config_type, config_method)

        model = model_config.get_model_by_id(model_id)
        if not model:
            return ""

        env_key = model.get_env_key()

        if config_method == APIConfigMethod.ENVIRONMENT:
            # 环境变量方式
            env_api_key = os.getenv(env_key)
            if env_api_key:
                st.success(f"✅ 已从环境变量 `{env_key}` 加载API密钥")
                masked_key = "*" * (len(env_api_key) - 4) + env_api_key[-4:] if len(env_api_key) > 4 else "*"
                # 使用唯一的key避免冲突
                st.text_input("API密钥", value=masked_key, type="password", disabled=True,
                           key=f"env_display_{config_type}_{model_id}")
                return env_api_key
            else:
                st.error(f"❌ 未检测到环境变量 `{env_key}`")
                st.info("💡 请设置环境变量后重新运行应用，或选择'网页输入配置'方式")
                return ""

        else:
            # 网页输入方式
            session_key = f"{config_type}_{model_id}"

            # 检查是否已保存
            if session_key in st.session_state.session_api_keys:
                saved_key = st.session_state.session_api_keys[session_key]
                st.success(f"✅ 已保存API密钥")

                masked_key = "*" * (len(saved_key) - 4) + saved_key[-4:] if len(saved_key) > 4 else "*"
                # 使用唯一的key避免冲突
                st.text_input("API密钥", value=masked_key, type="password", disabled=True,
                           key=f"saved_display_{config_type}_{model_id}")

                if st.button(f"清除{model.display_name}的API密钥", key=f"clear_{config_type}_{model_id}"):
                    del st.session_state.session_api_keys[session_key]
                    st.rerun()

                return saved_key
            else:
                # 输入API密钥 - 使用更具体的key确保唯一性
                api_key = st.text_input(
                    f"输入{model.display_name}的API密钥",
                    type="password",
                    key=f"api_key_input_{config_type}_{model_id}_{model.provider.value}",  # 更具体的key
                    help=f"请输入 {model.provider.value.upper()} 的API密钥"
                )

                if api_key and st.button(f"保存{model.display_name}的API密钥", key=f"save_{config_type}_{model_id}"):
                    st.session_state.session_api_keys[session_key] = api_key
                    st.success("API密钥已保存到当前会话")
                    st.rerun()

                return api_key

    def _handle_custom_api_key(self, model_id: str, config_type: str, config_method: APIConfigMethod) -> str:
        """处理自定义API密钥"""
        # 解析模型ID: custom:provider_name:model_name
        parts = model_id.split(":", 2)
        if len(parts) != 3:
            return ""

        _, provider_name, model_name = parts

        # 获取自定义API配置
        custom_config = custom_api_manager.get_config_by_name(provider_name)
        if not custom_config:
            st.error(f"❌ 未找到自定义API配置: {provider_name}")
            return ""

        if config_method == APIConfigMethod.ENVIRONMENT:
            # 自定义API不支持环境变量方式，显示提示
            st.error(f"❌ 自定义API '{provider_name}' 仅支持网页输入配置方式")
            st.info("💡 请选择'网页输入配置'方式来使用自定义API")
            return ""

        else:
            # 网页输入方式 - 从配置中获取API密钥
            if custom_config.is_active:
                st.success(f"✅ 已从配置加载API密钥")
                masked_key = custom_config.mask_api_key()
                st.text_input("API密钥", value=masked_key, type="password", disabled=True,
                           key=f"custom_display_{config_type}_{provider_name}")
                return custom_config.api_key
            else:
                st.error(f"❌ 自定义API '{provider_name}' 已禁用")
                st.info("💡 请在自定义API配置页面中启用此API")
                return ""

    def _render_api_connection_test(
        self,
        analysis_model_id: str, analysis_api_key: str,
        generation_model_id: str, generation_api_key: str
    ) -> bool:
        """渲染API连接测试"""
        # 测试按钮
        col1, col2 = st.columns(2)

        with col1:
            test_analysis = st.button(
                "🧪 测试分析模型连接",
                key=f"test_analysis_{analysis_model_id}",
                use_container_width=True,
                disabled=not analysis_api_key
            )

        with col2:
            test_generation = st.button(
                "🧪 测试生成模型连接",
                key=f"test_generation_{generation_model_id}",
                use_container_width=True,
                disabled=not generation_api_key
            )

        # 执行测试
        if test_analysis and analysis_api_key:
            self._test_single_api_connection("analysis", analysis_model_id, analysis_api_key)

        if test_generation and generation_api_key:
            self._test_single_api_connection("generation", generation_model_id, generation_api_key)

        # 显示连接状态
        self._display_connection_status()

        # 检查是否都连接成功
        analysis_connected = st.session_state.api_connection_status['analysis']['connected']
        generation_connected = st.session_state.api_connection_status['generation']['connected']

        if analysis_connected and generation_connected:
            st.success("🎉 所有模型API连接测试通过！可以开始使用系统了。")
            return True
        elif analysis_connected or generation_connected:
            st.warning("⚠️ 部分模型API连接测试通过，建议检查未连接的模型配置。")
            return False
        else:
            st.info("💡 请完成API密钥配置并测试连接。")
            return False

    def _test_single_api_connection(self, config_type: str, model_id: str, api_key: str):
        """测试单个API连接"""
        with st.spinner(f"正在测试{config_type}模型连接..."):
            try:
                test_prompt = "请回复'连接成功'，不要其他内容。"

                # 对于自定义API，使用更短的超时时间和重试次数
                if model_id.startswith("custom:"):
                    response = multi_llm_manager.generate_with_retry(
                        model_id, api_key, test_prompt, max_retries=1, timeout=30
                    )
                else:
                    response = multi_llm_manager.generate_with_retry(
                        model_id, api_key, test_prompt, max_retries=1
                    )

                if "连接成功" in response or "success" in response.lower():
                    st.success(f"✅ {config_type}模型连接测试成功！")
                    st.session_state.api_connection_status[config_type] = {
                        'connected': True,
                        'model': model_id,
                        'error': ''
                    }
                else:
                    st.warning(f"⚠️ {config_type}模型连接成功，但响应异常: {response[:100]}...")
                    st.session_state.api_connection_status[config_type] = {
                        'connected': True,
                        'model': model_id,
                        'error': f"响应异常: {response[:50]}..."
                    }

            except Exception as e:
                st.error(f"❌ {config_type}模型连接测试失败: {str(e)}")
                st.session_state.api_connection_status[config_type] = {
                    'connected': False,
                    'model': model_id,
                    'error': str(e)
                }

    def _display_connection_status(self):
        """显示连接状态"""
        st.write("#### 📊 API连接状态")

        col1, col2 = st.columns(2)

        with col1:
            analysis_status = st.session_state.api_connection_status['analysis']
            if analysis_status['connected']:
                st.success("✅ 分析模型已连接")
                model_display_name = self._get_model_display_name(analysis_status['model'])
                if model_display_name:
                    st.write(f"模型: {model_display_name}")
                if analysis_status['error']:
                    st.warning(f"注意: {analysis_status['error']}")
            else:
                st.error("❌ 分析模型未连接")
                if analysis_status['error']:
                    st.code(f"错误: {analysis_status['error']}")

        with col2:
            generation_status = st.session_state.api_connection_status['generation']
            if generation_status['connected']:
                st.success("✅ 生成模型已连接")
                model_display_name = self._get_model_display_name(generation_status['model'])
                if model_display_name:
                    st.write(f"模型: {model_display_name}")
                if generation_status['error']:
                    st.warning(f"注意: {generation_status['error']}")
            else:
                st.error("❌ 生成模型未连接")
                if generation_status['error']:
                    st.code(f"错误: {generation_status['error']}")

    def _get_model_display_name(self, model_id: str) -> str:
        """获取模型显示名称（支持自定义API）"""
        # 检查是否为自定义API模型
        if model_id.startswith("custom:"):
            parts = model_id.split(":", 2)
            if len(parts) == 3:
                _, provider_name, model_name = parts
                return f"🔧 {provider_name}: {model_name}"
            return model_id

        # 预定义模型
        model = model_config.get_model_by_id(model_id)
        return model.get_display_name() if model else model_id

    def get_current_config(self) -> Dict[str, Tuple[str, str]]:
        """
        获取当前配置

        Returns:
            Dict[str, Tuple[str, str]]: {'analysis': (model_id, api_key), 'generation': (model_id, api_key)}
        """
        config_method = st.session_state.api_config_method

        analysis_model_id = st.session_state.selected_analysis_model
        generation_model_id = st.session_state.selected_generation_model

        if config_method == APIConfigMethod.ENVIRONMENT:
            # 从环境变量获取（仅适用于预定义模型）
            analysis_model = model_config.get_model_by_id(analysis_model_id)
            generation_model = model_config.get_model_by_id(generation_model_id)

            analysis_api_key = os.getenv(analysis_model.get_env_key()) if analysis_model else ""
            generation_api_key = os.getenv(generation_model.get_env_key()) if generation_model else ""
        else:
            # 从会话状态获取或自定义API配置获取
            analysis_api_key = self._get_api_key_for_model(analysis_model_id, "analysis")
            generation_api_key = self._get_api_key_for_model(generation_model_id, "generation")

        return {
            'analysis': (analysis_model_id, analysis_api_key),
            'generation': (generation_model_id, generation_api_key)
        }

    def _get_api_key_for_model(self, model_id: str, config_type: str) -> str:
        """获取指定模型的API密钥"""
        # 检查是否为自定义API模型
        if model_id.startswith("custom:"):
            # 解析模型ID: custom:provider_name:model_name
            parts = model_id.split(":", 2)
            if len(parts) == 3:
                _, provider_name, model_name = parts
                custom_config = custom_api_manager.get_config_by_name(provider_name)
                if custom_config and custom_config.is_active:
                    return custom_config.api_key
            return ""

        # 预定义模型：从会话状态获取
        return st.session_state.session_api_keys.get(f"{config_type}_{model_id}", "")

    def is_config_complete(self) -> bool:
        """检查配置是否完整且连接测试通过"""
        analysis_connected = st.session_state.api_connection_status['analysis']['connected']
        generation_connected = st.session_state.api_connection_status['generation']['connected']
        return analysis_connected and generation_connected

    def render_connection_status_badge(self):
        """在主界面显示连接状态徽章"""
        analysis_connected = st.session_state.api_connection_status['analysis']['connected']
        generation_connected = st.session_state.api_connection_status['generation']['connected']

        if analysis_connected and generation_connected:
            st.success("🟢 AI模型连接正常")
        elif analysis_connected or generation_connected:
            st.warning("🟡 部分AI模型连接异常")
        else:
            st.error("🔴 AI模型未连接")


# 创建全局配置流程实例
model_config_flow = ModelConfigFlow()