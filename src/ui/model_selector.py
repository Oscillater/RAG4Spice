"""
模型选择UI组件

提供用户友好的模型选择和API密钥输入界面。
"""

import os
import streamlit as st
from typing import Optional, Tuple, Dict

from config.models import model_config, AIModel
from config.settings import settings


class ModelSelectorComponent:
    """模型选择组件"""

    def __init__(self):
        """初始化模型选择组件"""
        self._init_session_state()

    def _init_session_state(self):
        """初始化会话状态"""
        if 'analysis_model' not in st.session_state:
            st.session_state.analysis_model = settings.DEFAULT_MODEL
        if 'generation_model' not in st.session_state:
            st.session_state.generation_model = settings.DEFAULT_MODEL
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {}
        if 'analysis_model_validated' not in st.session_state:
            st.session_state.analysis_model_validated = False
        if 'generation_model_validated' not in st.session_state:
            st.session_state.generation_model_validated = False

    def render_model_selection(self) -> Dict[str, Tuple[str, str]]:
        """
        渲染模型选择界面，支持分别配置任务分析和代码生成模型

        Returns:
            Dict[str, Tuple[str, str]]: {'analysis': (模型ID, API密钥), 'generation': (模型ID, API密钥)}
        """
        st.subheader("🤖 AI模型配置")

        result = {}

        # 任务分析模型配置
        with st.expander("📊 任务分析模型", expanded=True):
            st.write("用于分析实验要求并分解为具体任务")
            analysis_model_id, analysis_api_key = self._render_single_model_config(
                "analysis",
                st.session_state.analysis_model
            )
            st.session_state.analysis_model = analysis_model_id
            result['analysis'] = (analysis_model_id, analysis_api_key)

        # 代码生成模型配置
        with st.expander("💻 代码生成模型", expanded=True):
            st.write("用于生成具体的HSPICE仿真代码")
            generation_model_id, generation_api_key = self._render_single_model_config(
                "generation",
                st.session_state.generation_model
            )
            st.session_state.generation_model = generation_model_id
            result['generation'] = (generation_model_id, generation_api_key)

        # 快速同步设置
        self._render_sync_settings()

        return result

    def _render_single_model_config(self, config_type: str, current_model_id: str) -> Tuple[str, str]:
        """
        渲染单个模型配置

        Args:
            config_type: 配置类型 ('analysis' 或 'generation')
            current_model_id: 当前选中的模型ID

        Returns:
            Tuple[str, str]: (模型ID, API密钥)
        """
        model_choices = model_config.get_model_choices_for_ui()

        # 模型选择
        selected_model_id = st.selectbox(
            f"选择{config_type}模型",
            options=list(model_choices.keys()),
            format_func=lambda x: model_choices[x],
            index=list(model_choices.keys()).index(current_model_id) if current_model_id in model_choices else 0,
            key=f"{config_type}_model_selector"
        )

        # 显示模型信息
        with st.expander(f"📋 {model_config.get_model_by_id(selected_model_id).display_name} 详细信息", expanded=False):
            self._render_model_info(selected_model_id)

        # API密钥输入
        api_key = self._render_api_key_input_for_model(selected_model_id, config_type)

        # 验证配置
        self._validate_single_model_config(config_type, selected_model_id, api_key)

        return selected_model_id, api_key

    def _render_api_key_input_for_model(self, model_id: str, config_type: str) -> str:
        """为特定模型类型渲染API密钥输入"""
        model = model_config.get_model_by_id(model_id)
        if not model:
            return ""

        # 检查环境变量中是否已有密钥
        env_key = model.get_env_key()
        env_api_key = os.getenv(env_key)

        api_key = ""
        if env_api_key:
            st.success(f"✅ 已检测到环境变量 `{env_key}` 中的API密钥")
            api_key = env_api_key
            st.text_input(
                "API密钥 (已从环境变量加载)",
                value="*" * 20 + api_key[-4:] if len(api_key) > 4 else "*",
                type="password",
                disabled=True,
                key=f"{config_type}_api_key_env_{model_id}"
            )
        else:
            st.warning(f"⚠️ 未检测到环境变量 `{env_key}`，请手动输入API密钥")

            # 检查会话状态中是否已保存
            session_key = f"{config_type}_{model_id}"
            if session_key in st.session_state.api_keys:
                saved_key = st.session_state.api_keys[session_key]
                api_key = saved_key
                st.text_input(
                    "API密钥",
                    value="*" * 20 + saved_key[-4:] if len(saved_key) > 4 else "*",
                    type="password",
                    key=f"{config_type}_api_key_saved_{model_id}",
                    help="API密钥已保存在当前会话中"
                )

                if st.button(f"清除{model.display_name}的API密钥", key=f"{config_type}_clear_{model_id}"):
                    del st.session_state.api_keys[session_key]
                    st.rerun()
            else:
                api_key = st.text_input(
                    "API密钥",
                    type="password",
                    key=f"{config_type}_api_key_input_{model_id}",
                    help=f"请输入 {model.provider.value.upper()} 的API密钥"
                )

                # 保存到会话状态
                if api_key and st.button(f"保存{model.display_name}的API密钥", key=f"{config_type}_save_{model_id}"):
                    st.session_state.api_keys[session_key] = api_key
                    st.success("API密钥已保存到当前会话")
                    st.rerun()

        return api_key

    def _validate_single_model_config(self, config_type: str, model_id: str, api_key: str):
        """验证单个模型配置"""
        if not api_key:
            st.error(f"❌ 请配置{config_type}模型的API密钥")
            if config_type == 'analysis':
                st.session_state.analysis_model_validated = False
            else:
                st.session_state.generation_model_validated = False
            return

        # 测试API连接
        if st.button(f"🧪 测试{config_type}模型API连接", key=f"{config_type}_test_{model_id}"):
            with st.spinner(f"正在测试{config_type}模型API连接..."):
                try:
                    from core.multi_llm import multi_llm_manager

                    # 使用简单的测试提示
                    test_prompt = "请回复'连接成功'，不要其他内容。"
                    response = multi_llm_manager.generate_with_retry(
                        model_id, api_key, test_prompt, max_retries=1
                    )

                    if "连接成功" in response or "success" in response.lower():
                        st.success(f"✅ {config_type}模型API连接测试成功！")
                        if config_type == 'analysis':
                            st.session_state.analysis_model_validated = True
                        else:
                            st.session_state.generation_model_validated = True
                    else:
                        st.warning(f"⚠️ {config_type}模型API连接成功，但响应异常: {response[:100]}...")
                        if config_type == 'analysis':
                            st.session_state.analysis_model_validated = True
                        else:
                            st.session_state.generation_model_validated = True

                except Exception as e:
                    st.error(f"❌ {config_type}模型API连接测试失败: {str(e)}")
                    if config_type == 'analysis':
                        st.session_state.analysis_model_validated = False
                    else:
                        st.session_state.generation_model_validated = False

    def _render_sync_settings(self):
        """渲染同步设置选项"""
        st.write("**快速设置**")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📋 分析→生成", help="将分析模型设置同步到生成模型"):
                st.session_state.generation_model = st.session_state.analysis_model
                st.rerun()

        with col2:
            if st.button("💻 生成→分析", help="将生成模型设置同步到分析模型"):
                st.session_state.analysis_model = st.session_state.generation_model
                st.rerun()

        # 使用相同模型的选项
        if st.checkbox("🔗 使用相同模型", help="任务分析和代码生成使用相同的模型"):
            if st.session_state.analysis_model != st.session_state.generation_model:
                st.session_state.generation_model = st.session_state.analysis_model
                st.rerun()

    def _render_model_info(self, model_id: str):
        """渲染模型信息"""
        model = model_config.get_model_by_id(model_id)
        if not model:
            return

        with st.expander(f"📋 {model.display_name} 详细信息", expanded=False):
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

    def _render_api_key_input(self, model_id: str) -> str:
        """渲染API密钥输入"""
        model = model_config.get_model_by_id(model_id)
        if not model:
            return ""

        st.write("**API密钥配置**")

        # 检查环境变量中是否已有密钥
        env_key = model.get_env_key()
        env_api_key = os.getenv(env_key)

        api_key = ""
        if env_api_key:
            st.success(f"✅ 已检测到环境变量 `{env_key}` 中的API密钥")
            api_key = env_api_key
            st.text_input(
                "API密钥 (已从环境变量加载)",
                value="*" * 20 + api_key[-4:] if len(api_key) > 4 else "*",
                type="password",
                disabled=True,
                key=f"api_key_env_{model_id}"
            )
        else:
            st.warning(f"⚠️ 未检测到环境变量 `{env_key}`，请手动输入API密钥")

            # 检查会话状态中是否已保存
            if model_id in st.session_state.api_keys:
                saved_key = st.session_state.api_keys[model_id]
                api_key = saved_key
                st.text_input(
                    "API密钥",
                    value="*" * 20 + saved_key[-4:] if len(saved_key) > 4 else "*",
                    type="password",
                    key=f"api_key_saved_{model_id}",
                    help="API密钥已保存在当前会话中"
                )

                if st.button(f"清除{model.display_name}的API密钥", key=f"clear_{model_id}"):
                    del st.session_state.api_keys[model_id]
                    st.rerun()
            else:
                api_key = st.text_input(
                    "API密钥",
                    type="password",
                    key=f"api_key_input_{model_id}",
                    help=f"请输入 {model.provider.value.upper()} 的API密钥"
                )

                # 保存到会话状态
                if api_key and st.button(f"保存{model.display_name}的API密钥", key=f"save_{model_id}"):
                    st.session_state.api_keys[model_id] = api_key
                    st.success("API密钥已保存到当前会话")
                    st.rerun()

        return api_key

    def _validate_model_config(self, model_id: str, api_key: str):
        """验证模型配置"""
        if not api_key:
            st.error("❌ 请配置API密钥")
            st.session_state.model_config_validated = False
            return

        # 测试API连接
        if st.button("🧪 测试API连接", key=f"test_{model_id}"):
            with st.spinner("正在测试API连接..."):
                try:
                    from core.multi_llm import multi_llm_manager

                    # 使用简单的测试提示
                    test_prompt = "请回复'连接成功'，不要其他内容。"
                    response = multi_llm_manager.generate_with_retry(
                        model_id, api_key, test_prompt, max_retries=1
                    )

                    if "连接成功" in response or "success" in response.lower():
                        st.success("✅ API连接测试成功！")
                        st.session_state.model_config_validated = True
                    else:
                        st.warning(f"⚠️ API连接成功，但响应异常: {response[:100]}...")
                        st.session_state.model_config_validated = True

                except Exception as e:
                    st.error(f"❌ API连接测试失败: {str(e)}")
                    st.session_state.model_config_validated = False

    def render_quick_setup(self) -> bool:
        """
        渲染快速设置界面，支持同时配置分析和生成模型

        Returns:
            bool: 是否完成设置
        """
        st.subheader("⚡ 快速设置")

        # 推荐模型
        recommended_models = model_config.get_recommended_models()
        model_choices = model_config.get_model_choices_for_ui()

        # 只显示推荐模型
        recommended_choices = {k: v for k, v in model_choices.items() if k in recommended_models}

        col1, col2 = st.columns(2)

        with col1:
            st.write("**任务分析模型**")
            analysis_model = st.selectbox(
                "选择分析模型",
                options=list(recommended_choices.keys()),
                format_func=lambda x: recommended_choices[x],
                key="quick_analysis_model"
            )

        with col2:
            st.write("**代码生成模型**")
            generation_model = st.selectbox(
                "选择生成模型",
                options=list(recommended_choices.keys()),
                format_func=lambda x: recommended_choices[x],
                key="quick_generation_model"
            )

        # 使用相同模型的选项
        use_same_model = st.checkbox("🔗 分析和生成使用相同模型", key="quick_same_model")
        if use_same_model:
            generation_model = analysis_model

        # 显示API密钥设置建议
        analysis_model_obj = model_config.get_model_by_id(analysis_model)
        generation_model_obj = model_config.get_model_by_id(generation_model)

        if analysis_model_obj and generation_model_obj:
            with st.expander("🔑 API密钥设置指南", expanded=True):
                tab1, tab2 = st.tabs(["分析模型", "生成模型"])
                with tab1:
                    self._render_api_setup_guide(analysis_model_obj)
                with tab2:
                    if not use_same_model:
                        self._render_api_setup_guide(generation_model_obj)
                    else:
                        st.info("与分析模型相同")

        # API密钥输入和测试
        success = False

        # 分析模型API密钥
        analysis_env_key = analysis_model_obj.get_env_key() if analysis_model_obj else ""
        analysis_env_api_key = os.getenv(analysis_env_key) if analysis_env_key else ""

        st.write("**任务分析模型API密钥**")
        if analysis_env_api_key:
            st.success(f"✅ 已检测到环境变量 `{analysis_env_key}` 中的API密钥")
            analysis_api_key = analysis_env_api_key
        else:
            analysis_api_key = st.text_input(
                "分析模型API密钥",
                type="password",
                key="quick_analysis_api_key",
                help=f"请输入分析模型的API密钥"
            )

        # 生成模型API密钥
        generation_env_key = generation_model_obj.get_env_key() if generation_model_obj else ""
        generation_env_api_key = os.getenv(generation_env_key) if generation_env_key else ""

        if not use_same_model:
            st.write("**代码生成模型API密钥**")
            if generation_env_api_key:
                st.success(f"✅ 已检测到环境变量 `{generation_env_key}` 中的API密钥")
                generation_api_key = generation_env_api_key
            else:
                generation_api_key = st.text_input(
                    "生成模型API密钥",
                    type="password",
                    key="quick_generation_api_key",
                    help=f"请输入生成模型的API密钥"
                )
        else:
            generation_api_key = analysis_api_key

        # 测试和保存按钮
        if st.button("🚀 测试并保存配置", type="primary", use_container_width=True):
            with st.spinner("正在测试API连接..."):
                try:
                    from core.multi_llm import multi_llm_manager
                    test_prompt = "请回复'连接成功'，不要其他内容。"

                    # 测试分析模型
                    if analysis_api_key:
                        try:
                            analysis_response = multi_llm_manager.generate_with_retry(
                                analysis_model, analysis_api_key, test_prompt, max_retries=1
                            )
                            st.success(f"✅ 分析模型连接成功")
                            analysis_success = True
                        except Exception as e:
                            st.error(f"❌ 分析模型连接失败: {str(e)}")
                            analysis_success = False
                    else:
                        st.error("❌ 请提供分析模型API密钥")
                        analysis_success = False

                    # 测试生成模型
                    if generation_api_key and (not use_same_model or generation_model != analysis_model):
                        try:
                            generation_response = multi_llm_manager.generate_with_retry(
                                generation_model, generation_api_key, test_prompt, max_retries=1
                            )
                            st.success(f"✅ 生成模型连接成功")
                            generation_success = True
                        except Exception as e:
                            st.error(f"❌ 生成模型连接失败: {str(e)}")
                            generation_success = False
                    elif use_same_model:
                        generation_success = analysis_success
                    else:
                        st.error("❌ 请提供生成模型API密钥")
                        generation_success = False

                    # 如果都成功，保存配置
                    if analysis_success and generation_success:
                        # 保存分析模型配置
                        st.session_state.analysis_model = analysis_model
                        if not analysis_env_api_key:
                            st.session_state.api_keys[f"analysis_{analysis_model}"] = analysis_api_key
                        st.session_state.analysis_model_validated = True

                        # 保存生成模型配置
                        st.session_state.generation_model = generation_model
                        if not use_same_model and not generation_env_api_key:
                            st.session_state.api_keys[f"generation_{generation_model}"] = generation_api_key
                        st.session_state.generation_model_validated = True

                        st.success("✅ 所有模型配置已保存并测试成功！")
                        success = True

                except Exception as e:
                    st.error(f"❌ 配置测试失败: {str(e)}")

        return success

    def _render_api_setup_guide(self, model: AIModel):
        """渲染API设置指南"""
        guides = {
            ModelProvider.GOOGLE: {
                "步骤": [
                    "1. 访问 [Google AI Studio](https://aistudio.google.com/)",
                    "2. 使用Google账号登录",
                    "3. 点击 'Create API Key' 创建API密钥",
                    "4. 复制API密钥并设置环境变量 `GOOGLE_API_KEY`"
                ]
            },
            ModelProvider.OPENAI: {
                "步骤": [
                    "1. 访问 [OpenAI Platform](https://platform.openai.com/)",
                    "2. 注册并登录账号",
                    "3. 进入 'API Keys' 页面",
                    "4. 点击 'Create new secret key' 创建密钥",
                    "5. 复制API密钥并设置环境变量 `OPENAI_API_KEY`"
                ]
            },
            ModelProvider.ANTHROPIC: {
                "步骤": [
                    "1. 访问 [Anthropic Console](https://console.anthropic.com/)",
                    "2. 注册并登录账号",
                    "3. 进入 'API Keys' 页面",
                    "4. 点击 'Create Key' 创建密钥",
                    "5. 复制API密钥并设置环境变量 `ANTHROPIC_API_KEY`"
                ]
            },
            ModelProvider.ALIBABA: {
                "步骤": [
                    "1. 访问 [阿里云百炼平台](https://bailian.console.aliyun.com/)",
                    "2. 注册并登录阿里云账号",
                    "3. 开通DashScope服务",
                    "4. 在 'API-KEY管理' 中创建密钥",
                    "5. 复制API密钥并设置环境变量 `DASHSCOPE_API_KEY`"
                ]
            },
            ModelProvider.BAIDU: {
                "步骤": [
                    "1. 访问 [百度智能云](https://cloud.baidu.com/)",
                    "2. 注册并登录百度账号",
                    "3. 开通千帆大模型平台服务",
                    "4. 在应用管理中创建应用获取API Key和Secret Key",
                    "5. 设置环境变量 `BAIDU_API_KEY` 和 `BAIDU_SECRET_KEY`"
                ]
            },
            ModelProvider.ZHIPU: {
                "步骤": [
                    "1. 访问 [智谱AI开放平台](https://open.bigmodel.cn/)",
                    "2. 注册并登录账号",
                    "3. 进入 'API密钥' 页面",
                    "4. 点击 '创建API密钥'",
                    "5. 复制API密钥并设置环境变量 `ZHIPUAI_API_KEY`"
                ]
            },
            ModelProvider.MOONSHOT: {
                "步骤": [
                    "1. 访问 [月之暗面开放平台](https://platform.moonshot.cn/)",
                    "2. 注册并登录账号",
                    "3. 进入 'API Keys' 页面",
                    "4. 点击 'Create API Key' 创建密钥",
                    "5. 复制API密钥并设置环境变量 `MOONSHOT_API_KEY`"
                ]
            },
            ModelProvider.DEEPSEEK: {
                "步骤": [
                    "1. 访问 [DeepSeek平台](https://platform.deepseek.com/)",
                    "2. 注册并登录账号",
                    "3. 进入 'API Keys' 页面",
                    "4. 点击 'Create API Key' 创建密钥",
                    "5. 复制API密钥并设置环境变量 `DEEPSEEK_API_KEY`"
                ]
            },
            ModelProvider.MISTRAL: {
                "步骤": [
                    "1. 访问 [Mistral AI](https://console.mistral.ai/)",
                    "2. 注册并登录账号",
                    "3. 进入 'API Keys' 页面",
                    "4. 点击 'Create API Key' 创建密钥",
                    "5. 复制API密钥并设置环境变量 `MISTRAL_API_KEY`"
                ]
            },
            ModelProvider.COHERE: {
                "步骤": [
                    "1. 访问 [Cohere Dashboard](https://dashboard.cohere.com/)",
                    "2. 注册并登录账号",
                    "3. 进入 'API Keys' 页面",
                    "4. 点击 'Create API Key' 创建密钥",
                    "5. 复制API密钥并设置环境变量 `COHERE_API_KEY`"
                ]
            },
        }

        guide = guides.get(model.provider)
        if guide:
            st.write(f"**{model.provider.value.upper()} API密钥设置步骤：**")
            for step in guide["步骤"]:
                st.write(step)

            st.info(f"💡 **提示**: 设置环境变量后重启应用，或直接在上方输入框中输入API密钥")

    def is_config_valid(self) -> bool:
        """检查当前配置是否有效"""
        analysis_valid = st.session_state.get('analysis_model_validated', False)
        generation_valid = st.session_state.get('generation_model_validated', False)
        return analysis_valid and generation_valid

    def is_analysis_config_valid(self) -> bool:
        """检查任务分析模型配置是否有效"""
        return st.session_state.get('analysis_model_validated', False)

    def is_generation_config_valid(self) -> bool:
        """检查代码生成模型配置是否有效"""
        return st.session_state.get('generation_model_validated', False)

    def get_analysis_config(self) -> Tuple[str, str]:
        """获取任务分析模型配置"""
        model_id = st.session_state.get('analysis_model', settings.DEFAULT_MODEL)
        api_key = self._get_api_key_for_model(model_id, 'analysis')
        return model_id, api_key

    def get_generation_config(self) -> Tuple[str, str]:
        """获取代码生成模型配置"""
        model_id = st.session_state.get('generation_model', settings.DEFAULT_MODEL)
        api_key = self._get_api_key_for_model(model_id, 'generation')
        return model_id, api_key

    def _get_api_key_for_model(self, model_id: str, config_type: str) -> str:
        """获取指定模型的API密钥"""
        model = model_config.get_model_by_id(model_id)
        if not model:
            return ""

        # 优先从环境变量获取
        env_key = model.get_env_key()
        env_api_key = os.getenv(env_key)
        if env_api_key:
            return env_api_key

        # 从会话状态获取
        session_key = f"{config_type}_{model_id}"
        return st.session_state.api_keys.get(session_key, "")

    def get_current_config(self) -> Tuple[str, str]:
        """获取当前配置（向后兼容）"""
        return self.get_analysis_config()