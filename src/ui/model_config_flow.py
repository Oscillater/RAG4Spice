"""
简化的AI模型配置流程组件

完全基于网页输入的配置方式：
1. 官方模型自动配置URL，用户只需输入API密钥
2. 自定义模型用户配置URL和API密钥
3. 一次测试，状态共享
"""

import streamlit as st
from typing import Optional, Tuple, Dict
from config.models import model_config, AIModel
from config.settings import settings
from config.custom_api import custom_api_manager
from core.multi_llm import multi_llm_manager
from ui.custom_api_config import custom_api_config_ui


class SimplifiedModelConfigFlow:
    """简化的AI模型配置流程组件"""

    def __init__(self):
        """初始化模型配置流程组件"""
        self._init_session_state()

    def _init_session_state(self):
        """初始化会话状态"""
        # 模型配置状态
        if 'selected_analysis_model' not in st.session_state:
            st.session_state.selected_analysis_model = settings.DEFAULT_MODEL
        if 'selected_generation_model' not in st.session_state:
            st.session_state.selected_generation_model = settings.DEFAULT_MODEL

        # API连接状态 - 实现状态共享
        if 'api_connection_status' not in st.session_state:
            st.session_state.api_connection_status = {
                'analysis': {'connected': False, 'model': '', 'error': '', 'last_tested': ''},
                'generation': {'connected': False, 'model': '', 'error': '', 'last_tested': ''}
            }

        # 会话存储的API密钥
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {}

    def render_config_flow(self) -> bool:
        """
        渲染简化的配置流程

        Returns:
            bool: 是否配置完成且连接测试通过
        """
        st.subheader("🤖 AI模型配置")

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
            key="analysis_model_select"
        )
        st.session_state.selected_analysis_model = analysis_model_id

        # 显示模型信息和自动配置
        self._display_model_info_with_auto_config(analysis_model_id)

        # API密钥输入
        analysis_api_key = self._render_api_key_input(analysis_model_id, "analysis")

        # 代码生成模型选择
        st.write("#### 💻 代码生成模型")

        # 是否使用相同模型的选项
        use_same_model = st.checkbox("🔗 使用相同模型进行代码生成", value=True, key="use_same_model")

        if use_same_model:
            generation_model_id = analysis_model_id
            generation_api_key = analysis_api_key
            st.session_state.selected_generation_model = generation_model_id

            # 显示提示
            st.info("✅ 代码生成模型与分析模型相同，无需重复配置")
        else:
            generation_model_id = st.selectbox(
                "选择用于生成HSPICE代码的模型",
                options=list(all_choices.keys()),
                format_func=lambda x: all_choices[x],
                index=self._get_model_index(st.session_state.selected_generation_model, all_choices),
                key="generation_model_select"
            )
            st.session_state.selected_generation_model = generation_model_id

            # 显示模型信息和自动配置
            self._display_model_info_with_auto_config(generation_model_id)

            # API密钥输入
            generation_api_key = self._render_api_key_input(generation_model_id, "generation")

        # API连接测试 - 实现状态共享
        st.write("### 🧪 API连接测试")
        test_success = self._render_smart_connection_test(
            analysis_model_id, analysis_api_key,
            generation_model_id, generation_api_key,
            use_same_model
        )

        return test_success

    def _get_model_index(self, model_id: str, model_choices: Dict[str, str]) -> int:
        """获取模型在选择列表中的索引"""
        try:
            return list(model_choices.keys()).index(model_id)
        except ValueError:
            return 0

    def _display_model_info_with_auto_config(self, model_id: str):
        """显示模型信息和自动配置状态"""
        # 检查是否为自定义API模型
        if model_id.startswith("custom:"):
            self._display_custom_model_info(model_id)
            return

        model = model_config.get_model_by_id(model_id)
        if not model:
            return

        # 检查是否为官方模型并显示自动配置
        if model_config.is_official_model(model_id):
            auto_config = model_config.get_auto_config_for_model(model_id)
            if auto_config:
                st.success("✅ 官方模型 - 已自动配置参数")

                col1, col2 = st.columns(2)
                with col1:
                    if auto_config.get('base_url'):
                        st.info(f"🔗 API地址: `{auto_config['base_url']}`")

                with col2:
                    st.write(f"**最大Token**: {auto_config.get('max_tokens', 'N/A')}")
                    st.write(f"**温度**: {auto_config.get('temperature', 'N/A')}")
                    st.write(f"**中文优化**: {'✅' if auto_config.get('is_chinese') else '❌'}")
        else:
            st.info("🔧 自定义模型 - 请手动配置")

        # 显示基本模型信息
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

        st.info(f"🔧 自定义API - {provider_name}")

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
                status_icon = "✅" if custom_config.test_status == "success" else "❌" if custom_config.test_status == "failed" else "⏳"
                st.write(f"**测试状态**: {status_icon} {custom_config.test_status}")
                st.write(f"**最后测试**: {custom_config.last_tested}")

    def _render_api_key_input(self, model_id: str, config_type: str) -> str:
        """渲染API密钥输入"""
        # 检查是否为自定义API模型
        if model_id.startswith("custom:"):
            return self._render_custom_api_key_input(model_id, config_type)

        model = model_config.get_model_by_id(model_id)
        if not model:
            return ""

        session_key = f"{config_type}_{model_id}"
        api_key = ""

        # 检查会话状态中是否已保存
        if session_key in st.session_state.api_keys:
            saved_key = st.session_state.api_keys[session_key]
            api_key = saved_key
            st.text_input(
                "API密钥",
                value="*" * 20 + saved_key[-4:] if len(saved_key) > 4 else "*",
                type="password",
                key=f"api_key_saved_{config_type}_{model_id}",
                help="API密钥已保存在当前会话中"
            )

            if st.button(f"清除{model.display_name}的API密钥", key=f"clear_{config_type}_{model_id}"):
                del st.session_state.api_keys[session_key]
                st.rerun()
        else:
            # 简化的API密钥输入
            api_key = st.text_input(
                "API密钥",
                type="password",
                key=f"api_key_input_{config_type}_{model_id}",
                help=f"请输入 {model.provider.value.upper()} 的API密钥"
            )

            # 保存到会话状态
            if api_key and st.button(f"保存{model.display_name}的API密钥", key=f"save_{config_type}_{model_id}"):
                st.session_state.api_keys[session_key] = api_key
                st.success("API密钥已保存到当前会话")
                st.rerun()

        return api_key

    def _render_custom_api_key_input(self, model_id: str, config_type: str) -> str:
        """渲染自定义API密钥输入"""
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

    def _render_smart_connection_test(
        self,
        analysis_model_id: str, analysis_api_key: str,
        generation_model_id: str, generation_api_key: str,
        use_same_model: bool
    ) -> bool:
        """渲染智能连接测试 - 实现状态共享"""

        # 智能测试逻辑
        test_analysis = False
        test_generation = False

        # 确定需要测试的模型
        if use_same_model:
            # 使用相同模型，只需要测试一次
            if analysis_api_key:
                test_analysis = st.button(
                    "🧪 测试模型连接（分析+生成）",
                    key=f"test_shared_{analysis_model_id}",
                    use_container_width=True,
                    disabled=not analysis_api_key
                )
        else:
            # 使用不同模型，分别测试
            col1, col2 = st.columns(2)

            with col1:
                if analysis_api_key:
                    test_analysis = st.button(
                        "🧪 测试分析模型连接",
                        key=f"test_analysis_{analysis_model_id}",
                        use_container_width=True,
                        disabled=not analysis_api_key
                    )

            with col2:
                if generation_api_key:
                    test_generation = st.button(
                        "🧪 测试生成模型连接",
                        key=f"test_generation_{generation_model_id}",
                        use_container_width=True,
                        disabled=not generation_api_key
                    )

        # 执行测试
        if test_analysis and analysis_api_key:
            self._test_single_api_connection("analysis", analysis_model_id, analysis_api_key)

            # 如果使用相同模型，同时更新生成状态
            if use_same_model:
                st.session_state.api_connection_status['generation'] = st.session_state.api_connection_status['analysis'].copy()

        if test_generation and generation_api_key and not use_same_model:
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
                        model_id, api_key, test_prompt, max_retries=2
                    )

                if "连接成功" in response or "success" in response.lower():
                    st.success(f"✅ {config_type}模型连接测试成功！")
                    st.session_state.api_connection_status[config_type] = {
                        'connected': True,
                        'model': model_id,
                        'error': '',
                        'last_tested': '刚刚'
                    }
                else:
                    st.warning(f"⚠️ {config_type}模型连接成功，但响应异常: {response[:100]}...")
                    st.session_state.api_connection_status[config_type] = {
                        'connected': True,
                        'model': model_id,
                        'error': f"响应异常: {response[:50]}...",
                        'last_tested': '刚刚'
                    }

            except Exception as e:
                st.error(f"❌ {config_type}模型连接测试失败: {str(e)}")
                st.session_state.api_connection_status[config_type] = {
                    'connected': False,
                    'model': model_id,
                    'error': str(e),
                    'last_tested': '刚刚'
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
                st.write(f"最后测试: {analysis_status['last_tested']}")
            else:
                st.error("❌ 分析模型未连接")
                if analysis_status['error']:
                    st.code(f"错误: {analysis_status['error']}")
                st.write(f"最后测试: {analysis_status['last_tested']}")

        with col2:
            generation_status = st.session_state.api_connection_status['generation']
            if generation_status['connected']:
                st.success("✅ 生成模型已连接")
                model_display_name = self._get_model_display_name(generation_status['model'])
                if model_display_name:
                    st.write(f"模型: {model_display_name}")
                if generation_status['error']:
                    st.warning(f"注意: {generation_status['error']}")
                st.write(f"最后测试: {generation_status['last_tested']}")
            else:
                st.error("❌ 生成模型未连接")
                if generation_status['error']:
                    st.code(f"错误: {generation_status['error']}")
                st.write(f"最后测试: {generation_status['last_tested']}")

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
        analysis_model_id = st.session_state.selected_analysis_model
        generation_model_id = st.session_state.selected_generation_model

        analysis_api_key = st.session_state.api_keys.get(f"analysis_{analysis_model_id}", "")
        generation_api_key = st.session_state.api_keys.get(f"generation_{generation_model_id}", "")

        return {
            'analysis': (analysis_model_id, analysis_api_key),
            'generation': (generation_model_id, generation_api_key)
        }

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
model_config_flow = SimplifiedModelConfigFlow()