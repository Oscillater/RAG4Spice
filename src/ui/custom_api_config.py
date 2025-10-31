"""
自定义API配置界面组件

提供类似Cherry Studio的自定义API配置界面：
1. 添加自定义API提供商
2. 自动发现模型
3. 测试API连接
4. 管理已配置的API
"""

import streamlit as st
from typing import Optional, List, Dict, Any
from config.custom_api import custom_api_manager, CustomAPIConfig
from config.settings import settings


class CustomAPIConfigUI:
    """自定义API配置界面组件"""

    def __init__(self):
        """初始化组件"""
        self._init_session_state()

    def _init_session_state(self):
        """初始化会话状态"""
        if 'custom_api_show_add_form' not in st.session_state:
            st.session_state.custom_api_show_add_form = False
        if 'custom_api_editing_config' not in st.session_state:
            st.session_state.custom_api_editing_config = None

    def render_config_page(self):
        """渲染配置页面"""
        st.title("🔧 自定义API配置")
        st.markdown("---")

        # 添加新API按钮
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ 添加自定义API", type="primary", use_container_width=True):
                st.session_state.custom_api_show_add_form = True
                st.session_state.custom_api_editing_config = None
                st.rerun()

        with col2:
            if st.button("🔄 刷新列表", use_container_width=True):
                st.rerun()

        # 显示添加/编辑表单
        if st.session_state.custom_api_show_add_form:
            self._render_add_edit_form()

        # 显示已配置的API列表
        self._render_config_list()

    def _render_add_edit_form(self):
        """渲染添加/编辑表单"""
        is_editing = st.session_state.custom_api_editing_config is not None

        with st.expander("📝 API配置表单", expanded=True):
            if is_editing:
                config = st.session_state.custom_api_editing_config
                st.subheader(f"✏️ 编辑 API: {config.provider_name}")
            else:
                st.subheader("➕ 添加新的自定义API")

            # 表单字段
            with st.form("custom_api_form"):
                col1, col2 = st.columns(2)

                with col1:
                    provider_name = st.text_input(
                        "提供商名称 *",
                        value=config.provider_name if is_editing else "",
                        help="为这个API配置起一个容易识别的名称",
                        disabled=is_editing  # 编辑时不允许修改名称
                    )

                    base_url = st.text_input(
                        "API基础URL *",
                        value=config.base_url if is_editing else "",
                        help="例如: https://api.openai.com/v1 或 http://localhost:11434/v1",
                        placeholder="https://api.example.com/v1"
                    )

                with col2:
                    api_key = st.text_input(
                        "API密钥 *",
                        value=config.mask_api_key() if is_editing else "",
                        type="password",
                        help="API访问密钥，将安全存储",
                        disabled=is_editing  # 编辑时显示掩码，需要重新输入
                    )

                    if is_editing:
                        st.info("🔒 编辑时需要重新输入API密钥")
                        new_api_key = st.text_input(
                            "新API密钥",
                            type="password",
                            help="留空则保持原密钥不变"
                        )
                    else:
                        new_api_key = api_key

                description = st.text_area(
                    "描述（可选）",
                    value=config.description if is_editing else "",
                    help="简单描述这个API的用途或特点"
                )

                # 按钮区域
                col_submit, col_cancel = st.columns([1, 1])

                with col_submit:
                    submit_button = st.form_submit_button(
                        "💾 保存配置" if not is_editing else "💾 更新配置",
                        type="primary",
                        use_container_width=True
                    )

                with col_cancel:
                    cancel_button = st.form_submit_button(
                        "❌ 取消",
                        use_container_width=True
                    )

                # 处理表单提交
                if submit_button:
                    self._handle_form_submit(
                        provider_name, base_url, new_api_key, description, is_editing
                    )

                if cancel_button:
                    st.session_state.custom_api_show_add_form = False
                    st.session_state.custom_api_editing_config = None
                    st.rerun()

    def _handle_form_submit(self, provider_name: str, base_url: str, api_key: str,
                           description: str, is_editing: bool):
        """处理表单提交"""
        try:
            # 验证必填字段
            if not provider_name.strip():
                st.error("❌ 提供商名称不能为空")
                return

            if not base_url.strip():
                st.error("❌ API基础URL不能为空")
                return

            if not api_key.strip() and not is_editing:
                st.error("❌ API密钥不能为空")
                return

            # 处理编辑时的API密钥
            if is_editing:
                config = st.session_state.custom_api_editing_config
                if not api_key.strip():  # 如果新密钥为空，使用原密钥
                    api_key = config.api_key

            # 创建配置对象
            if is_editing:
                # 更新现有配置
                updates = {
                    'base_url': base_url.strip(),
                    'api_key': api_key.strip(),
                    'description': description.strip(),
                    'test_status': '',  # 重置测试状态
                    'last_tested': None
                }

                success = custom_api_manager.update_config(provider_name, updates)
                if success:
                    st.success(f"✅ API配置 '{provider_name}' 更新成功！")
                    # 自动测试连接
                    self._auto_test_connection(provider_name)
                else:
                    st.error(f"❌ 更新API配置失败")

            else:
                # 创建新配置并自动发现模型
                success = custom_api_manager.create_config_from_url(
                    provider_name.strip(),
                    base_url.strip(),
                    api_key.strip(),
                    description.strip()
                )
                if success:
                    st.success(f"✅ API配置 '{provider_name}' 添加成功！")
                    # create_config_from_url 已经包含了测试连接和模型发现
                else:
                    st.error(f"❌ 添加API配置失败，可能是名称已存在或连接测试失败")

            # 重置表单状态
            st.session_state.custom_api_show_add_form = False
            st.session_state.custom_api_editing_config = None
            st.rerun()

        except Exception as e:
            st.error(f"❌ 操作失败: {str(e)}")

    def _auto_test_connection(self, provider_name: str):
        """自动测试API连接"""
        try:
            config = custom_api_manager.get_config_by_name(provider_name)
            if config:
                with st.spinner(f"🔄 正在测试 {provider_name} 的连接..."):
                    success, message = custom_api_manager.test_api_connection(config)
                    if success:
                        st.success(f"🎉 {provider_name} 连接测试成功，发现 {len(config.models)} 个模型")
                    else:
                        st.warning(f"⚠️ {provider_name} 连接测试失败: {message}")
        except Exception as e:
            st.error(f"❌ 自动测试连接失败: {str(e)}")

    def _render_config_list(self):
        """渲染配置列表"""
        configs = custom_api_manager.get_all_configs()

        if not configs:
            st.info("📭 暂无自定义API配置，点击上方按钮添加")
            return

        st.subheader("📋 已配置的API")

        # 按状态分组显示
        active_configs = [config for config in configs if config.is_active]
        inactive_configs = [config for config in configs if not config.is_active]

        if active_configs:
            st.write("### 🟢 启用的API")
            for config in active_configs:
                self._render_config_card(config)

        if inactive_configs:
            st.write("### 🔴 禁用的API")
            for config in inactive_configs:
                self._render_config_card(config)

    def _render_config_card(self, config: CustomAPIConfig):
        """渲染单个配置卡片"""
        with st.container():
            # 卡片容器
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                # 基本信息
                status_icon = "🟢" if config.is_active else "🔴"
                test_icon = "✅" if config.test_status == "success" else "❌" if config.test_status == "failed" else "❓"

                st.markdown(f"### {status_icon} {test_icon} {config.provider_name}")

                # 详细信息
                with st.expander("📋 详细信息", expanded=False):
                    st.write(f"**URL**: `{config.base_url}`")
                    st.write(f"**API密钥**: `{config.mask_api_key()}`")
                    st.write(f"**状态**: {'启用' if config.is_active else '禁用'}")
                    st.write(f"**模型数量**: {len(config.models)}")

                    if config.description:
                        st.write(f"**描述**: {config.description}")

                    if config.last_tested:
                        st.write(f"**最后测试**: {config.last_tested}")

                    if config.models:
                        st.write("**可用模型**:")
                        for model in config.models[:10]:  # 最多显示10个
                            st.write(f"- `{model}`")
                        if len(config.models) > 10:
                            st.write(f"... 还有 {len(config.models) - 10} 个模型")

            with col2:
                # 操作按钮
                if st.button("🧪 测试", key=f"test_{config.provider_name}", use_container_width=True):
                    self._test_connection(config)

                if st.button("🔄 刷新模型", key=f"refresh_{config.provider_name}", use_container_width=True):
                    self._refresh_models(config)

            with col3:
                # 管理按钮
                if st.button("✏️ 编辑", key=f"edit_{config.provider_name}", use_container_width=True):
                    st.session_state.custom_api_show_add_form = True
                    st.session_state.custom_api_editing_config = config
                    st.rerun()

                # 启用/禁用切换
                toggle_text = "🔴 禁用" if config.is_active else "🟢 启用"
                if st.button(toggle_text, key=f"toggle_{config.provider_name}", use_container_width=True):
                    custom_api_manager.update_config(config.provider_name, {
                        'is_active': not config.is_active
                    })
                    st.rerun()

                # 删除按钮
                if st.button("🗑️ 删除", key=f"delete_{config.provider_name}", use_container_width=True):
                    self._delete_config(config)

            st.divider()

    def _test_connection(self, config: CustomAPIConfig):
        """测试API连接"""
        with st.spinner(f"🔄 正在测试 {config.provider_name} 的连接..."):
            try:
                success, message = custom_api_manager.test_api_connection(config)
                if success:
                    st.success(f"🎉 {config.provider_name} 连接测试成功！")
                    st.info(f"发现 {len(config.models)} 个模型")
                else:
                    st.error(f"❌ {config.provider_name} 连接测试失败: {message}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 测试连接失败: {str(e)}")

    def _refresh_models(self, config: CustomAPIConfig):
        """刷新模型列表"""
        with st.spinner(f"🔄 正在刷新 {config.provider_name} 的模型列表..."):
            try:
                models = custom_api_manager.discover_models(config)
                if models:
                    st.success(f"🎉 成功刷新模型列表，发现 {len(models)} 个模型")
                else:
                    st.warning("⚠️ 未发现任何模型")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 刷新模型列表失败: {str(e)}")

    def _delete_config(self, config: CustomAPIConfig):
        """删除配置"""
        if st.session_state.get(f"confirm_delete_{config.provider_name}", False):
            # 确认删除
            success = custom_api_manager.delete_config(config.provider_name)
            if success:
                st.success(f"✅ 已删除 API 配置: {config.provider_name}")
            else:
                st.error(f"❌ 删除失败: {config.provider_name}")
            st.rerun()
        else:
            # 显示确认按钮
            st.session_state[f"confirm_delete_{config.provider_name}"] = True
            st.error(f"⚠️ 确认要删除 '{config.provider_name}' 吗？再次点击删除按钮确认。")
            st.rerun()

    def get_model_choices_for_ui(self) -> Dict[str, str]:
        """获取UI用的模型选择列表（包含自定义API）"""
        choices = {}

        # 获取自定义API的模型
        custom_models = custom_api_manager.get_all_available_models()
        for provider_name, models in custom_models.items():
            for model in models:
                key = f"custom:{provider_name}:{model}"
                choices[key] = f"🔧 {provider_name}: {model}"

        return choices

    def get_config_for_model(self, model_key: str) -> Optional[CustomAPIConfig]:
        """根据模型键获取配置"""
        if not model_key.startswith("custom:"):
            return None

        parts = model_key.split(":", 2)
        if len(parts) != 3:
            return None

        _, provider_name, model_id = parts
        return custom_api_manager.get_config_by_name(provider_name)

    def render_model_selector(self, key: str, help_text: str = "") -> Optional[str]:
        """渲染自定义API模型选择器"""
        choices = self.get_model_choices_for_ui()

        if not choices:
            st.info("📭 暂无自定义API模型，请先添加自定义API配置")
            return None

        return st.selectbox(
            "选择自定义API模型",
            options=list(choices.keys()),
            format_func=lambda x: choices[x],
            key=key,
            help=help_text
        )


# 创建全局UI组件实例
custom_api_config_ui = CustomAPIConfigUI()