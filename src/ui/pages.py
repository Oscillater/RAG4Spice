"""
页面逻辑模块

定义各个页面的业务逻辑和流程控制。
"""

import streamlit as st
import os
from typing import Optional, Dict, Any

from models.task_models import TaskAnalysis, Task, GenerationResult
# from core.llm import analyze_tasks  # 已迁移到multi_llm
from core.retrieval import generate_task_code, retrieval_manager
from core.multi_llm import multi_llm_manager
from ui.components import (
    FileUploadComponent, TaskAnalysisComponent, TaskEditComponent,
    GenerationResultComponent, ErrorDisplayComponent, SuccessDisplayComponent
)
from ui.model_selector import ModelSelectorComponent
from ui.model_config_flow import model_config_flow
from ui.custom_api_config import custom_api_config_ui
from config.settings import settings
from config.custom_api import custom_api_manager
from utils.validators import ValidationError


class MainPage:
    """主页面类"""

    def __init__(self):
        """初始化主页面"""
        self._init_session_state()

    def _init_session_state(self):
        """初始化会话状态"""
        if 'task_analysis' not in st.session_state:
            st.session_state.task_analysis = None
        if 'last_prompt' not in st.session_state:
            st.session_state.last_prompt = ""
        if 'last_response' not in st.session_state:
            st.session_state.last_response = ""
        if 'generation_results' not in st.session_state:
            st.session_state.generation_results = []

        # 确保API连接状态已初始化（避免在配置流程之前访问时报错）
        if 'api_connection_status' not in st.session_state:
            st.session_state.api_connection_status = {
                'analysis': {'connected': False, 'model': '', 'error': ''},
                'generation': {'connected': False, 'model': '', 'error': ''}
            }

        # 确保API配置方法已初始化
        if 'api_config_method' not in st.session_state:
            from ui.model_config_flow import APIConfigMethod
            st.session_state.api_config_method = APIConfigMethod.ENVIRONMENT

        # 确保模型相关状态已初始化
        if 'analysis_model' not in st.session_state:
            st.session_state.analysis_model = None
        if 'generation_model' not in st.session_state:
            st.session_state.generation_model = None
        if 'selected_analysis_model' not in st.session_state:
            st.session_state.selected_analysis_model = settings.DEFAULT_MODEL
        if 'selected_generation_model' not in st.session_state:
            st.session_state.selected_generation_model = settings.DEFAULT_MODEL
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {}
        if 'analysis_model_validated' not in st.session_state:
            st.session_state.analysis_model_validated = False
        if 'generation_model_validated' not in st.session_state:
            st.session_state.generation_model_validated = False

        # 确保自定义API配置状态已初始化
        if 'custom_api_show_add_form' not in st.session_state:
            st.session_state.custom_api_show_add_form = False
        if 'custom_api_editing_config' not in st.session_state:
            st.session_state.custom_api_editing_config = None

        # 初始化模型选择器
        self.model_selector = ModelSelectorComponent()

    def _render_config_status_warnings(self):
        """渲染配置状态警告"""
        validation_status = settings.get_validation_status()

        # Tesseract警告
        if not validation_status["tesseract"]:
            st.warning("⚠️ **Tesseract OCR未配置**")
            st.info("💡 图片识别功能需要Tesseract OCR。请安装后设置环境变量 `TESSERACT_CMD`")
        else:
            st.success("✅ Tesseract OCR已配置")

        # API密钥信息（不再是警告，只是提示）
        if validation_status["has_any_api_key"]:
            configured_count = sum(1 for key in validation_status.get("api_keys", []) if key["has_key"])
            if configured_count > 0:
                st.success(f"✅ 已在环境变量中配置 {configured_count} 个AI模型")
        else:
            st.info("💡 **AI模型配置提示**")
            st.write("未在环境变量中检测到API密钥，您可以通过侧边栏配置任何支持的AI模型。")
            st.write("🔧 **支持的模型提供商包括：**")
            st.write("- Google Gemini, OpenAI, Anthropic Claude")
            st.write("- 阿里云通义千问, 百度文心一言, 智谱清言")
            st.write("- 月之暗面Kimi, DeepSeek, Mistral AI, Cohere")
            st.info("👉 请在侧边栏完成AI模型配置后开始使用")

    def _render_api_connection_status(self):
        """渲染API连接状态"""
        # 使用状态容器来显示连接状态
        with st.container():
            # 在页面顶部显示连接状态徽章
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.write("")  # 空列用于布局

            with col2:
                if model_config_flow.is_config_complete():
                    st.success("🟢 AI模型连接正常")
                else:
                    # 检查部分连接状态
                    analysis_connected = st.session_state.api_connection_status['analysis']['connected']
                    generation_connected = st.session_state.api_connection_status['generation']['connected']
                    if analysis_connected or generation_connected:
                        st.warning("🟡 部分AI模型连接异常")
                    else:
                        st.error("🔴 AI模型未连接")

            with col3:
                # 显示自定义API状态和重新配置按钮
                custom_configs = custom_api_manager.get_active_configs()
                if custom_configs:
                    st.info(f"🔧 {len(custom_configs)} 个自定义API")
                else:
                    st.write("")  # 空列用于布局

                if st.button("🔄 重新配置", help="重新配置AI模型", key="main_reconfigure"):
                    # 重置配置状态
                    st.session_state.api_connection_status = {
                        'analysis': {'connected': False, 'model': '', 'error': ''},
                        'generation': {'connected': False, 'model': '', 'error': ''}
                    }
                    st.rerun()

            # 显示详细连接信息（可折叠）
            if not model_config_flow.is_config_complete():
                with st.expander("📊 查看详细连接状态", expanded=False):
                    model_config_flow._display_connection_status()

    def render(self):
        """渲染主页面"""
        # 设置页面标题
        st.title("🤖 HSPICE RAG 代码生成助手")
        st.caption("上传实验截图，分析任务，生成HSPICE代码")

        # 显示配置状态提示
        self._render_config_status_warnings()

        # 显示API连接状态（所有模式下都显示）
        self._render_api_connection_status()

        # 模型配置部分（始终显示配置流程）
        with st.sidebar:
            # 添加自定义API配置入口
            st.markdown("---")
            if st.button("🔧 自定义API配置", help="配置自定义大模型API（类似Cherry Studio）",
                     use_container_width=True, type="secondary"):
                st.session_state.show_custom_api_config = True
                st.rerun()

            # 如果点击了自定义API配置，显示配置页面
            if st.session_state.get('show_custom_api_config', False):
                custom_api_config_ui.render_config_page()
                if st.button("🔙 返回主配置", key="back_to_main_config"):
                    st.session_state.show_custom_api_config = False
                    st.rerun()
                return

            # 使用新的配置流程
            config_complete = model_config_flow.render_config_flow()

            if not config_complete:
                st.warning("⚠️ 请完成AI模型配置和连接测试")
                st.info("💡 配置完成后系统将自动刷新页面")
                return

            # 获取模型配置
            model_configs = model_config_flow.get_current_config()
            analysis_model_id, analysis_api_key = model_configs['analysis']
            generation_model_id, generation_api_key = model_configs['generation']

            # 保存当前模型配置到会话状态
            st.session_state.analysis_model_id = analysis_model_id
            st.session_state.analysis_api_key = analysis_api_key
            st.session_state.generation_model_id = generation_model_id
            st.session_state.generation_api_key = generation_api_key

        # 第一部分：文件上传和文本提取
        self._render_file_upload_section()

        # 添加分隔线
        st.divider()

        # 第二部分：任务编辑与代码生成
        self._render_task_edit_section()

    def _render_file_upload_section(self):
        """渲染文件上传部分"""
        st.subheader("1. 上传实验要求文件")

        # 文件上传
        upload_component = FileUploadComponent()
        uploaded_file = upload_component.render_file_upload(
            "选择包含实验要求的文件",
            ["png", "jpg", "jpeg", "pdf"],
            "支持图片文件用于OCR识别，或PDF文件直接提取文本"
        )

        extracted_text = ""
        if uploaded_file:
            try:
                extracted_text = upload_component.display_uploaded_file(uploaded_file)

                # 创建一个 TaskAnalysisComponent 实例来调用其 render_extracted_text 方法
                analysis_component = TaskAnalysisComponent()
                analysis_component.render_extracted_text(extracted_text)

            except Exception as e:
                ErrorDisplayComponent.render_error("文件处理失败", e)

        # 任务分析
        if extracted_text:
            self._handle_task_analysis(extracted_text)

    def _handle_task_analysis(self, extracted_text: str):
        """处理任务分析"""
        analysis_component = TaskAnalysisComponent()

        if analysis_component.render_analyze_button(extracted_text):
            with st.spinner("AI分析任务中..."):
                try:
                    # 获取模型配置
                    model_configs = model_config_flow.get_current_config()
                    analysis_model_id, analysis_api_key = model_configs['analysis']

                    # 执行任务分析
                    task_analysis_dict = multi_llm_manager.analyze_tasks(
                        analysis_model_id, analysis_api_key, extracted_text
                    )
                    task_analysis_obj = TaskAnalysis.from_dict(task_analysis_dict)
                    # 保存分析结果
                    st.session_state.task_analysis = task_analysis_obj

                    # 保存调试信息
                    st.session_state.last_prompt = extracted_text
                    st.session_state.last_response = str(task_analysis_dict)

                    # 显示分析结果
                    analysis_component.render_task_analysis_result(task_analysis_obj)

                    # 显示调试信息
                    if hasattr(st.session_state, 'last_prompt') and hasattr(st.session_state, 'last_response'):
                        analysis_component.render_debug_info(
                            st.session_state.last_prompt,
                            st.session_state.last_response
                        )

                except ValidationError as e:
                    ErrorDisplayComponent.render_validation_error(e)
                except Exception as e:
                    ErrorDisplayComponent.render_error("任务分析失败", e)

                    # 显示调试信息
                    if hasattr(st.session_state, 'last_prompt'):
                        analysis_component.render_debug_info(
                            st.session_state.last_prompt,
                            getattr(st.session_state, 'last_response', '无响应数据')
                        )

    def _render_task_edit_section(self):
        """渲染任务编辑部分"""
        st.subheader("2. 任务编辑与代码生成")

        if st.session_state.task_analysis is None:
            SuccessDisplayComponent.render_info("请先上传文件并进行任务分析")
            return

        task_analysis = st.session_state.task_analysis

        # 显示分析结果
        analysis_component = TaskAnalysisComponent()
        analysis_component.render_task_analysis_result(task_analysis)

        # 编辑总体描述
        edit_component = TaskEditComponent()
        # 处理字典或TaskAnalysis对象
        if hasattr(task_analysis, 'general_description'):
            general_description_value = task_analysis.general_description
        elif isinstance(task_analysis, dict):
            general_description_value = task_analysis.get('general_description', '')
        else:
            general_description_value = ''

        general_description = edit_component.render_general_description_edit(
            general_description_value
        )

        # 编辑任务列表
        # 处理字典或TaskAnalysis对象
        if hasattr(task_analysis, 'tasks'):
            tasks_value = task_analysis.tasks
        elif isinstance(task_analysis, dict):
            tasks_value = task_analysis.get('tasks', [])
        else:
            tasks_value = []

        tasks = edit_component.render_task_list(tasks_value)

        # 更新session state中的任务列表
        current_analysis = st.session_state.task_analysis
        if hasattr(current_analysis, 'tasks'):
            current_analysis.tasks = tasks
        elif isinstance(current_analysis, dict):
            current_analysis['tasks'] = [task.to_dict() for task in tasks]

        # 检查是否有生成请求
        self._check_generation_requests(tasks)

        # 显示生成结果
        self._render_generation_results()

        # 处理添加任务
        if tasks:
            if edit_component.render_add_task_button():
                new_task = Task(
                    id=len(tasks) + 1,
                    title=f"任务{len(tasks) + 1}.sp",
                    description="请在此输入任务描述",
                    additional_info="",
                    knowledge="",
                    generate_request=False
                )
                tasks.append(new_task)
                # 确保session_state中的task_analysis是对象
                current_analysis = st.session_state.task_analysis
                if hasattr(current_analysis, 'tasks'):
                    current_analysis.tasks = tasks
                elif isinstance(current_analysis, dict):
                    current_analysis['tasks'] = tasks
                else:
                    # 如果是其他类型，创建新的TaskAnalysis对象
                    from models.task_models import TaskAnalysis
                    if isinstance(current_analysis, dict):
                        st.session_state.task_analysis = TaskAnalysis.from_dict(current_analysis)
                    else:
                        st.session_state.task_analysis = TaskAnalysis(
                            general_description='',
                            tasks=tasks
                        )
                st.rerun()

    def _render_generation_results(self):
        """渲染生成结果"""
        print(f"渲染生成结果，当前数量: {len(st.session_state.generation_results)}")
        if not st.session_state.generation_results:
            print("没有生成结果，直接返回")
            return

        st.divider()
        st.subheader("🎉 HSPICE代码生成结果")

        # 显示所有生成结果
        for index, result in enumerate(st.session_state.generation_results):
            print(f"渲染结果 {index}: {result.title}, success: {result.success}")
            GenerationResultComponent.render_generation_result(result, index)

    def _check_generation_requests(self, tasks: list):
        """检查代码生成请求"""
        if not tasks:
            return

        # 获取模型配置
        model_configs = model_config_flow.get_current_config()
        generation_model_id, generation_api_key = model_configs['generation']

        # 检查每个任务的生成请求
        for task in tasks:
            if task.generate_request:
                self._generate_single_task_code(task, generation_model_id, generation_api_key)

    def _generate_single_task_code(self, task: Task, generation_model_id: str, generation_api_key: str):
        """生成单个任务的HSPICE代码"""
        with st.spinner(f"正在生成 {task.title} 的HSPICE代码..."):
            try:
                # 获取检索知识
                documents = retrieval_manager.retrieve_knowledge(
                    task.title + " " + task.description
                )
                context = retrieval_manager.format_retrieved_documents(documents)

                # 获取任务分析结果
                task_analysis = st.session_state.task_analysis

                # 处理字典或TaskAnalysis对象，获取general_description
                if hasattr(task_analysis, 'general_description'):
                    general_description_value = task_analysis.general_description
                elif isinstance(task_analysis, dict):
                    general_description_value = task_analysis.get('general_description', '')
                else:
                    general_description_value = ''

                # 获取任务知识信息
                task_knowledge = ""
                if hasattr(task, 'knowledge'):
                    task_knowledge = task.knowledge
                elif isinstance(task, dict):
                    task_knowledge = task.get('knowledge', '')

                # 获取补充信息
                additional_info = ""
                if hasattr(task, 'additional_info'):
                    additional_info = task.additional_info
                elif isinstance(task, dict):
                    additional_info = task.get('additional_info', '')

                # 生成HSPICE代码
                analysis, hspice_code = multi_llm_manager.generate_hspice_code(
                    generation_model_id,
                    generation_api_key,
                    context,
                    general_description_value,
                    additional_info,  # 补充信息
                    task.description,
                    task.title,
                    task_knowledge
                )

                # 创建生成结果
                generation_result = GenerationResult(
                    task_id=task.id,
                    title=task.title,
                    description=task.description,
                    analysis=analysis,
                    hspice_code=hspice_code
                )

                print(f"创建生成结果: {generation_result.title}")
                print(f"分析长度: {len(analysis)}, 代码长度: {len(hspice_code)}")
                print(f"当前generation_results数量: {len(st.session_state.generation_results)}")

                # 保存生成结果
                st.session_state.generation_results.append(generation_result)
                print(f"添加后generation_results数量: {len(st.session_state.generation_results)}")

                # 重置生成请求标志
                task.generate_request = False

                # 更新session state中的任务
                current_analysis = st.session_state.task_analysis
                if hasattr(current_analysis, 'tasks'):
                    for i, t in enumerate(current_analysis.tasks):
                        if t.id == task.id:
                            current_analysis.tasks[i] = task
                            break
                elif isinstance(current_analysis, dict):
                    for i, t in enumerate(current_analysis['tasks']):
                        if t['id'] == task.id:
                            current_analysis['tasks'][i] = task.to_dict()
                            break

                # 显示成功消息
                SuccessDisplayComponent.render_success(f"成功生成 {task.title} 的HSPICE代码")

            except Exception as e:
                ErrorDisplayComponent.render_error(f"生成 {task.title} 的HSPICE代码失败", e)


# 便捷函数
def render_main_page():
    """渲染主页面"""
    main_page = MainPage()
    main_page.render()


def render_page_with_sidebar():
    """渲染带侧边栏的页面"""
    with st.sidebar:
        st.title("🧭 导航")
        page_selection = st.selectbox(
            "选择页面",
            ["主页", "任务分析", "设置"],
            index=0
        )

        page_map = {
            "主页": "main",
            "任务分析": "task_analysis",
            "设置": "settings"
        }

        selected_page = page_map.get(page_selection, "main")

    # 渲染选中的页面
    PageRouter.render_page(selected_page)


if __name__ == "__main__":
    MainPage().render()