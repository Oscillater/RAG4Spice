"""
页面逻辑模块

定义各个页面的业务逻辑和流程控制。
"""

import streamlit as st
from typing import Optional, Dict, Any

from ..models.task_models import TaskAnalysis, Task, GenerationResult
from ..core.llm import analyze_tasks
from ..core.retrieval import generate_task_code
from ..ui.components import (
    FileUploadComponent, TaskAnalysisComponent, TaskEditComponent,
    GenerationResultComponent, ErrorDisplayComponent, SuccessDisplayComponent
)
from ..utils.validators import ValidationError


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

    def render(self):
        """渲染主页面"""
        # 设置页面标题
        st.title("🤖 HSPICE RAG 代码生成助手")
        st.caption("上传实验截图，分析任务，生成HSPICE代码")

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
                
                # ===============================================================
                # START: 核心修改部分
                # ---------------------------------------------------------------
                # 错误代码 (已注释掉):
                # upload_component.render_extracted_text(extracted_text)
                
                # 正确代码:
                # 创建一个 TaskAnalysisComponent 实例来调用其 render_extracted_text 方法
                analysis_component = TaskAnalysisComponent()
                analysis_component.render_extracted_text(extracted_text)
                # ---------------------------------------------------------------
                # END: 核心修改部分
                # ===============================================================

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
                    # 验证输入
                    if not extracted_text.strip():
                        st.warning("OCR结果为空，请检查文件或重新上传")
                        return

                    # 执行任务分析
                    task_analysis_dict = analyze_tasks(extracted_text)

                    # 转换为TaskAnalysis对象
                    task_analysis = TaskAnalysis.from_dict(task_analysis_dict)

                    # 保存到会话状态
                    st.session_state.task_analysis = task_analysis
                    SuccessDisplayComponent.render_success("任务分析完成！")

                    # 显示分析结果
                    analysis_component.render_task_analysis_result(task_analysis)

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
        general_description = edit_component.render_general_description_edit(
            task_analysis.general_description
        )

        # 编辑任务列表
        tasks = edit_component.render_task_list(task_analysis.tasks)

        # 检查是否有生成请求
        self._check_generation_requests(tasks)

        # 处理添加任务
        if tasks:
            if edit_component.render_add_task_button():
                new_task_id = task_analysis.get_next_task_id()
                new_task = Task(
                    id=new_task_id,
                    title=f"任务{new_task_id}.sp",
                    description="新添加的HSPICE仿真任务描述"
                )
                task_analysis.add_task(new_task)
                st.session_state.task_analysis = task_analysis
                st.rerun()
        else:
            if edit_component.render_add_first_task_button():
                new_task = Task(
                    id=1,
                    title="任务1.sp",
                    description="新添加的HSPICE仿真任务描述"
                )
                task_analysis.add_task(new_task)
                st.session_state.task_analysis = task_analysis
                st.rerun()

        # 保存编辑结果
        if edit_component.render_save_button():
            task_analysis.general_description = general_description
            task_analysis.tasks = tasks
            st.session_state.task_analysis = task_analysis
            SuccessDisplayComponent.render_success("任务分析已更新")

    def _check_generation_requests(self, tasks: list):
        """检查并处理代码生成请求"""
        edit_component = TaskEditComponent()
        result_component = GenerationResultComponent()

        for i, task in enumerate(tasks):
            # 检查是否有生成请求
            generate_key = f"generate_task_{i}_data"
            if generate_key in st.session_state:
                task_to_generate = st.session_state[generate_key]
                del st.session_state[generate_key]  # 清除请求标记

                # 生成代码
                self._generate_code_for_task(task_to_generate)

    def _generate_code_for_task(self, task: Task):
        """为单个任务生成代码"""
        with st.spinner(f"正在生成 {task.title} 的HSPICE代码..."):
            try:
                # 获取总体描述
                general_description = st.session_state.task_analysis.general_description

                # 生成代码
                result_dict = generate_task_code(
                    task=task.to_dict(),
                    general_description=general_description,
                    visual_info=task.visual_info
                )

                # 创建结果对象
                result = GenerationResult(
                    task_id=result_dict["task_id"],
                    title=result_dict["title"],
                    description=result_dict["description"],
                    analysis=result_dict.get("analysis", ""),
                    hspice_code=result_dict.get("hspice_code", ""),
                    error=result_dict.get("error", ""),
                    success=result_dict.get("success", True)
                )

                # 显示结果
                result_component = GenerationResultComponent() # 修正：这里也需要实例化
                result_component.render_generation_result(result)

                # 保存结果到会话状态
                if 'generation_results' not in st.session_state:
                    st.session_state.generation_results = []
                st.session_state.generation_results.append(result)

            except Exception as e:
                ErrorDisplayComponent.render_error(f"调用AI失败", e)


class TaskAnalysisPage:
    """任务分析页面（备用）"""

    @staticmethod
    def render():
        """渲染任务分析页面"""
        st.title("📋 任务分析")
        st.write("此页面用于单独的任务分析功能")


class SettingsPage:
    """设置页面（备用）"""

    @staticmethod
    def render():
        """渲染设置页面"""
        st.title("⚙️ 设置")
        st.write("此页面用于系统配置")


class PageRouter:
    """页面路由器"""

    @staticmethod
    def render_page(page_name: str = "main"):
        """
        根据页面名称渲染对应页面

        Args:
            page_name: 页面名称
        """
        if page_name == "main":
            main_page = MainPage()
            main_page.render()
        elif page_name == "task_analysis":
            TaskAnalysisPage.render()
        elif page_name == "settings":
            SettingsPage.render()
        else:
            st.error(f"未知页面: {page_name}")
            MainPage().render()


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