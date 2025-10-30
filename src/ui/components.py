"""
UI组件模块

提供可重用的Streamlit UI组件。
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Callable
from PIL import Image

from ..models.task_models import Task, TaskAnalysis, GenerationResult
from ..utils.validators import ValidationError


class FileUploadComponent:
    """文件上传组件"""

    @staticmethod
    def render_file_upload(
        title: str,
        file_types: List[str],
        help_text: str = None
    ) -> Optional[Any]:
        """
        渲染文件上传组件

        Args:
            title: 组件标题
            file_types: 允许的文件类型列表
            help_text: 帮助文本

        Returns:
            上传的文件对象或None
        """
        st.subheader(title)
        uploaded_file = st.file_uploader(
            f"选择包含实验要求的文件",
            type=file_types,
            help=help_text
        )
        return uploaded_file

    @staticmethod
    def display_uploaded_file(uploaded_file) -> str:
        """
        显示上传的文件并提取文本

        Args:
            uploaded_file: 上传的文件对象

        Returns:
            str: 提取的文本
        """
        if uploaded_file is None:
            return ""

        file_type = uploaded_file.type
        extracted_text = ""

        if file_type == "application/pdf":
            # 处理PDF文件
            st.success("📄 已上传PDF文件")
            with st.spinner("正在提取PDF文本..."):
                try:
                    from ..utils.pdf_parser import extract_text_from_pdf
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    if extracted_text:
                        st.success(f"✅ PDF文本提取成功，共提取 {len(extracted_text)} 个字符")
                    else:
                        st.warning("⚠️ PDF文本提取结果为空")
                except Exception as e:
                    st.error(f"PDF文本提取失败: {e}")
        else:
            # 处理图片文件
            image = Image.open(uploaded_file)
            st.image(image, caption="已上传的图片", use_container_width=True)

            # OCR处理
            try:
                with st.spinner("正在进行OCR识别..."):
                    from ..utils.ocr import extract_text_from_image
                    extracted_text = extract_text_from_image(image)
                    if extracted_text:
                        st.success(f"✅ OCR识别成功，共识别 {len(extracted_text)} 个字符")
                    else:
                        st.warning("⚠️ OCR识别结果为空")
            except Exception as e:
                st.error(f"OCR处理失败: {e}")

        return extracted_text


class TaskAnalysisComponent:
    """任务分析组件"""

    @staticmethod
    def render_analyze_button(extracted_text: str) -> bool:
        """
        渲染分析按钮

        Args:
            extracted_text: 提取的文本

        Returns:
            bool: 是否点击分析按钮
        """
        if not extracted_text:
            return False

        return st.button("🔍 分析任务", key="analyze_tasks", type="primary")

    @staticmethod
    def render_extracted_text(extracted_text: str):
        """
        显示提取的文本

        Args:
            extracted_text: 提取的文本
        """
        if extracted_text:
            st.subheader("📝 提取的文本")
            st.text_area(
                "提取的文本内容",
                value=extracted_text,
                height=150,
                key="extracted_text_display"
            )

    @staticmethod
    def render_task_analysis_result(task_analysis: TaskAnalysis):
        """
        渲染任务分析结果

        Args:
            task_analysis: 任务分析结果
        """
        st.subheader("📋 AI分析结果")
        st.json(task_analysis.to_dict())

    @staticmethod
    def render_debug_info(prompt: str, response: str):
        """
        渲染调试信息

        Args:
            prompt: 提示词
            response: 响应文本
        """
        with st.expander("🔧 调试信息"):
            st.text_area("Prompt", value=prompt, height=100)
            st.text_area("原始响应", value=response, height=150)


class TaskEditComponent:
    """任务编辑组件"""

    @staticmethod
    def render_general_description_edit(general_description: str) -> str:
        """
        渲染总体描述编辑器

        Args:
            general_description: 总体描述

        Returns:
            str: 编辑后的总体描述
        """
        st.subheader("📝 实验总体描述")
        return st.text_area(
            "总体描述（包含环境配置）",
            value=general_description,
            height=100,
            key="general_description_edit",
            help="这里应该包含所有HSPICE文件共用的环境配置，如温度、电源电压、模型文件等"
        )

    @staticmethod
    def render_task_list(tasks: List[Task]) -> List[Task]:
        """
        渲染任务列表

        Args:
            tasks: 任务列表

        Returns:
            List[Task]: 更新后的任务列表
        """
        st.subheader("🎯 HSPICE文件任务")

        if not tasks:
            st.warning("📝 暂无任务，请重新分析或添加任务")
            return tasks

        st.info("📝 **每个Task对应一个独立的HSPICE文件**，包含该文件的完整仿真功能描述")

        for task_idx, task in enumerate(tasks):
            with st.expander(f"📄 {task.title} - HSPICE任务{task.id}"):
                updated_task = TaskEditComponent._render_single_task_edit(task, task_idx)
                if updated_task:
                    tasks[task_idx] = updated_task

        return tasks

    @staticmethod
    def _render_single_task_edit(task: Task, task_idx: int) -> Optional[Task]:
        """
        渲染单个任务编辑器

        Args:
            task: 任务对象
            task_idx: 任务索引

        Returns:
            Task: 更新后的任务对象
        """
        col1, col2 = st.columns([3, 1])

        with col1:
            task_title = st.text_input(
                "HSPICE文件名",
                value=task.title,
                key=f"task_title_{task_idx}",
                help="文件名格式，如：inverter_test.sp"
            )
            task_desc = st.text_area(
                "任务描述",
                value=task.description,
                height=120,
                key=f"task_desc_{task_idx}",
                help="描述该HSPICE文件要实现的完整仿真功能和内容，可包含多种分析类型"
            )

        with col2:
            st.write(f"任务ID: {task.id}")
            st.markdown("**HSPICE任务**")

            # 生成按钮
            if st.button("🚀 生成代码", key=f"generate_task_{task_idx}", type="primary"):
                # 更新任务信息
                updated_task = Task(
                    id=task.id,
                    title=task_title,
                    description=task_desc,
                    visual_info=task.visual_info
                )
                # 存储到session state
                st.session_state[f"generate_task_{task_idx}_data"] = updated_task
                return updated_task

            # 删除按钮
            if st.button("🗑️ 删除任务", key=f"delete_task_{task_idx}"):
                if len(st.session_state.task_analysis.tasks) > 1:
                    st.session_state.task_analysis.remove_task(task.id)
                    st.rerun()
                else:
                    st.warning("至少需要保留一个HSPICE任务")

        # 视觉信息输入
        st.markdown("**🔌 该HSPICE任务的电路图视觉信息：**")
        task_visual_info = st.text_area(
            f"请提供{task.title}所需的电路图信息：",
            value=task.visual_info,
            height=100,
            key=f"task_visual_info_{task_idx}",
            placeholder="包括但不限于：\n- MOS管源漏栅极位置\n- 元件连接关系\n- 节点标注\n- 信号流向\n- 元件参数值\n- 电源/地连接等",
            help="这些信息将用于生成该HSPICE文件的代码"
        )

        # 更新任务的视觉信息
        task.visual_info = task_visual_info

        return Task(
            id=task.id,
            title=task_title,
            description=task_desc,
            visual_info=task.visual_info
        )

    @staticmethod
    def render_add_task_button() -> bool:
        """
        渲染添加任务按钮

        Returns:
            bool: 是否点击添加按钮
        """
        return st.button("➕ 添加新HSPICE任务")

    @staticmethod
    def render_add_first_task_button() -> bool:
        """
        渲染添加第一个任务按钮

        Returns:
            bool: 是否点击添加按钮
        """
        return st.button("➕ 添加第一个HSPICE任务", type="primary")

    @staticmethod
    def render_save_button() -> bool:
        """
        渲染保存按钮

        Returns:
            bool: 是否点击保存按钮
        """
        return st.button("💾 保存编辑结果", type="primary")


class GenerationResultComponent:
    """代码生成结果组件"""

    @staticmethod
    def render_generation_result(result: GenerationResult):
        """
        渲染单个任务的生成结果

        Args:
            result: 生成结果
        """
        if not result.success:
            st.error(f"❌ {result.title} 生成失败: {result.error}")
            return

        st.subheader(f"🎉 {result.title} 生成结果")

        tab1, tab2 = st.tabs(["💡 模型分析", "💻 HSPICE 代码"])

        with tab1:
            if result.analysis:
                st.markdown(result.analysis)
            else:
                st.info("模型没有提供额外的分析。")

        with tab2:
            if result.hspice_code:
                st.code(result.hspice_code, language="spice")
                # 提供下载按钮
                st.download_button(
                    label=f"📥 下载 {result.title}",
                    data=result.hspice_code,
                    file_name=result.title,
                    mime="text/plain"
                )
            else:
                st.warning("在模型的输出中未能找到有效的HSPICE代码块。")

    @staticmethod
    def render_generation_progress(current: int, total: int):
        """
        渲染生成进度

        Args:
            current: 当前进度
            total: 总数
        """
        progress = current / total if total > 0 else 0
        st.progress(progress)
        st.write(f"生成进度: {current}/{total} ({progress:.1%})")


class ErrorDisplayComponent:
    """错误显示组件"""

    @staticmethod
    def render_error(title: str, error: Exception, show_details: bool = True):
        """
        渲染错误信息

        Args:
            title: 错误标题
            error: 异常对象
            show_details: 是否显示详细信息
        """
        st.error(f"❌ {title}: {str(error)}")

        if show_details:
            with st.expander("🔍 错误详情"):
                st.write(f"错误类型: {type(error).__name__}")
                st.write(f"错误信息: {str(error)}")
                import traceback
                st.text_area("完整堆栈跟踪", value=traceback.format_exc(), height=200)

    @staticmethod
    def render_validation_error(error: ValidationError):
        """
        渲染验证错误

        Args:
            error: 验证错误
        """
        st.error(f"⚠️ 数据验证错误: {str(error)}")


class SuccessDisplayComponent:
    """成功信息显示组件"""

    @staticmethod
    def render_success(message: str):
        """
        渲染成功信息

        Args:
            message: 成功信息
        """
        st.success(f"✅ {message}")

    @staticmethod
    def render_info(message: str):
        """
        渲染信息提示

        Args:
            message: 信息内容
        """
        st.info(f"ℹ️ {message}")

    @staticmethod
    def render_warning(message: str):
        """
        渲染警告信息

        Args:
            message: 警告信息
        """
        st.warning(f"⚠️ {message}")


# 便捷函数
def render_file_upload_section(file_types: List[str]) -> str:
    """
    渲染文件上传区域

    Args:
        file_types: 允许的文件类型

    Returns:
        str: 提取的文本
    """
    upload_component = FileUploadComponent()
    uploaded_file = upload_component.render_file_upload(
        "1. 上传实验要求文件",
        file_types,
        "支持图片文件（PNG, JPG, JPEG）和PDF文件"
    )

    if uploaded_file:
        extracted_text = upload_component.display_uploaded_file(uploaded_file)
        upload_component.render_extracted_text(extracted_text)
        return extracted_text

    return ""


def render_task_analysis_section(extracted_text: str) -> bool:
    """
    渲染任务分析区域

    Args:
        extracted_text: 提取的文本

    Returns:
        bool: 是否进行分析
    """
    analysis_component = TaskAnalysisComponent()
    analysis_component.render_extracted_text(extracted_text)
    return analysis_component.render_analyze_button(extracted_text)