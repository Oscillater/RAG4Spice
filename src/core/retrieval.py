"""
检索模块

提供向量检索功能，从知识库中检索相关的HSPICE知识。
"""

from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from core.database import db_manager
# from core.llm import llm_manager  # 已迁移到multi_llm
from core.multi_llm import multi_llm_manager
from config.settings import settings
from utils.text_processing import ensure_string


class RetrievalManager:
    """检索管理器"""

    def __init__(self):
        """初始化检索管理器"""
        self.retriever = None
        self._initialize_retriever()

    def _initialize_retriever(self):
        """初始化检索器"""
        try:
            # 确保数据库存在
            if not db_manager.database_exists():
                raise RuntimeError("向量数据库不存在，请先运行 build_database.py")

            # 获取检索器
            self.retriever = db_manager.get_retriever(settings.RETRIEVAL_K)
            print("检索器初始化完成")

        except Exception as e:
            raise RuntimeError(f"检索器初始化失败: {str(e)}")

    def retrieve_knowledge(self, query: str, k: int = None) -> List[Document]:
        """
        检索相关知识

        Args:
            query: 查询文本
            k: 检索结果数量

        Returns:
            List[Document]: 检索到的文档列表

        Raises:
            ValueError: 查询文本为空
            RuntimeError: 检索失败
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        if k is None:
            k = settings.RETRIEVAL_K

        try:
            print(f"正在检索知识，查询: {query[:100]}...")
            documents = self.retriever.invoke(query.strip())
            print(f"检索到 {len(documents)} 个相关文档")

            return documents

        except Exception as e:
            raise RuntimeError(f"知识检索失败: {str(e)}")

    def format_retrieved_documents(self, documents: List[Document]) -> str:
        """
        格式化检索到的文档

        Args:
            documents: 文档列表

        Returns:
            str: 格式化后的文本
        """
        if not documents:
            return "未检索到相关知识"

        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            if content:
                formatted_docs.append(f"[文档{i}]\n{content}")

        return "\n\n".join(formatted_docs)

    def generate_single_task_code(
        self,
        task: Dict[str, Any],
        general_description: str,
        additional_info: str,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        为单个任务生成HSPICE代码

        Args:
            task: 任务信息
            general_description: 总体描述
            additional_info: 补充信息
            model_id: 模型ID (可选，默认使用会话中的模型)
            api_key: API密钥 (可选，默认使用会话中的密钥)

        Returns:
            Dict: 生成结果

        Raises:
            ValueError: 任务信息无效
            RuntimeError: 代码生成失败
        """
        # 验证任务信息
        if not isinstance(task, dict):
            raise ValueError("任务信息必须是字典类型")

        required_fields = ['id', 'title', 'description']
        for field in required_fields:
            if field not in task:
                raise ValueError(f"任务缺少必需字段: {field}")

        task_id = task['id']
        title = task['title']
        description = ensure_string(task['description'])

        try:
            print(f"开始生成任务 {task_id}: {title}")

            # 1. 检索相关知识
            context = self._retrieve_context_for_task(general_description)

            # 2. 生成代码
            if model_id and api_key:
                # 使用指定的模型
                analysis, hspice_code = multi_llm_manager.generate_hspice_code(
                    model_id=model_id,
                    api_key=api_key,
                    context=context,
                    requirements=ensure_string(general_description),
                    mos_connections=ensure_string(additional_info),
                    task_description=description,
                    filename=title
                )
            else:
                # 使用多模型LLM管理器
                analysis, hspice_code = multi_llm_manager.generate_hspice_code(
                    model_id=None,  # 使用默认模型
                    api_key=None,  # 使用默认API密钥
                    context=context,
                    requirements=ensure_string(general_description),
                    additional_info=ensure_string(additional_info),
                    task_description=description,
                    filename=title
                )

            result = {
                "task_id": task_id,
                "title": title,
                "description": description,
                "analysis": analysis,
                "hspice_code": hspice_code,
                "success": True
            }

            print(f"任务 {task_id} 生成完成")
            return result

        except Exception as e:
            error_msg = f"任务 {task_id} 生成失败: {str(e)}"
            print(error_msg)

            return {
                "task_id": task_id,
                "title": title,
                "description": description,
                "analysis": "",
                "hspice_code": "",
                "error": error_msg,
                "success": False
            }

    def _retrieve_context_for_task(self, general_description: str) -> str:
        """
        为任务检索上下文知识

        Args:
            general_description: 总体描述

        Returns:
            str: 格式化的上下文知识
        """
        try:
            # 确保查询文本有效
            query = ensure_string(general_description)
            if not query:
                query = "HSPICE仿真"
                print("警告: general_description 为空，使用默认查询")

            # 检索文档
            documents = self.retrieve_knowledge(query)

            # 格式化文档
            return self.format_retrieved_documents(documents)

        except Exception as e:
            print(f"上下文检索失败: {str(e)}")
            return "知识检索失败，将基于通用知识生成代码"

    def batch_generate_tasks(
        self,
        tasks: List[Dict[str, Any]],
        general_description: str
    ) -> List[Dict[str, Any]]:
        """
        批量生成多个任务的代码

        Args:
            tasks: 任务列表
            general_description: 总体描述

        Returns:
            List[Dict]: 生成结果列表
        """
        if not tasks:
            return []

        results = []
        print(f"开始批量生成 {len(tasks)} 个任务的HSPICE代码")

        for i, task in enumerate(tasks):
            try:
                # 获取任务的补充信息
                additional_info = task.get("additional_info", "")

                # 生成单个任务的代码
                result = self.generate_single_task_code(
                    task=task,
                    general_description=general_description,
                    additional_info=additional_info
                )

                results.append(result)

                # 显示进度
                progress = (i + 1) / len(tasks) * 100
                print(f"进度: {progress:.1f}% ({i + 1}/{len(tasks)})")

            except Exception as e:
                error_msg = f"批量生成中任务 {i + 1} 失败: {str(e)}"
                print(error_msg)

                results.append({
                    "task_id": task.get("id", i + 1),
                    "title": task.get("title", f"任务{i + 1}.sp"),
                    "description": task.get("description", ""),
                    "analysis": "",
                    "hspice_code": "",
                    "error": error_msg,
                    "success": False
                })

        # 统计结果
        successful = sum(1 for r in results if r.get("success", False))
        print(f"批量生成完成: {successful}/{len(tasks)} 成功")

        return results

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        获取检索统计信息

        Returns:
            Dict: 统计信息
        """
        try:
            db_info = db_manager.get_database_info()
            retrieval_info = {
                "retrieval_k": settings.RETRIEVAL_K,
                "retriever_initialized": self.retriever is not None,
            }

            return {**db_info, **retrieval_info}

        except Exception as e:
            return {"error": f"获取统计信息失败: {str(e)}"}


# 创建全局检索管理器实例
retrieval_manager = RetrievalManager()


def retrieve_knowledge(query: str, k: int = None) -> List[Document]:
    """
    便捷函数：检索相关知识

    Args:
        query: 查询文本
        k: 检索结果数量

    Returns:
        List[Document]: 检索到的文档列表
    """
    return retrieval_manager.retrieve_knowledge(query, k)


def generate_task_code(
    task: Dict[str, Any],
    general_description: str,
    additional_info: str = ""
) -> Dict[str, Any]:
    """
    便捷函数：生成单个任务的HSPICE代码

    Args:
        task: 任务信息
        general_description: 总体描述
        additional_info: 补充信息

    Returns:
        Dict: 生成结果
    """
    return retrieval_manager.generate_single_task_code(
        task, general_description, additional_info
    )