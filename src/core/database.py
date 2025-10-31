"""
向量数据库管理模块

提供向量数据库的创建、管理和查询功能。
"""

import os
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config.settings import settings
from utils.validators import FileValidator, ValidationError


class DatabaseManager:
    """向量数据库管理器"""

    def __init__(self):
        """初始化数据库管理器"""
        self.persist_directory = settings.ensure_directory(settings.PERSIST_DIRECTORY)
        self.embedding_model = settings.EMBEDDING_MODEL
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self._embeddings = None
        self._db = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """获取嵌入模型实例"""
        if self._embeddings is None:
            print("正在初始化嵌入模型...")
            self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            print("嵌入模型初始化完成")
        return self._embeddings

    def load_documents(self, pdf_path: str) -> List[Document]:
        """
        从PDF文件加载文档

        Args:
            pdf_path: PDF文件路径

        Returns:
            List[Document]: 加载的文档列表

        Raises:
            FileNotFoundError: PDF文件不存在
            RuntimeError: 文档加载失败
        """
        try:
            # 验证文件
            pdf_path = FileValidator.validate_file_exists(pdf_path)
            pdf_path = FileValidator.validate_file_extension(pdf_path, ['pdf'])

            print(f"开始加载PDF文档: {pdf_path}")
            start_time = time.time()

            # 加载PDF
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            if not documents:
                raise RuntimeError("无法从PDF文件中加载任何内容")

            load_time = time.time() - start_time
            print(f"成功加载 {len(documents)} 页文档，耗时 {load_time:.2f} 秒")

            return documents

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValidationError, RuntimeError)):
                raise
            raise RuntimeError(f"文档加载失败: {str(e)}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档

        Args:
            documents: 原始文档列表

        Returns:
            List[Document]: 分割后的文档列表

        Raises:
            ValueError: 文档列表为空
            RuntimeError: 文档分割失败
        """
        if not documents:
            raise ValueError("文档列表不能为空")

        try:
            print(f"开始分割文档，块大小: {self.chunk_size}, 重叠: {self.chunk_overlap}")
            start_time = time.time()

            # 创建文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )

            # 分割文档
            texts = text_splitter.split_documents(documents)

            if not texts:
                raise RuntimeError("文档分割后没有产生任何内容")

            split_time = time.time() - start_time
            print(f"文档被分割为 {len(texts)} 个片段，耗时 {split_time:.2f} 秒")

            return texts

        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"文档分割失败: {str(e)}")

    def create_database(self, documents: List[Document]) -> Chroma:
        """
        创建向量数据库

        Args:
            documents: 文档列表

        Returns:
            Chroma: 创建的向量数据库

        Raises:
            ValueError: 文档列表为空
            RuntimeError: 数据库创建失败
        """
        if not documents:
            raise ValueError("文档列表不能为空")

        try:
            print("开始创建向量数据库...")
            start_time = time.time()

            # 创建数据库
            db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

            # 持久化到磁盘
            db.persist()

            create_time = time.time() - start_time
            print(f"向量数据库创建完成，耗时 {create_time:.2f} 秒")
            print(f"数据库已保存到: {self.persist_directory}")

            return db

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(f"向量数据库创建失败: {str(e)}")

    def load_database(self) -> Chroma:
        """
        加载已存在的向量数据库

        Returns:
            Chroma: 加载的向量数据库

        Raises:
            FileNotFoundError: 数据库不存在
            RuntimeError: 数据库加载失败
        """
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"向量数据库不存在: {self.persist_directory}")

        try:
            print(f"正在加载向量数据库: {self.persist_directory}")
            start_time = time.time()

            self._db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

            load_time = time.time() - start_time
            print(f"向量数据库加载完成，耗时 {load_time:.2f} 秒")

            return self._db

        except Exception as e:
            raise RuntimeError(f"向量数据库加载失败: {str(e)}")

    def get_retriever(self, k: int = None) -> Any:
        """
        获取检索器

        Args:
            k: 检索结果数量

        Returns:
            检索器对象
        """
        if k is None:
            k = settings.RETRIEVAL_K

        if self._db is None:
            self.load_database()

        return self._db.as_retriever(search_kwargs={"k": k})

    def database_exists(self) -> bool:
        """
        检查数据库是否存在

        Returns:
            bool: 数据库是否存在
        """
        return os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory)

    def get_database_info(self) -> Dict[str, Any]:
        """
        获取数据库信息

        Returns:
            Dict[str, Any]: 数据库信息
        """
        info = {
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "exists": self.database_exists()
        }

        if self.database_exists():
            try:
                db = self.load_database()
                # 获取集合信息
                collection = db._collection
                info.update({
                    "document_count": collection.count(),
                    "collection_name": collection.name
                })
            except Exception as e:
                info["error"] = str(e)

        return info

    def clear_database(self) -> bool:
        """
        清除数据库

        Returns:
            bool: 清除是否成功

        Raises:
            RuntimeError: 清除失败
        """
        try:
            if self.database_exists():
                import shutil
                shutil.rmtree(self.persist_directory)
                print(f"已清除数据库: {self.persist_directory}")
            else:
                print("数据库不存在，无需清除")

            # 重置内部状态
            self._db = None
            return True

        except Exception as e:
            raise RuntimeError(f"数据库清除失败: {str(e)}")

    def build_from_pdf(self, pdf_path: str = None, force_rebuild: bool = False) -> Chroma:
        """
        从PDF文件构建数据库

        Args:
            pdf_path: PDF文件路径，如果为None则使用默认路径
            force_rebuild: 是否强制重建

        Returns:
            Chroma: 创建或加载的向量数据库

        Raises:
            FileNotFoundError: PDF文件不存在
            RuntimeError: 构建失败
        """
        if pdf_path is None:
            pdf_path = settings.PDF_PATH

        # 检查是否需要重建
        if not force_rebuild and self.database_exists():
            print("数据库已存在，直接加载...")
            return self.load_database()

        # 清除现有数据库（如果存在）
        if self.database_exists():
            print("清除现有数据库...")
            self.clear_database()

        # 加载文档
        documents = self.load_documents(pdf_path)

        # 分割文档
        texts = self.split_documents(documents)

        # 创建数据库
        return self.create_database(texts)


# 创建全局数据库管理器实例
db_manager = DatabaseManager()


def build_database(pdf_path: str = None, force_rebuild: bool = False) -> Chroma:
    """
    便捷函数：构建向量数据库

    Args:
        pdf_path: PDF文件路径
        force_rebuild: 是否强制重建

    Returns:
        Chroma: 向量数据库
    """
    return db_manager.build_from_pdf(pdf_path, force_rebuild)


def get_retriever(k: int = None) -> Any:
    """
    便捷函数：获取检索器

    Args:
        k: 检索结果数量

    Returns:
        检索器对象
    """
    return db_manager.get_retriever(k)