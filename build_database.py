# build_database.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 定义持久化目录
PERSIST_DIRECTORY = "hspice_db"
# 定义Embedding模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build():
    print("开始加载PDF...")
    # 1. 加载PDF文档
    loader = PyPDFLoader("resource/hspice_manual.pdf")
    documents = loader.load()
    if not documents:
        print("错误：无法加载PDF文档。")
        return

    print(f"成功加载 {len(documents)} 页。")

    print("开始切分文档...")
    # 2. 切分文档
    # Mentor's Note: 这里的切分策略是MVP版本，比较粗糙。
    # 进阶版我们会按HSPICE的命令或章节来切分，效果会更好。
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"文档被切分为 {len(texts)} 个片段。")

    print("初始化Embedding模型...")
    # 3. 初始化开源的Embedding模型，这个会在本地运行
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("开始构建并持久化向量数据库...")
    # 4. 构建向量数据库并持久化到磁盘
    # 这步会下载模型（首次运行需要时间），然后计算所有文本片段的向量
    db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
    db.persist()
    print("数据库构建完成并已保存到磁盘！")

if __name__ == "__main__":
    build()