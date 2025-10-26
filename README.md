
# RAG4Spice: 智能HSPICE代码生成助手
**RAG4Spice** 是一个基于检索增强生成（RAG）技术的智能工具，旨在解决大型语言模型（LLM）在HSPICE等专用领域仿真语言上训练数据不足的问题。通过结合本地知识库，本项目可以根据用户的实验要求和电路网表，自动生成准确、可执行的HSPICE仿真代码。

## ✨ 主要功能

*   **OCR识别**：自动从上传的实验要求图片中提取文本。
*   **本地知识库**：基于您提供的HSPICE手册（PDF）构建向量数据库，确保信息的专业性和准确性。
*   **智能代码生成**：结合检索到的相关知识，利用Google Gemini模型生成高质量的HSPICE代码。
*   **Web用户界面**：基于Streamlit构建，提供直观、易于操作的图形界面。

## 🛠️ 技术架构

本项目的核心是检索增强生成（RAG）流程，具体步骤如下：

1.  **知识库构建 (离线)**：
    *   读取指定的PDF文档（例如 `hspice_manual.pdf`）。
    *   将文档内容智能切分成小块（Chunks）。
    *   使用 `Sentence-Transformers` 模型将文本块向量化（Embeddings）。
    *   将向量化后的数据存入 `Chroma` 本地向量数据库。

2.  **代码生成 (在线)**：
    *   **用户输入**：用户上传实验图片并通过OCR识别出“实验要求”，并手动输入“电路网表”。
    *   **检索 (Retrieve)**：将“实验要求”文本进行向量化，并在Chroma数据库中进行语义搜索，找出最相关的知识片段。
    *   **增强 (Augment)**：将检索到的知识片段、用户的实验要求和电路网表，一同填入一个精心设计的Prompt模板中。
    *   **生成 (Generate)**：将填充好的Prompt发送给Google Gemini Pro模型，生成最终的HSPICE代码。

```mermaid
graph TD
    A[用户上传实验图片/输入网表] --> B{Streamlit UI};
    B --> C[OCR提取实验要求];
    C --> D{Embedding模型};
    D --> E[Chroma向量数据库];
    E -- 检索最相关的知识 --> F{Prompt模板};
    C -- 实验要求 --> F;
    A -- 电路网表 --> F;
    F --> G[Google Gemini API];
    G --> H[生成的HSPICE代码];
    H --> B;
```

## 🚀 快速开始

请按照以下步骤在你的本地环境中配置并运行本项目。

### 1. 克隆项目
```bash
git clone https://github.com/your-username/RAG4Spice.git
cd RAG4Spice
```

### 2. 创建并激活虚拟环境
推荐使用 `uv` 或 `venv` 来管理Python环境。
```bash
# 使用 uv
uv venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# 或者使用 venv
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 安装 Tesseract-OCR
本项目使用 `pytesseract` 进行光学字符识别，它依赖于Google的Tesseract-OCR引擎。
*   请从 [Tesseract官方下载页面](https://github.com/UB-Mannheim/tesseract/wiki) 下载并安装适合你操作系统的版本。
*   **重要**: 在安装过程中，请记住你的Tesseract安装路径（例如 `D:/Tesseract/tesseract.exe`）。

### 5. 配置环境变量
本项目使用 `.env` 文件来管理密钥和配置。
1.  将 `.env.example` 文件复制并重命名为 `.env`。
2.  打开 `.env` 文件并填入以下信息：
    *   `GOOGLE_API_KEY`: 你从 Google AI Studio 获取的 Gemini API 密钥。
    *   `TESSERACT_CMD`: 你的 Tesseract-OCR 可执行文件的完整路径。

```dotenv
# .env file example
GOOGLE_API_KEY="your_google_api_key_here"
TESSERACT_CMD="D:/Tesseract/tesseract.exe" # Windows示例路径
# TESSERACT_CMD="/usr/bin/tesseract" # Linux示例路径
```

### 6. 构建本地知识库
```bash
python build_database.py
```
成功后，你会在项目根目录下看到一个 `hspice_db` 文件夹。

### 7. 运行应用
一切准备就绪！运行以下命令启动Streamlit应用。
```bash
streamlit run app.py
```
应用启动后，浏览器将自动打开一个本地网址，你就可以开始使用了。

## 📁 项目结构
```
RAG4Spice/
├── .venv/                # Python虚拟环境
├── hspice_db/            # 生成的向量数据库
├── resource/             # 存放用于构建知识库的源文件 (如.pdf)
├── .env                  # 存储你的API密钥和配置 (私密)
├── .env.example          # .env文件的模板
├── .gitignore            # Git忽略文件配置
├── app.py                # Streamlit应用主程序
├── build_database.py     # 构建向量数据库的脚本
└── requirements.txt      # 项目依赖
```

## 🔮 未来计划
这是一个基础版本，未来计划进行以下改进：

*   [ ] 优化Workflow，提升模型输出的精确性。
*   [ ] 优化文本切割（Chunking）策略以提高检索精度。
*   [ ] 引入Re-ranking模型进一步优化检索结果。
