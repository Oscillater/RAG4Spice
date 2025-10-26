# app.py
import os
import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np

import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()
# --- 初始化 ---
# Mentor's Note: 把这些配置放在开头，方便管理
PERSIST_DIRECTORY = "hspice_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Set Tesseract command path from environment variable
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
# 配置Google Gemini API
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    llm = genai.GenerativeModel('gemini-2.5-pro')
except KeyError:
    st.error("请先设置 GOOGLE_API_KEY 环境变量！")
    st.stop()

# 加载Embedding模型和向量数据库
# Mentor's Note: 这些是重量级对象，我们只在程序启动时加载一次
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 3}) # k=3 表示检索最相关的3个片段

retriever = load_retriever()

# --- Prompt 模板 ---
# Mentor's Note: 这是我们系统的灵魂！一个好的Prompt决定了生成质量的上限。
PROMPT_TEMPLATE = """
你是一位精通HSPICE仿真的资深电路设计工程师。
请根据用户提供的“实验目标”和“电路网表”，并参考下方提供的“相关HSPICE知识”，生成一份完整、可执行的HSPICE仿真代码。

# 上下文信息
## 相关HSPICE知识:
{context}

## 实验目标:
{requirements}

## 电路网表:
{netlist}

# 输出要求
请将最终的HSPICE代码包裹在```hspice ... ```中。不要在代码前后添加任何多余的解释。
"""

# --- Streamlit 界面 ---
st.title("🤖 HSPICE RAG 代码生成助手")
st.caption("上传实验截图，输入网表，让AI为你写代码")

# --- 左右布局 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 上传实验图片")
    uploaded_file = st.file_uploader("选择一张包含实验要求的图片", type=["png", "jpg", "jpeg"])
    
    extracted_text = ""
    if uploaded_file is not None:
        # 显示图片
        image = Image.open(uploaded_file)
        st.image(image, caption="已上传的图片", use_container_width=True)
        
        # OCR处理
        try:
            img_array = np.array(image)
            # Pytesseract 需要 BGR 格式
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            extracted_text = pytesseract.image_to_string(img_cv)
        except Exception as e:
            st.error(f"OCR处理失败: {e}")

with col2:
    st.subheader("2. 确认信息并生成代码")
    
    # 使用 session_state 来持久化OCR结果
    if 'ocr_text' not in st.session_state or extracted_text:
        st.session_state.ocr_text = extracted_text

    requirements_text = st.text_area(
        "实验要求 (OCR结果, 可手动修改)", 
        value=st.session_state.ocr_text, 
        height=200
    )
    
    netlist_text = st.text_area(
        "电路网表 (请手动输入)", 
        placeholder="*示例*\nVdd vdd 0 1.8V\nM1 vout vin vss vss nmos ...",
        height=200
    )

    if st.button("🚀 生成HSPICE代码"):
        if not requirements_text.strip():
            st.warning("请输入或从图片中提取实验要求。")
        elif not netlist_text.strip():
            st.warning("请输入电路网表。")
        else:
            with st.spinner("AI思考中，请稍候..."):
                # --- 加入下面的调试代码 ---
                print("--- 开始调试 ---")
                
                # 1. 检索 (Retrieve)
                print("步骤1: 正在从本地数据库检索知识...")
                retrieved_docs = retriever.invoke(requirements_text)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                print("步骤1完成: 已成功检索到知识。")
                # print("检索到的内容:", context) # 如果想看具体内容，可以取消这行的注释
                
                # 2. 组装Prompt
                print("步骤2: 正在组装Prompt...")
                prompt = PROMPT_TEMPLATE.format(
                    context=context,
                    requirements=requirements_text,
                    netlist=netlist_text
                )
                print("步骤2完成: Prompt已准备好。")
                # print("最终的Prompt:", prompt) # 如果想看最终发给AI的内容，可以取消这行的注释

                # 3. 生成 (Generate)
                try:
                    print("步骤3: 正在调用Google Gemini API，请耐心等待网络响应...")
                    response = llm.generate_content(prompt)
                    print("步骤3完成: 已成功从API获取到响应！") # <--- 如果能看到这句，说明网络通了！
                    
                    # 清理和提取代码
                    hspice_code = response.text.strip()
                    if "```hspice" in hspice_code:
                        hspice_code = hspice_code.split("```hspice")[1].split("```")
                    
                    st.subheader("🎉 生成结果")
                    st.code(hspice_code, language="spice")
                    print("--- 调试结束 ---")

                except Exception as e:
                    print(f"!!! 错误: 在调用LLM API时发生异常: {e}") # <--- 打印出具体的错误信息
                    st.error(f"调用AI失败，错误信息: {e}")