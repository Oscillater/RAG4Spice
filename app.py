# -*- coding: utf-8 -*-
# app.py
import os
import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
import json
import re
import PyPDF2
import io

import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

def extract_text_from_pdf(uploaded_file):
    """从上传的PDF文件中提取文本"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"PDF文本提取失败: {e}")
        return ""

# --- 初始化 ---

PERSIST_DIRECTORY = "hspice_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 配置 Tesseract 环境变量
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
# 配置Google Gemini API
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    llm_pro = genai.GenerativeModel('gemini-2.5-flash')
    llm_flash = genai.GenerativeModel('gemini-2.5-flash')
except KeyError:
    st.error("请先设置 GOOGLE_API_KEY 环境变量！")
    st.stop()

# 加载Embedding模型和向量数据库

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 3}) # k=3 表示检索最相关的3个片段

retriever = load_retriever()

# --- Prompt 模板 ---
TASK_ANALYSIS_PROMPT = """
你是一位HSPICE仿真专家，请分析以下实验要求，按照两层架构进行任务分解，并以JSON格式输出。

**重要指令：除非实验极其简单且只有单一分析类型，否则请优先生成多个task（多个HSPICE文件）！**

重要说明：
- 用户将手动提供电路图中LLM无法从视觉上提取的所有信息
- 包括但不限于：MOS管源漏栅极位置、元件连接关系、节点标注、信号流向等
- 因为OCR/AI无法准确识别电路图中的视觉连接细节
- 你的任务是基于文字描述进行合理的任务分解

**两层任务分解架构：**

1. **实验总体描述** (general_description): 包含实验总体概述和所有HSPICE文件共用的环境配置
   - 实验目的和总体要求
   - 公共的仿真环境配置（温度、电源电压、模型文件、全局参数等）
   - 所有HSPICE文件共享的条件和设置

2. **具体任务** (tasks): 按HSPICE文件数量进行任务分组
   - 每个task对应一个独立的HSPICE仿真文件
   - 每个task包含该文件的完整功能描述和测试内容
   - **强烈建议生成多个task而不是1个task**，除非实验确实极其简单
   - 优先考虑将复杂实验分解为多个独立的HSPICE文件
   - 如果需要多个独立文件，则按功能或测试类型分成多个task，尽可能生成多个task而不是1个task

**任务分组原则（基于HSPICE文件）：**

1. **多文件优先原则**
   - **默认情况下应该生成多个task**，每个task对应一个独立的HSPICE文件
   - 即使实验相对简单，如果有多种分析类型，也建议按分析类型分文件
   - 多文件有助于代码管理、调试和模块化设计

2. **单文件例外原则**
   - 只有在实验极其简单且只有一种分析类型时，才考虑创建单个task
   - 单文件task描述包含所有需要执行的仿真分析

3. **多文件适用情况**
   - **不同的分析类型**（DC、AC、TRAN、噪声分析等分别在不同文件中）
   - **不同的测试环境**（如不同温度、不同电源配置）
   - **不同的电路模块**（如放大器、滤波器、偏置电路分别测试）
   - **不同的测试条件**（如不同输入信号幅度、不同负载条件）
   - **复杂的大型仿真**需要分文件管理
   - **参数扫描和优化**可以独立成文件

4. **任务描述要求**
   - 每个task的描述应该完整说明该HSPICE文件需要执行的所有仿真
   - 可以是多种分析类型的组合（DC、AC、TRAN等）
   - 描述要清晰说明该文件的测试目标和预期结果

实验文本：{ocr_text}

请输出JSON格式：
{{
  "general_description": "实验总体描述和公共环境配置",
  "tasks": [
    {{
      "id": 1,
      "title": "文件名1.sp",
      "description": "该HSPICE文件的完整功能描述和测试内容，包含所有需要执行的仿真分析"
    }},
    {{
      "id": 2,
      "title": "文件名2.sp",
      "description": "该HSPICE文件的完整功能描述和测试内容，包含所有需要执行的仿真分析"
    }}
  ]
}}

**分解示例1（单文件）：**
实验要求："测试反相器的传输特性，包括DC扫描和瞬态响应"

分解结果：
- general_description: "反相器传输特性测试。环境配置：VDD=5V，温度=27°C，使用0.18um工艺模型"
- Task1: "inverter_test.sp" - "DC传输特性分析：输入电压从0到5V扫描，测量输出特性；瞬态响应分析：输入1kHz方波信号，观察输出响应波形"

**分解示例2（多文件）：**
实验要求："分别测试放大器和滤波器的频率响应，并在不同温度下测试"

分解结果：
- general_description: "放大器和滤波器频率响应测试。公共配置：电源±5V，使用0.18um工艺模型"
- Task1: "amplifier_freq.sp" - "放大器AC频率响应分析：1Hz-1MHz扫描，计算增益和相位"
- Task2: "filter_freq.sp" - "滤波器AC频率响应分析：1Hz-1MHz扫描，分析滤波特性"

**关键要点：**
- general_description必须包含所有HSPICE文件共用的环境配置
- **优先选择多任务结构，每个task对应一个独立的HSPICE文件**
- **只有在实验极其简单且只有单一分析类型时才使用单任务**
- 每个task的description要完整描述该文件的所有仿真内容
- 多任务结构有助于代码管理、调试和模块化设计
- 考虑按分析类型、测试环境、电路模块等因素合理分解任务
"""

PROMPT_TEMPLATE = """
你是一位精通HSPICE仿真的资深电路设计工程师。
请根据用户提供的"实验目标"、"MOS管连接信息"和"任务列表"，并参考下方提供的"相关HSPICE知识"，完成以下任务：

1.  首先，对仿真思路进行简要分析，解释你将如何实现实验目标。
2.  然后，生成一份完整、可执行的HSPICE仿真代码。

# 上下文信息
## 相关HSPICE知识:
{context}

## 实验目标:
{requirements}

## MOS管连接信息:
{mos_connections}

## 任务列表:
{tasks}

# 输出要求
- 你的分析内容应该直接书写。
- 请务必将最终的HSPICE代码包裹在```hspice ... ```中。
"""

def ensure_string(value):
    """
    确保值是字符串类型，处理各种可能的输入类型

    Args:
        value: 需要转换为字符串的值，可以是任意类型

    Returns:
        str: 转换后的字符串
    """
    if isinstance(value, str):
        return value.strip()
    elif isinstance(value, dict):
        # 如果是字典，尝试提取常见的文本字段
        description = value.get("description", "")
        if description:
            return str(description).strip()
        text = value.get("text", "")
        if text:
            return str(text).strip()
        # 如果没有找到常见字段，将整个字典转换为字符串
        return str(value).strip()
    elif isinstance(value, list):
        # 如果是列表，连接所有元素
        return " ".join(str(item) for item in value).strip()
    else:
        # 其他类型直接转换为字符串
        return str(value).strip()

def parse_task_analysis(response_text):
    """
    解析任务分析的结果，提取JSON数据。
    支持新的两层架构（tasks）。

    返回:
        dict: 包含general_description和tasks的字典
    """
    try:
        # 统一处理换行符
        cleaned_text = response_text.strip().replace('\r\n', '\n')

        # 调试：打印原始响应
        print(f"原始响应: {repr(response_text)}")
        print(f"清理后文本: {repr(cleaned_text)}")

        # 尝试多种方法提取JSON
        json_str = None

        # 方法1: 直接解析整个响应（如果是纯JSON）
        try:
            task_data = json.loads(cleaned_text)
            print("方法1成功：直接解析JSON")
            return convert_task_format(task_data)
        except json.JSONDecodeError as e:
            print(f"方法1失败: {e}")
            print("尝试方法2...")

        # 方法2: 查找JSON代码块
        if "```json" in cleaned_text:
            import re
            json_match = re.search(r'```json\s*\n(.*?)\n```', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                print("方法2成功：找到JSON代码块")
        elif "```" in cleaned_text:
            import re
            json_match = re.search(r'```\s*\n(.*?\{.*?\}.*?)\n```', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                print("方法2成功：找到代码块中的JSON")

        # 方法3: 查找第一个{到最后一个}
        if json_str is None:
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = cleaned_text[json_start:json_end]
                print("方法3成功：提取JSON部分")

        # 方法4: 尝试修复常见的JSON格式问题
        if json_str:
            try:
                # 移除可能的控制字符和多余空格
                import re
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # 移除控制字符
                json_str = re.sub(r'\s+', ' ', json_str)  # 规范化空格
                json_str = json_str.strip()

                print(f"清理后的JSON字符串: {repr(json_str)}")
                task_data = json.loads(json_str)
                return convert_task_format(task_data)
            except json.JSONDecodeError as e:
                print(f"JSON解析仍然失败: {e}")
                print(f"尝试解析的JSON: {repr(json_str)}")
                # 继续到默认错误处理
        else:
            print("无法找到有效的JSON结构")

        # 如果所有方法都失败，返回默认结构
        return {
            "general_description": ensure_string(f"无法解析任务分析结果\n原始响应: {response_text[:200]}..."),
            "tasks": []
        }

    except Exception as e:
        print(f"任务分析解析发生未预期错误: {e}")
        import traceback
        print(f"完整错误堆栈: {traceback.format_exc()}")

        # 创建安全的错误信息，避免包含可能导致问题的字符
        safe_error_msg = str(e).replace('"', "'").replace('\n', ' ').replace('\r', ' ')
        safe_response_text = response_text[:200].replace('"', "'").replace('\n', ' ').replace('\r', ' ')

        return {
            "general_description": f"任务分析结果格式错误: {safe_error_msg} 原始响应: {safe_response_text}...",
            "tasks": []
        }

def convert_task_format(task_data):
    """
    转换任务格式，确保新的两层架构（tasks）能正常工作。

    Args:
        task_data (dict): 原始任务数据

    Returns:
        dict: 标准化后的任务数据，包含general_description和tasks
    """
    try:
        # 防御性检查：确保task_data是字典类型
        if not isinstance(task_data, dict):
            print(f"错误：task_data不是字典类型，而是{type(task_data)}")
            return {
                "general_description": "任务数据格式错误",
                "tasks": []
            }

        # 检查是否包含tasks（新格式）
        if "tasks" in task_data:
            print("检测到tasks格式")
            tasks = task_data["tasks"]

            # 防御性检查：确保tasks是列表
            if not isinstance(tasks, list):
                print(f"错误：tasks不是列表类型，而是{type(tasks)}")
                return {
                    "general_description": ensure_string(task_data.get("general_description", "")),
                    "tasks": []
                }

            # 标准化tasks格式
            normalized_tasks = []
            for i, task in enumerate(tasks):
                if isinstance(task, dict):
                    normalized_task = {
                        "id": task.get("id", i + 1),
                        "title": task.get("title", f"任务{i+1}.sp"),
                        "description": ensure_string(task.get("description", ""))
                    }
                else:
                    normalized_task = {
                        "id": i + 1,
                        "title": f"任务{i+1}.sp",
                        "description": ensure_string(str(task))
                    }
                normalized_tasks.append(normalized_task)

            return {
                "general_description": ensure_string(task_data.get("general_description", "")),
                "tasks": normalized_tasks
            }

        # 兼容旧的sections格式，自动转换为新的tasks格式
        if "sections" in task_data:
            print("检测到sections格式，正在转换为tasks格式...")
            sections = task_data["sections"]

            if not isinstance(sections, list):
                print(f"错误：sections不是列表类型，而是{type(sections)}")
                return {
                    "general_description": ensure_string(task_data.get("general_description", "")),
                    "tasks": []
                }

            # 将sections转换为tasks
            converted_tasks = []
            for section in sections:
                if isinstance(section, dict):
                    # 合并section描述和子任务描述
                    section_desc = ensure_string(section.get("description", ""))
                    subtasks = section.get("subtasks", [])

                    if isinstance(subtasks, list) and subtasks:
                        # 如果有子任务，合并所有子任务描述
                        subtask_descriptions = []
                        for subtask in subtasks:
                            if isinstance(subtask, dict):
                                subtask_desc = ensure_string(subtask.get("description", ""))
                                if subtask_desc:
                                    subtask_descriptions.append(subtask_desc)

                        if subtask_descriptions:
                            full_description = f"{section_desc}; {'; '.join(subtask_descriptions)}"
                        else:
                            full_description = section_desc
                    else:
                        full_description = section_desc

                    converted_task = {
                        "id": section.get("id", len(converted_tasks) + 1),
                        "title": section.get("title", f"任务{len(converted_tasks) + 1}.sp"),
                        "description": full_description
                    }
                    converted_tasks.append(converted_task)

            print(f"转换完成：{len(sections)}个sections已转换为{len(converted_tasks)}个tasks")
            return {
                "general_description": ensure_string(task_data.get("general_description", "")),
                "tasks": converted_tasks
            }

        # 如果既没有tasks也没有sections，返回默认结构
        print("未检测到有效的任务结构，返回默认结构")
        return {
            "general_description": ensure_string(task_data.get("general_description", "")),
            "tasks": []
        }

    except Exception as e:
        print(f"convert_task_format发生错误: {e}")
        import traceback
        print(f"完整错误堆栈: {traceback.format_exc()}")

        # 返回安全的默认结构
        return {
            "general_description": f"任务格式转换失败: {str(e)}",
            "tasks": []
        }

def generate_single_task_code(task, visual_info):
    """
    为单个task生成HSPICE代码

    Args:
        task (dict): 包含task信息的字典
        visual_info (str): 该task的视觉信息
    """
    with st.spinner(f"正在生成 {task['title']} 的HSPICE代码..."):
        try:
            # 1. 检索 (Retrieve)
            print(f"步骤1: 正在从本地数据库检索知识 for {task['title']}...")

            # 防御性检查：确保 general_description 是字符串
            general_desc = ensure_string(st.session_state.task_analysis.get("general_description", ""))
            if not general_desc:
                print("警告: general_description 为空，使用默认查询")
                general_desc = "HSPICE仿真"

            print(f"使用查询字符串: {repr(general_desc[:100])}...")
            retrieved_docs = retriever.invoke(general_desc)
            context = "\\n\\n".join([doc.page_content for doc in retrieved_docs])
            print("步骤1完成: 已成功检索到知识。")

            # 2. 组装Prompt
            print(f"步骤2: 正在组装Prompt for {task['title']}...")

            # 单task生成的prompt模板
            SINGLE_TASK_PROMPT_TEMPLATE = """
你是一位精通HSPICE仿真的资深电路设计工程师。
请根据用户提供的"实验目标"、"电路图视觉信息"和"仿真任务描述"，并参考下方提供的"相关HSPICE知识"，完成以下任务：

1.  首先，对仿真思路进行简要分析，解释你将如何实现该HSPICE文件的仿真目标。
2.  然后，生成一份完整、可执行的HSPICE仿真代码。

# 上下文信息
## 相关HSPICE知识:
{context}

## 实验目标:
{requirements}

## 电路图视觉信息:
{mos_connections}

## 仿真任务描述:
{task_description}

## HSPICE文件名:
{filename}

# 输出要求
- 你的分析内容应该直接书写。
- 请务必将最终的HSPICE代码包裹在```hspice ... ```中。
- 生成的代码应该是一个完整的、可独立运行的HSPICE文件。
- 代码中应该包含所有必要的分析命令来完成指定的仿真任务。
"""

            prompt = SINGLE_TASK_PROMPT_TEMPLATE.format(
                context=context,
                requirements=st.session_state.task_analysis["general_description"],
                mos_connections=visual_info,
                task_description=task['description'],
                filename=task['title']
            )
            print("步骤2完成: Prompt已准备好。")

            # 3. 生成 (Generate)
            print(f"步骤3: 正在调用Google Gemini API for {task['title']}...")

            # 重试机制
            max_retries = 3
            retry_delay = 2  # 秒

            for attempt in range(max_retries):
                try:
                    print(f"尝试第 {attempt + 1} 次调用API...")
                    response = llm_pro.generate_content(
                        prompt,
                        request_options={"timeout": 600}
                    )
                    print(f"步骤3完成: 已成功从API获取到响应！")
                    break  # 成功则跳出重试循环
                except Exception as api_error:
                    print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {api_error}")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        # 最后一次尝试失败
                        raise api_error

            analysis_text, hspice_code = parse_llm_output(response.text)

            # 显示生成结果
            st.subheader(f"🎉 {task['title']} 生成结果")

            tab1, tab2 = st.tabs(["💡 模型分析", "💻 HSPICE 代码"])

            with tab1:
                if analysis_text:
                    st.markdown(analysis_text)
                else:
                    st.info("模型没有提供额外的分析。")

            with tab2:
                if hspice_code:
                    st.code(hspice_code, language="spice")
                    # 提供下载按钮
                    st.download_button(
                        label=f"📥 下载 {task['title']}",
                        data=hspice_code,
                        file_name=task['title'],
                        mime="text/plain"
                    )
                else:
                    st.warning("在模型的输出中未能找到有效的HSPICE代码块。")

            print(f"--- {task['title']} 生成完成 ---")

        except Exception as e:
            print(f"!!! 错误: 在调用LLM API时发生异常: {e}")
            st.error(f"调用AI失败，错误信息: {e}")

def parse_llm_output(response_text):
    """
    解析LLM的输出，分离分析和代码。

    返回:
        (analysis, hspice_code) 元组
    """
    analysis = ""
    hspice_code = ""

    # 统一处理换行符，防止\r\n等问题
    cleaned_text = response_text.strip().replace('\r\n', '\n')

    # 使用 ```hspice 作为分割点
    code_delimiter = "```hspice"

    if code_delimiter in cleaned_text:
        parts = cleaned_text.split(code_delimiter, 1)
        analysis = parts[0].strip()

        # 进一步从第二部分中分离代码和可能的后续文本
        code_part = parts[1]
        if "```" in code_part:
            hspice_code = code_part.split("```", 1)[0].strip()
        else:
            # 如果没有闭合的```，就将整个部分视为代码
            hspice_code = code_part.strip()
    else:
        # 如果模型没有按要求输出代码块，则将全部内容视为分析
        analysis = cleaned_text

    return analysis, hspice_code


# --- Streamlit 界面 ---
st.title("🤖 HSPICE RAG 代码生成助手")
st.caption("上传实验截图，分析任务，生成HSPICE代码")

# --- 三行布局 ---
# 第一行：上传实验图片或PDF
st.subheader("1. 上传实验要求文件")
uploaded_file = st.file_uploader("选择包含实验要求的文件", type=["png", "jpg", "jpeg", "pdf"])

extracted_text = ""
if uploaded_file is not None:
    file_type = uploaded_file.type

    if file_type == "application/pdf":
        # 处理PDF文件
        st.success("📄 已上传PDF文件")
        with st.spinner("正在提取PDF文本..."):
            try:
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
                img_array = np.array(image)
                # Pytesseract 需要 BGR 格式
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                extracted_text = pytesseract.image_to_string(img_cv)
                if extracted_text:
                    st.success(f"✅ OCR识别成功，共识别 {len(extracted_text)} 个字符")
                else:
                    st.warning("⚠️ OCR识别结果为空")
        except Exception as e:
            st.error(f"OCR处理失败: {e}")

# 显示提取的文本结果
if extracted_text:
    st.subheader("📝 提取的文本")
    st.text_area("提取的文本内容", value=extracted_text, height=150, key="extracted_text_display")

    # 添加任务分析按钮
    if st.button("🔍 分析任务", key="analyze_tasks"):
        with st.spinner("AI分析任务中..."):
            try:
                # 检查OCR文本是否为空
                if not extracted_text.strip():
                    st.warning("OCR结果为空，请检查文件或重新上传")
                    st.stop()

                prompt = TASK_ANALYSIS_PROMPT.format(ocr_text=extracted_text)
                st.session_state.last_prompt = prompt  # 保存prompt用于调试

                # 调用AI模型
                # 重试机制
                max_retries = 3
                retry_delay = 2  # 秒

                for attempt in range(max_retries):
                    try:
                        print(f"任务分析尝试第 {attempt + 1} 次调用API...")
                        response = llm_flash.generate_content(
                            prompt,
                            request_options={"timeout": 30000}
                        )
                        break  # 成功则跳出重试循环
                    except Exception as api_error:
                        print(f"任务分析API调用失败 (尝试 {attempt + 1}/{max_retries}): {api_error}")
                        if attempt < max_retries - 1:
                            print(f"等待 {retry_delay} 秒后重试...")
                            import time
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避
                        else:
                            # 最后一次尝试失败
                            raise api_error

                # 检查响应是否有效
                if not response or not response.text:
                    st.error("AI模型返回空响应")
                    st.stop()

                st.session_state.last_response = response.text  # 保存响应用于调试

                # 解析任务分析结果
                task_analysis = parse_task_analysis(response.text if hasattr(response, 'text') else str(response))
                st.session_state.task_analysis = task_analysis
                st.success("✅ 任务分析完成！")

                # 显示调试信息（可折叠）
                with st.expander("🔧 调试信息"):
                    st.text_area("Prompt", value=prompt, height=100)
                    st.text_area("原始响应", value=response.text, height=150)

            except Exception as e:
                st.error(f"任务分析失败: {e}")
                print(f"详细错误信息: {type(e).__name__}: {str(e)}")

                # 显示调试信息
                if 'last_prompt' in st.session_state:
                    with st.expander("🔧 调试信息"):
                        st.text_area("上次Prompt", value=st.session_state.last_prompt, height=100)
                        if 'last_response' in st.session_state:
                            st.text_area("上次响应", value=st.session_state.last_response, height=150)

                # 显示更多错误详情
                with st.expander("🔍 错误详情"):
                    st.write(f"错误类型: {type(e).__name__}")
                    st.write(f"错误信息: {str(e)}")
                    import traceback
                    st.text_area("完整堆栈跟踪", value=traceback.format_exc(), height=200)

# 添加分隔线
st.divider()

# 第二行：任务编辑与代码生成
st.subheader("2. 任务编辑与代码生成")

if 'task_analysis' in st.session_state:
    # 1. 展示JSON结果
    st.subheader("📋 AI分析结果")
    st.json(st.session_state.task_analysis)

    # 2. 总体描述编辑（已包含环境配置）
    st.subheader("📝 实验总体描述")
    general_description = st.text_area(
        "总体描述（包含环境配置）",
        value=st.session_state.task_analysis.get("general_description", ""),
        height=100,
        key="general_description_edit",
        help="这里应该包含所有HSPICE文件共用的环境配置，如温度、电源电压、模型文件等"
    )

    # 3. 任务编辑
    st.subheader("🎯 HSPICE文件任务")

    # 获取tasks
    tasks = st.session_state.task_analysis.get("tasks", [])

    if tasks:
        st.info("📝 **每个Task对应一个独立的HSPICE文件**，包含该文件的完整仿真功能描述")

        for task_idx, task in enumerate(tasks):
            with st.expander(f"📄 {task['title']} - HSPICE任务{task['id']}"):
                # Task信息编辑
                col1, col2 = st.columns([3, 1])

                with col1:
                    task_title = st.text_input(
                        "HSPICE文件名",
                        value=task["title"],
                        key=f"task_title_{task_idx}",
                        help="文件名格式，如：inverter_test.sp"
                    )
                    task_desc = st.text_area(
                        "任务描述",
                        value=task.get("description", ""),
                        height=120,
                        key=f"task_desc_{task_idx}",
                        help="描述该HSPICE文件要实现的完整仿真功能和内容，可包含多种分析类型"
                    )

                with col2:
                    st.write(f"任务ID: {task['id']}")
                    st.markdown("**HSPICE任务**")

                    # 生成按钮
                    if st.button("🚀 生成代码", key=f"generate_task_{task_idx}", type="primary"):
                        # 更新task信息
                        tasks[task_idx]["title"] = task_title
                        tasks[task_idx]["description"] = task_desc
                        st.session_state.task_analysis["tasks"] = tasks

                        # 生成代码
                        generate_single_task_code(tasks[task_idx], tasks[task_idx].get("visual_info", ""))

                    # 删除按钮
                    if st.button("🗑️ 删除任务", key=f"delete_task_{task_idx}"):
                        if len(tasks) > 1:  # 至少保留一个task
                            tasks.pop(task_idx)
                            st.session_state.task_analysis["tasks"] = tasks
                            st.rerun()
                        else:
                            st.warning("至少需要保留一个HSPICE任务")

                # 该任务的视觉信息输入
                st.markdown("**🔌 该HSPICE任务的电路图视觉信息：**")
                task_visual_info = st.text_area(
                    f"请提供{task['title']}所需的电路图信息：",
                    value=task.get("visual_info", ""),
                    height=100,
                    key=f"task_visual_info_{task_idx}",
                    placeholder="包括但不限于：\n- MOS管源漏栅极位置\n- 元件连接关系\n- 节点标注\n- 信号流向\n- 元件参数值\n- 电源/地连接等",
                    help="这些信息将用于生成该HSPICE文件的代码"
                )
                # 更新task的视觉信息
                tasks[task_idx]["visual_info"] = task_visual_info

                # 更新task信息
                tasks[task_idx]["title"] = task_title
                tasks[task_idx]["description"] = task_desc

        # 添加新任务按钮
        if st.button("➕ 添加新HSPICE任务"):
            new_task_id = max([t["id"] for t in tasks]) + 1 if tasks else 1
            tasks.append({
                "id": new_task_id,
                "title": f"任务{new_task_id}.sp",
                "description": "新添加的HSPICE仿真任务描述",
                "visual_info": ""
            })
            st.session_state.task_analysis["tasks"] = tasks
            st.rerun()

    else:
        st.warning("📝 暂无任务，请重新分析或添加任务")

        # 添加第一个任务的按钮
        if st.button("➕ 添加第一个HSPICE任务"):
            tasks.append({
                "id": 1,
                "title": "任务1.sp",
                "description": "新添加的HSPICE仿真任务描述",
                "visual_info": ""
            })
            st.session_state.task_analysis["tasks"] = tasks
            st.rerun()

    # 4. 保存编辑结果
    if st.button("💾 保存编辑结果", type="primary"):
        # 更新session_state中的任务分析
        st.session_state.task_analysis["general_description"] = general_description
        st.session_state.task_analysis["tasks"] = tasks
        st.success("✅ 任务分析已更新")

else:
    st.info("请先上传文件并进行任务分析")