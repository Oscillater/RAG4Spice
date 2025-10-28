# app.py
import os
import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
import json
import re

import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()
# --- 初始化 ---

PERSIST_DIRECTORY = "hspice_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 配置 Tesseract 环境变量
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
# 配置Google Gemini API
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    llm_pro = genai.GenerativeModel('gemini-2.5-pro')
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
你是一位HSPICE仿真专家，请分析以下OCR提取的实验要求，按照明确的规则进行任务分解，并以JSON格式输出。

重要说明：
- 用户将手动提供电路图中LLM无法从视觉上提取的所有信息
- 包括但不限于：MOS管源漏栅极位置、元件连接关系、节点标注、信号流向等
- 因为OCR/AI无法准确识别电路图中的视觉连接细节
- 你的任务是基于文字描述进行合理的任务分解

**任务分解规则：**

1. **多器件同类型测试 → 按器件拆分**
   - 示例："测试PMOS和NMOS的传输特性" → 两个任务："PMOS传输特性测试"、"NMOS传输特性测试"

2. **单器件多种特性测试 → 按特性拆分**
   - 示例："测试PMOS的传输特性和频率响应" → 两个任务："PMOS传输特性测试"、"PMOS频率响应测试"

3. **多种分析类型 → 按分析类型拆分**
   - 示例："进行DC和AC分析" → 两个任务："DC分析"、"AC分析"

**不拆分的情况：**
- 单器件单特性的测试
- 整体电路的单项综合测试

OCR文本：{ocr_text}

请输出简洁的JSON格式，包含：
1. general_description: 对整个实验的总体描述（包括需要用户提供的电路图视觉信息等）
2. tasks: 任务列表，每个任务包含id和description

输出JSON格式：
{{
  "general_description": "实验总体描述",
  "tasks": [
    {{
      "id": 1,
      "description": "具体任务描述1"
    }},
    {{
      "id": 2,
      "description": "具体任务描述2"
    }}
  ]
}}

**分解示例参考：**
- "测试PMOS和NMOS的DC和AC特性" → 4个任务：PMOS-DC、PMOS-AC、NMOS-DC、NMOS-AC
- "测试电容的充放电特性和频率响应" → 2个任务：电容充放电特性测试、电容频率响应测试
- "整体电路的功耗分析" → 1个任务：整体电路功耗分析（当然，如果OCR文本中有更多的要求，那么可以按照要求分成几个细分任务）
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

def parse_task_analysis(response_text):
    """
    解析任务分析的结果，提取JSON数据。

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
            return task_data
        except json.JSONDecodeError:
            print("方法1失败，尝试方法2...")

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
            if json_start != -1 and json_end != -1:
                json_str = cleaned_text[json_start:json_end]
                print("方法3成功：提取JSON部分")

        if json_str:
            print(f"提取的JSON字符串: {repr(json_str)}")
            task_data = json.loads(json_str)
            return task_data
        else:
            print("无法找到有效的JSON结构")
            # 如果没有找到JSON，返回默认结构
            return {
                "general_description": f"无法解析任务分析结果\n原始响应: {response_text[:200]}...",
                "tasks": []
            }

    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"尝试解析的JSON: {repr(json_str) if 'json_str' in locals() else 'None'}")
        return {
            "general_description": f"任务分析结果格式错误: {str(e)}\n原始响应: {response_text[:200]}...",
            "tasks": []
        }
    except Exception as e:
        print(f"任务分析解析错误: {e}")
        return {
            "general_description": f"任务分析处理失败: {str(e)}\n原始响应: {response_text[:200]}...",
            "tasks": []
        }

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
# 第一行：上传实验图片
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

# 显示OCR结果
if extracted_text:
    st.subheader("📝 OCR结果")
    st.text_area("提取的文本", value=extracted_text, height=150, key="ocr_result_display")

    # 添加任务分析按钮
    if st.button("🔍 分析任务", key="analyze_tasks"):
        with st.spinner("AI分析任务中..."):
            try:
                # 检查OCR文本是否为空
                if not extracted_text.strip():
                    st.warning("OCR结果为空，请检查图片或重新上传")
                    st.stop()

                prompt = TASK_ANALYSIS_PROMPT.format(ocr_text=extracted_text)
                st.session_state.last_prompt = prompt  # 保存prompt用于调试

                # 调用AI模型
                response = llm_flash.generate_content(prompt)

                # 检查响应是否有效
                if not response or not response.text:
                    st.error("AI模型返回空响应")
                    st.stop()

                st.session_state.last_response = response.text  # 保存响应用于调试

                # 解析任务分析结果
                task_analysis = parse_task_analysis(response.text)
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

# 第二行：任务确认与编辑
st.subheader("2. 任务确认与编辑")

if 'task_analysis' in st.session_state:
    # 1. 展示JSON结果
    st.subheader("📋 AI分析结果")
    st.json(st.session_state.task_analysis)

    # 2. 总体描述编辑
    st.subheader("📝 实验总体描述")
    general_description = st.text_area(
        "总体描述（可编辑）",
        value=st.session_state.task_analysis.get("general_description", ""),
        height=100,
        key="general_description_edit"
    )

    # 3. 电路图视觉信息输入
    st.subheader("🔌 电路图视觉信息")
    visual_info = st.text_area(
        "请提供电路图中LLM无法从视觉上提取的所有信息：",
        placeholder="包括但不限于：\n- MOS管源漏栅极位置\n- 元件连接关系\n- 节点标注\n- 信号流向\n- 元件参数值\n- 电源/地连接等",
        height=120,
        key="visual_info_input"
    )

    # 4. 任务列表编辑
    st.subheader("🎯 任务列表")
    tasks = st.session_state.task_analysis.get("tasks", [])

    for i, task in enumerate(tasks):
        with st.expander(f"任务 {task['id']}: {task['description'][:50]}..."):
            task_desc = st.text_area(
                f"任务描述",
                value=task["description"],
                height=80,
                key=f"task_edit_{i}"
            )
            # 更新任务描述
            tasks[i]["description"] = task_desc

    # 添加新任务按钮
    if st.button("➕ 添加任务"):
        new_task_id = max([t["id"] for t in tasks]) + 1 if tasks else 1
        tasks.append({"id": new_task_id, "description": "新任务描述"})
        st.session_state.task_analysis["tasks"] = tasks
        st.rerun()

    # 删除任务选择
    if len(tasks) > 1:
        tasks_to_delete = st.multiselect(
            "选择要删除的任务",
            options=[f"任务 {t['id']}" for t in tasks],
            key="delete_tasks"
        )
        if tasks_to_delete and st.button("🗑️ 删除选中任务"):
            # 删除选中的任务
            task_ids_to_delete = [int(t.split()[1]) for t in tasks_to_delete]
            st.session_state.task_analysis["tasks"] = [
                t for t in tasks if t["id"] not in task_ids_to_delete
            ]
            st.rerun()

    # 5. 保存编辑结果
    if st.button("💾 保存编辑结果", type="primary"):
        # 更新session_state中的任务分析
        st.session_state.task_analysis["general_description"] = general_description
        st.session_state.task_analysis["tasks"] = tasks
        st.session_state.visual_info = visual_info
        st.success("✅ 任务分析已更新")

else:
    st.info("请先上传图片并进行任务分析")

# 添加分隔线
st.divider()

# 第三行：生成HSPICE代码
st.subheader("3. 生成HSPICE代码")

# 检查是否有任务分析结果
if 'task_analysis' in st.session_state and 'visual_info' in st.session_state:
    # 显示当前的编辑状态
    st.subheader("📊 当前配置")
    st.markdown("**总体描述:**")
    st.write(st.session_state.task_analysis.get("general_description", ""))

    st.markdown("**任务列表:**")
    for task in st.session_state.task_analysis.get("tasks", []):
        st.write(f"- 任务 {task['id']}: {task['description']}")

    st.markdown("**视觉信息:**")
    st.write(st.session_state.visual_info if st.session_state.visual_info else "未提供")

    # 生成代码按钮
    if st.button("🚀 生成HSPICE代码", type="primary"):
        if not st.session_state.task_analysis.get("tasks"):
            st.warning("请至少添加一个任务")
        elif not st.session_state.visual_info.strip():
            st.warning("请提供电路图视觉信息")
        else:
            with st.spinner("AI生成代码中，请稍候..."):
                try:
                    # 1. 检索 (Retrieve)
                    print("步骤1: 正在从本地数据库检索知识...")
                    retrieved_docs = retriever.invoke(st.session_state.task_analysis["general_description"])
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    print("步骤1完成: 已成功检索到知识。")

                    # 2. 组装Prompt
                    print("步骤2: 正在组装Prompt...")
                    tasks_str = "\n".join([f"任务{task['id']}: {task['description']}" for task in st.session_state.task_analysis["tasks"]])

                    prompt = PROMPT_TEMPLATE.format(
                        context=context,
                        requirements=st.session_state.task_analysis["general_description"],
                        mos_connections=st.session_state.visual_info,
                        tasks=tasks_str
                    )
                    print("步骤2完成: Prompt已准备好。")

                    # 3. 生成 (Generate)
                    print("步骤3: 正在调用Google Gemini API...")
                    response = llm_pro.generate_content(prompt)
                    print("步骤3完成: 已成功从API获取到响应！")

                    analysis_text, hspice_code = parse_llm_output(response.text)

                    st.subheader("🎉 生成结果")

                    tab1, tab2 = st.tabs(["💡 模型分析", "💻 HSPICE 代码"])

                    with tab1:
                        if analysis_text:
                            st.markdown(analysis_text)
                        else:
                            st.info("模型没有提供额外的分析。")

                    with tab2:
                        if hspice_code:
                            st.code(hspice_code, language="spice")
                        else:
                            st.warning("在模型的输出中未能找到有效的HSPICE代码块。")

                    print("--- 调试结束 ---")

                except Exception as e:
                    print(f"!!! 错误: 在调用LLM API时发生异常: {e}")
                    st.error(f"调用AI失败，错误信息: {e}")
else:
    st.info("请先完成上方的任务分析和编辑")
