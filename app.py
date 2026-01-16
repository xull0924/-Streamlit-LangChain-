import streamlit as st
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain.chains import ConversationChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from llm import get_llm
from rag import build_retriever_from_pdf

st.set_page_config(page_title="LLM Assistant", page_icon="🤖")
st.title("🤖 大模型助手（聊天 + 知识库RAG）")

# 加载 .env 文件
load_dotenv()

# --- Key check ---
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    st.error("""
    未检测到 DASHSCOPE_API_KEY。请选择以下任一方式配置：

    1. 在项目根目录创建 .env 文件，内容为：
       DASHSCOPE_API_KEY=你的阿里云API_Key

    2. 设置系统环境变量：
       Windows: setx DASHSCOPE_API_KEY "你的Key"
       Linux/macOS: export DASHSCOPE_API_KEY="你的Key"
    """)
    st.stop()

# --- Sidebar: mode switch ---
mode = st.sidebar.radio("选择模式", ["聊天模式", "知识库模式（RAG）"])
model_name = st.sidebar.selectbox("选择模型", ["qwen-turbo", "qwen-plus", "qwen-max"], index=0)
if st.sidebar.button("清空对话"):
    st.session_state.messages = []
    # 也重置聊天记忆
    llm = get_llm(model_name=model_name)
    st.session_state.chat_chain = ConversationChain(llm=llm, memory=ConversationBufferMemory())
system_prompt = st.sidebar.text_area(
    "系统提示词",
    value="你是一个严谨的助理。若没有依据请明确说不知道。"
)
k = st.sidebar.slider("检索条数 TopK", 2, 10, 4)

# --- init session ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_chain" not in st.session_state:
    llm = get_llm(model_name=model_name)
    st.session_state.chat_chain = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(),
    )

# RAG chain is built only when user uploads a PDF
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None  # 初始化为 None，等待用户上传 PDF

# If user changes model, rebuild chains safely
if "last_model_name" not in st.session_state:
    st.session_state.last_model_name = model_name

if st.session_state.last_model_name != model_name:
    # rebuild chat chain
    llm = get_llm(model_name=model_name)
    st.session_state.chat_chain = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(),
    )
    # rebuild rag chain if exists
    if st.session_state.rag_chain is not None:
        # keep retriever but swap llm
        old_retriever = st.session_state.rag_chain.retriever
        st.session_state.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=old_retriever,
            return_source_documents=True
        )
    st.session_state.last_model_name = model_name
    st.session_state.messages = []  # clear UI history on model switch (simple & safe)

# --- RAG setup UI ---
if mode == "知识库模式（RAG）":
    uploaded = st.sidebar.file_uploader("上传 PDF 作为知识库", type=["pdf"])
    if uploaded is not None:
        os.makedirs(".cache", exist_ok=True)
        pdf_path = os.path.join(".cache", "kb.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded.read())

        with st.sidebar:
            st.success("PDF 已上传，正在构建知识库（首次会稍慢）...")

        try:
            retriever = build_retriever_from_pdf(pdf_path)
            llm = get_llm(model_name=model_name)
            st.session_state.rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
            st.sidebar.success("知识库构建完成 ✅")
        except Exception as e:
            st.sidebar.error(f"构建失败：{e}")
            st.session_state.rag_chain = None

# --- display history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- input ---
prompt = st.chat_input("请输入你的问题…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                if mode == "聊天模式":
                    answer = st.session_state.chat_chain.predict(input=f"[系统]\n{system_prompt}\n\n[用户]\n{prompt}")
                    sources = None
                else:
                    if st.session_state.rag_chain is None:
                        answer = "请先在左侧上传一个 PDF，构建知识库后再提问。"
                        sources = None
                    else:
                        result = st.session_state.rag_chain({"query": f"{system_prompt}\n\n{prompt}"})
                        answer = result["result"]
                        sources = result.get("source_documents", [])
            except Exception as e:
                answer = f"发生错误：{e}"
                sources = None

        st.markdown(answer)

        if sources:
            with st.expander("📚 查看引用来源"):
                for i, d in enumerate(sources, 1):
                    meta = d.metadata or {}
                    page = meta.get("page", meta.get("page_number", "未知"))
                    if isinstance(page, int):
                        page += 1
                    st.markdown(f"**[{i}] 页码：{page}**")
                    st.write(d.page_content[:400] + "…")

    st.session_state.messages.append({"role": "assistant", "content": answer})