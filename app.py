# 导入核心库
import streamlit as st
import os
from dotenv import load_dotenv

# 配置 HuggingFace 国内镜像，加速模型下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 导入 LangChain 相关模块
from langchain_classic.chains import ConversationChain
from langchain_classic.chains import RetrievalQA
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.llms import Tongyi
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ====================== 核心函数定义 ======================
def get_llm(model_name: str = "qwen-turbo"):
    """
    获取通义千问 LLM 实例
    :param model_name: 模型名称，可选 qwen-turbo/qwen-plus/qwen-max
    :return: 初始化后的 Tongyi LLM 实例
    :raise RuntimeError: 未配置 DASHSCOPE_API_KEY 时抛出异常
    """
    # 从环境变量中获取阿里云通义千问 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY environment variable.")
    # 初始化并返回通义千问 LLM
    return Tongyi(model_name=model_name, dashscope_api_key=api_key)

def build_retriever_from_pdf(pdf_path: str, k: int = 4):
    """
    从 PDF 文件构建检索器（RAG 核心）
    :param pdf_path: PDF 文件路径
    :param k: 检索时返回的最相似文档数量（TopK）
    :return: FAISS 向量库的检索器实例
    """
    # 1. 加载 PDF 文档
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 2. 文本分块：将长文本切分为小片段，避免超出模型上下文限制
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 每个文本块的最大长度
        chunk_overlap=150  # 块之间的重叠长度，保证上下文连续性
    )
    chunks = text_splitter.split_documents(docs)
    
    # 3. 初始化嵌入模型（用于将文本转为向量）
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # 轻量级嵌入模型，适合中文
    )
    
    # 4. 构建 FAISS 向量库并返回检索器
    db = FAISS.from_documents(chunks, embeddings)
    return db.as_retriever(search_kwargs={"k": k})

# ====================== Streamlit 页面配置 ======================
# 设置页面标题、图标和布局
st.set_page_config(page_title="LLM Assistant", page_icon="🤖")
st.title("🤖 大模型助手（聊天 + 知识库RAG）")

# ====================== 环境变量加载 ======================
# 加载 .env 文件中的环境变量（主要是 DASHSCOPE_API_KEY）
load_dotenv()

# 检查 API Key 是否配置，未配置则提示并终止程序
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

# ====================== 侧边栏配置 ======================
# 模式选择：聊天模式 / 知识库模式（RAG）
mode = st.sidebar.radio("选择模式", ["聊天模式", "知识库模式（RAG）"])

# 模型选择：通义千问不同版本
selected_model = st.sidebar.selectbox(
    "选择模型", 
    ["qwen-turbo", "qwen-plus", "qwen-max"], 
    index=0  # 默认选中 qwen-turbo（性价比最高）
)
model_name: str = str(selected_model) if selected_model else "qwen-turbo"

# 清空对话按钮
if st.sidebar.button("清空对话"):
    st.session_state.messages = []
    # 重置聊天链和记忆
    llm = get_llm(model_name=model_name)
    st.session_state.chat_chain = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory()
    )

# 系统提示词输入框
system_prompt = st.sidebar.text_area(
    "系统提示词",
    value="你是一个严谨助理。若没有依据请明确说不知道。",
    help="用于约束大模型的回答风格和规则"
)

# RAG 检索条数配置
k = st.sidebar.slider("检索条数 TopK", 2, 10, 4)

# ====================== 会话状态初始化 ======================
# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化聊天链（带记忆功能）
if "chat_chain" not in st.session_state:
    llm = get_llm(model_name=model_name)
    st.session_state.chat_chain = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(),  # 对话记忆，保存上下文
    )

# 初始化 RAG 链（初始为 None，需上传 PDF 后构建）
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# 记录最后一次选择的模型，用于模型切换时重置链
if "last_model_name" not in st.session_state:
    st.session_state.last_model_name = model_name

# ====================== 模型切换处理 ======================
# 如果用户切换了模型，重新初始化聊天链和 RAG 链
if st.session_state.last_model_name != model_name:
    # 重建聊天链
    llm = get_llm(model_name=model_name)
    st.session_state.chat_chain = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory()
    )
    # 如果已有 RAG 链，重建 RAG 链（保留检索器，替换 LLM）
    if st.session_state.rag_chain is not None:
        old_retriever = st.session_state.rag_chain.retriever
        st.session_state.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=old_retriever,
            return_source_documents=True  # 返回引用来源
        )
    # 更新最后一次模型名称，并清空对话历史
    st.session_state.last_model_name = model_name
    st.session_state.messages = []

# ====================== RAG 模式：PDF 上传与知识库构建 ======================
if mode == "知识库模式（RAG）":
    # 侧边栏 PDF 上传组件
    uploaded_file = st.sidebar.file_uploader("上传 PDF 作为知识库", type=["pdf"])
    
    if uploaded_file is not None:
        # 创建缓存目录（存放上传的 PDF）
        os.makedirs(".cache", exist_ok=True)
        pdf_path = os.path.join(".cache", "kb.pdf")
        
        # 将上传的文件写入本地
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # 提示正在构建知识库
        with st.sidebar:
            st.success("PDF 已上传，正在构建知识库（首次会稍慢）...")
        
        # 构建知识库（异常捕获）
        try:
            retriever = build_retriever_from_pdf(pdf_path, k=k)
            llm = get_llm(model_name=model_name)
            # 构建 RAG 链
            st.session_state.rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True  # 保留来源信息，用于展示引用
            )
            st.sidebar.success("知识库构建完成 ✅")
        except Exception as e:
            st.sidebar.error(f"构建失败：{e}")
            st.session_state.rag_chain = None

# ====================== 对话历史展示 ======================
# 遍历会话状态中的消息，展示聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ====================== 用户输入与回答处理 ======================
# 聊天输入框
prompt = st.chat_input("请输入你的问题…")

if prompt:
    # 将用户输入添加到对话历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成助手回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 聊天模式：直接调用对话链
                if mode == "聊天模式":
                    # 拼接系统提示词和用户输入
                    input_text = f"[系统]\n{system_prompt}\n\n[用户]\n{prompt}"
                    answer = st.session_state.chat_chain.predict(input=input_text)
                    sources = None  # 聊天模式无来源信息
                
                # 知识库模式：调用 RAG 链
                else:
                    # 未构建知识库时提示用户上传 PDF
                    if st.session_state.rag_chain is None:
                        answer = "请先在左侧上传一个 PDF，构建知识库后再提问。"
                        sources = None
                    else:
                        # 调用 RAG 链，传入系统提示词和用户问题
                        result = st.session_state.rag_chain({
                            "query": f"{system_prompt}\n\n{prompt}"
                        })
                        answer = result["result"]
                        sources = result.get("source_documents", [])
            
            # 捕获所有异常，避免程序崩溃
            except Exception as e:
                answer = f"发生错误：{e}"
                sources = None

        # 展示回答内容
        st.markdown(answer)

        # 展示引用来源（如果有）
        if sources:
            with st.expander("📚 查看引用来源"):
                for i, doc in enumerate(sources, 1):
                    # 获取文档元数据（页码）
                    meta = doc.metadata or {}
                    page = meta.get("page", meta.get("page_number", "未知"))
                    # 修正页码（部分 PDF 加载器页码从 0 开始）
                    if isinstance(page, int):
                        page += 1
                    # 展示来源信息
                    st.markdown(f"**[{i}] 页码：{page}**")
                    # 展示文档内容（截断为前 400 字符）
                    st.write(doc.page_content[:400] + "…")

    # 将助手回答添加到对话历史
    st.session_state.messages.append({"role": "assistant", "content": answer})
