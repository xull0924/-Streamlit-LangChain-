Version:1.0StartHTML:0000000163EndHTML:0000051069StartFragment:0000044628EndFragment:0000051029SourceURL:file:///C:/Users/27670/llm_assistant/README.md<style></style>

 LLM 智能助手项目

基于 Streamlit + LangChain + 通义千问的智能聊天与文档问答助手。

 功能特性

纯聊天模式（调用通义千问模型）

知识库模式（上传 PDF 文档进行 RAG 问答）

支持多模型切换（qwen-turbo/qwen-plus/qwen-max）

快速开始

环境要求

- Python 3.11.8+

- Windows/Linux/macOS

安装步骤

1. 克隆项目

git clone <项目地址>

cd llm_assistant

2. 创建虚拟环境

python -m venv .venv

3. 激活虚拟环境
- Windows:

powershell

.venv\Scripts\Activate.ps1

4. 安装依赖

pip install -r requirements.txt

配置 API Key，使用 .env 文件

1. 在项目根目录创建 `.env` 文件

2. 添加以下内容：

DASHSCOPE_API_KEY=你的阿里云API_Key

3. 保存文件

在powershell运行应用

streamlit run app.py

项目结构

llm_assistant/

├── .venv/  Python 虚拟环境

├── .cache/  缓存目录（自动生成）

├── app.py  主应用文件

├── llm.py  LLM 模型封装

├── rag.py  RAG 功能实现

├── requirements.txt  项目依赖

├── README.md  说明文档

└── .env  环境变量（需手动创建）

常见问题

Q: 无法下载 HuggingFace 模型？

A: 使用国内镜像：

import os

os.environ['HF_ENDPOINT'] ='https://hf-mirror.com'

复制

Q: 内存不足？

A: 可切换至更轻量的模型，如 `qwen-turbo`
