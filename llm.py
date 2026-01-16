import os
from langchain_community.llms import Tongyi

def get_llm(model_name: str = "qwen-turbo"):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY environment variable.")
    return Tongyi(model_name=model_name, dashscope_api_key=api_key)
