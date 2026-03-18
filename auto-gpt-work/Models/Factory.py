# Models/Factory.py 完整修改版
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings  # 新增
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings


class ChatModelFactory:
    model_params = {
        "temperature": 0,
        "model_kwargs": {"seed": 42},
    }

    @classmethod
    def get_model(cls, model_name: str, use_azure: bool = False):
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("未找到 API 密钥，请检查 .env 文件")
        
        qwen_models = ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-max-longcontext"]
        
        if model_name in qwen_models:
            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                **cls.model_params,
            )
        elif "gpt" in model_name:
            if not use_azure:
                return ChatOpenAI(model=model_name, openai_api_key=api_key, **cls.model_params)
            else:
                return AzureChatOpenAI(
                    azure_deployment=model_name,
                    api_version="2024-05-01-preview",
                    openai_api_key=api_key,
                    **cls.model_params
                )
        elif model_name == "qwen2":
            return ChatOpenAI(
                model="alibaba/Qwen2-72B-Instruct",
                openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
                openai_api_base="https://api.siliconflow.cn/v1",
                **cls.model_params,
            )
        else:
            raise ValueError(f"不支持的模型：{model_name}")

    @classmethod
    def get_default_model(cls):
        return cls.get_model("qwen-plus")


class EmbeddingModelFactory:
    @classmethod
    def get_model(cls, model_name: str, use_azure: bool = False):
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("未找到 API 密钥，请检查 .env 文件")
        
        # 通义千问嵌入模型（使用官方 SDK）
        qwen_embedding_models = ["text-embedding-v1", "text-embedding-v2", "text-embedding-v3"]
        
        if model_name in qwen_embedding_models:
            return DashScopeEmbeddings(
                model=model_name,
                dashscope_api_key=api_key,
            )
        # 备用：OpenAI 兼容接口
        elif model_name.startswith("text-embedding"):
            return OpenAIEmbeddings(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                check_embedding_ctx_length=False,
            )
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

    @classmethod
    def get_default_model(cls):
        return cls.get_model("text-embedding-v3")