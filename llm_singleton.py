"Singleton Concept Sample Code"
import copy
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from transformers import AutoTokenizer
from typing import Optional
import os
import torch
HF_TOKEN: Optional[str] = os.environ["HF_TOKEN"]

class LLMSingleton():
    "THe Singleton Class"

    llm = Ollama(model="codegemma:latest", request_timeout=120.0)

    def __new__(cls):
        return cls

    # def __init__(cls):
        # model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
        # print(f"LLM Singleton Initialized with model: {model_name}")
        # tokenizer = AutoTokenizer.from_pretrained(
        #     model_name,
        #     token=HF_TOKEN,
        # )
        # stopping_ids = [
        #     tokenizer.eos_token_id,
        #     tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        # ]

        # llm = HuggingFaceLLM(
        #     model_name=model_name,
        #     model_kwargs={
        #         "token": HF_TOKEN,
        #         "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
        #         # "quantization_config": quantization_config
        #     },
        #     generate_kwargs={
        #         "do_sample": True,
        #         "temperature": 0.6,
        #         "top_p": 0.9,
        #     },
        #     tokenizer_name=model_name,
        #     tokenizer_kwargs={"token": HF_TOKEN},
        #     stopping_ids=stopping_ids,
        # )
        # emb_model_name = "BAAI/bge-small-en-v1.5"
        # embed_model = HuggingFaceEmbedding(model_name=emb_model_name)
        # self.llm_model = llm
        # self.emb_model = embed_model
        # llm = Ollama(model="codegemma:2b", request_timeout=120.0)
        # cls.llm_model = llm

    @classmethod
    def get_llm_model(cls):
        "Use @classmethod to access class level variables"
        return cls.llm
    
    @classmethod
    def get_emb_model(cls):
        "Use @classmethod to access class level variables"
        return cls.emb_model