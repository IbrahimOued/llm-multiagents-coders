"Singleton Concept Sample Code"
import copy
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer
from typing import Optional
import os
import torch
HF_TOKEN: Optional[str] = os.environ["HF_TOKEN"]

class LLMSingleton():
    "THe Singleton Class"
    llm = None

    def __new__(cls):
        if cls.llm is not None:
            return cls.llm
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"LLM Singleton Initialized with model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=HF_TOKEN,
        )
        stopping_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        llm = HuggingFaceLLM(
            model_name=model_name,
            model_kwargs={
                "token": HF_TOKEN,
                "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
                # "quantization_config": quantization_config
            },
            generate_kwargs={
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.9,
            },
            tokenizer_name=model_name,
            tokenizer_kwargs={"token": HF_TOKEN},
            stopping_ids=stopping_ids,
        )
        cls.llm = llm
        return cls.llm