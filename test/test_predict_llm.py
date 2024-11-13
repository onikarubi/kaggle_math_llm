import abc
import os
from langchain_core.language_models import BaseLLM
from langchain_community.llms.vllm import VLLM
from langchain_google_genai import ChatGoogleGenerativeAI


class BasePredictLlM(metaclass=abc.ABCMeta):
    def __init__(self, model_id: str, model_args=None):
        self.model_id = model_id
        self.model_args = model_args

    @abc.abstractmethod
    def create_llm(self) -> BaseLLM:
        raise NotImplementedError


class ChatGenPredictLLM(BasePredictLLM):
    def __init__(self, model_id: str, model_args=None):
        super().__init__(model_id, model_args)
        self.api_key = os.getenv('GOOGLE_API_KEY')

        if self.api_key is None:
            raise ValueError('GOOGLE_API_KEY is not set')

    def create_llm(self):
        llm = ChatGenPredictLLM()


class PredictLLMFactory:
    @classmethod
    def create_llm(cls, llm_type: str, model_id: str, model_args=None):
        pass
