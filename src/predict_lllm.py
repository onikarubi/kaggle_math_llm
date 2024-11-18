import abc
import os
from typing import Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_community.llms.vllm import VLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import enum


class BasicLLMParams(BaseModel):
    """
    Basic parameters for Language Model configurations.

    This class defines the common parameters used across different language models.
    It inherits from Pydantic's BaseModel for easy validation and serialization.

    Attributes:
        max_tokens (int): The maximum number of tokens to generate. Defaults to 1024.
        temperature (float): Controls randomness in generation. Higher values make output more random. Defaults to 0.7.
        top_p (float): Controls diversity via nucleus sampling. Must be between 0 and 1. Defaults to 1.0.
        top_k (int): Controls diversity by limiting to top k tokens. Defaults to 50.
    """

    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=50)


class BasePredictModel(metaclass=abc.ABCMeta):
    def __init__(self, model_id: str, model_args: dict[str, Any] = None):
        """
        Initialize a BasePredictModel instance.

        Args:
            model_id (str): The ID of the Language Model to use.
            model_args (dict[str, Any], optional): The arguments to pass to the Language Model constructor.
                Defaults to None.
        """

        self.model_id = model_id
        self.default_model_args = model_args or {}
        self.basic_model_params = BasicLLMParams(**self.default_model_args)

    @property
    def model_args(self) -> dict[str, Any]:
        """
        The arguments to pass to the Language Model constructor.

        Returns:
            dict[str, Any]: The model arguments as a dictionary.
        """

        return self.basic_model_params.model_dump() 

    def _combined_args(self, args: dict[str, Any] = None) -> dict[str, Any]:
        """
        Combine the default model arguments with any additional arguments.

        Args:
            args (dict[str, Any], optional): Additional arguments to combine with the default model arguments.
                Defaults to None.

        Returns:
            dict[str, Any]: The combined arguments.
        """
        if args is None:
            config_args = self.model_args
        else:
            config_args = {**self.model_args, **args}

        return config_args

    @abc.abstractmethod
    def create_llm(self, **opt_args) -> BaseChatModel:
        """
        Create a Language Model (LLM) of the given type.

        Args:
            **opt_args: Additional arguments to pass to the LLM constructor.

        Returns:
            BaseChatModel: The created LLM.

        Raises:
            NotImplementedError: If the given llm_type is not recognized.
        """
        raise NotImplementedError


class ChatGenPredictModel(BasePredictModel):
    def __init__(self, model_id, model_args: dict[str, Any] = None):
        """
        Initialize a ChatGenPredictModel instance.

        Args:
            model_id (str): The ID of the Language Model to use.
            model_args (dict[str, Any], optional): The arguments to pass to the Language Model constructor.
                Defaults to None.

        Raises:
            ValueError: If the environment variable "GOOGLE_API_KEY" is not set.
        """
        super().__init__(model_id, model_args)
        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError('Environment variable "GOOGLE_API_KEY" is not set.')

    def create_llm(self, opt_args: dict[str, Any] = None) -> BaseChatModel:
        """
        Create a Language Model (LLM) of the given type.

        Args:
            opt_args (dict[str, Any], optional): Additional arguments to pass to the LLM constructor.
                Defaults to None.

        Returns:
            BaseChatModel: The created LLM.
        """
        config_args = self._combined_args(opt_args)

        llm = ChatGoogleGenerativeAI(
            api_key=self.api_key,
            model=self.model_id,
            **config_args,
        )
        return llm


class PredictLLMType(enum.Enum):
    CHATGEN = "chatgen"
    OPENAI = "openai"
    LOCAL = "local"


class PredictLLMFactory:
    @classmethod
    def create_llm(
        cls, llm_type: PredictLLMType, model_id: str, basic_model_args: dict[str, Any] = None, **opt_args) -> BaseChatModel:

        """
        Create a Language Model (LLM) of the given type.

        Args:
            llm_type (PredictLLMType): The type of LLM to create.
            model_id (str): The ID of the LLM model to create.
            basic_model_args (dict[str, Any], optional): Basic model arguments to pass to the LLM constructor.
            **opt_args: Additional arguments to pass to the LLM constructor.

        Returns:
            BaseChatModel: The created LLM.

        Raises:
            ValueError: If the given llm_type is not recognized.
        """
        if llm_type == PredictLLMType.CHATGEN:
            model = ChatGenPredictModel(model_id, basic_model_args)

        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")

        return model.create_llm(**opt_args)
