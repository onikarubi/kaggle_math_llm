from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from src.predict_llm import BasicLLMParams, ChatGenPredictModel, PredictLLMFactory, PredictLLMType
import os
import utils.logger_config as logger_config
import logging
import pytest

logger = logger_config.setup_logger(__name__, logging.StreamHandler, logging.DEBUG)

test_gen_model = "gemini-1.5-flash-latest"
@pytest.fixture
def default_llm_args():
    params = BasicLLMParams()
    return params.model_dump()

@pytest.fixture
def valid_gen_llm_args():
    return {
        "max_tokens": 512,
        "top_p": 0.7,
    }

def test_config_model_params(default_llm_args, valid_gen_llm_args):
    model = ChatGenPredictModel(model_id=test_gen_model)
    assert model.model_id == test_gen_model
    assert model.model_args == default_llm_args
    default_valid_args = model._combined_args()
    assert default_valid_args == default_llm_args

    valid_args = model._combined_args(valid_gen_llm_args)
    assert valid_args != default_llm_args

def test_llm_create(default_llm_args):
    llm = PredictLLMFactory.create_llm(
        llm_type=PredictLLMType.CHATGEN,
        model_id=test_gen_model,
    )
    logger.debug(f"llm: {llm}")
    assert isinstance(llm, BaseChatModel)



