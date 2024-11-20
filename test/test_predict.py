from src.predict_lllm import PredictLLMFactory, PredictLLMType
from src.predict import MathProblemPredictor
from langchain_core.language_models import BaseChatModel
from langchain.globals import set_debug, set_verbose
from utils.logger_config import setup_logger
from typing import Generator, Tuple
import polars as pl
import pytest
import logging

set_debug(True)
set_verbose(True)

logger = setup_logger(__name__, logging.StreamHandler, logging.DEBUG)

test_gen_model = "gemini-1.5-flash-latest"
model_args = {
    'max_tokens': 1024,
    'temperature': 0.7,
}
problem = "What is $1-1$?"

@pytest.fixture
def valid_predictor():
    llm = PredictLLMFactory.create_llm(llm_type=PredictLLMType.CHATGEN, model_id=test_gen_model, basic_model_args=model_args)
    predictor = MathProblemPredictor(llm)
    return predictor

def get_answer(df: pl.DataFrame, row: dict):
    if 'answer' in df.columns:
        return row['answer']

    return 0

def get_predict_answer(df: pl.DataFrame, predictor: MathProblemPredictor) -> Generator[Tuple[int, int], None, None]:
    for row in df.iter_rows(named=True):
        problem = row['problem']
        answer = get_answer(df, row)
        response = predictor.predict(problem)
        predict_answer = response.answer
        yield answer, predict_answer

def read_csv(file_path: str) -> pl.DataFrame:
    lazy_frame = pl.scan_csv(file_path)
    df = lazy_frame.collect(streaming=True)
    return df

@pytest.mark.skip
def test_predict(valid_predictor):    
    df = read_csv("src/datasets/test.csv")

    for answer, predict_answer in get_predict_answer(df, valid_predictor):
        logger.debug(f"answer: {answer}, predict_answer: {predict_answer}")
        assert answer == predict_answer

    assert isinstance(df, pl.DataFrame)


