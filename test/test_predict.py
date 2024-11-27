from src.predict_llm import PredictLLMFactory, PredictLLMType
from src.predict import MathProblemPredictor, MathProblemPredictorWithStructuredOutput
from langchain.globals import set_debug, set_verbose
from utils.logger_config import setup_logger
from typing import Generator, Tuple
import polars as pl
import pytest
import logging

set_debug(True)
set_verbose(True)

logger = setup_logger(__name__, logging.StreamHandler, logging.DEBUG)

test_gen_model = "gemini-1.5-flash-8b"
model_args = {
    'max_tokens': 1024,
    'temperature': 0,
}
problem = "What is $1-1$?"
reference_csv = "src/datasets/reference.csv"


def get_answer(df: pl.DataFrame, row: dict):
    if "answer" in df.columns:
        return row["answer"]

    return 0


def get_predict_answer(
    df: pl.DataFrame, predictor: MathProblemPredictor
) -> Generator[Tuple[int, int], None, None]:
    for row in df.iter_rows(named=True):
        problem = row["problem"]
        answer = get_answer(df, row)
        response = predictor.predict(problem)
        predict_answer = response.answer
        yield answer, predict_answer


def read_csv(file_path: str) -> pl.DataFrame:
    lazy_frame = pl.scan_csv(file_path)
    df = lazy_frame.collect(streaming=True)
    return df


@pytest.fixture
def predictor_with_structured_output():
    llm = PredictLLMFactory.create_llm(llm_type=PredictLLMType.CHATGEN, model_id=test_gen_model, basic_model_args=model_args)
    predictor = MathProblemPredictorWithStructuredOutput(llm)
    return predictor

@pytest.mark.skip
def test_read_predictor(predictor_with_structured_output):
    df = read_csv(reference_csv)

    for answer, predict_answer in get_predict_answer(
        df, predictor_with_structured_output
    ):
        logger.debug(f"answer: {answer}, predict_answer: {predict_answer}")
        try:
            assert answer == predict_answer
            logger.info('Successfully read and predicted')
        except AssertionError:
            logger.error(
                f"Prediction failed: Expected {answer}, but got {predict_answer}"
            )
            raise

    assert isinstance(df, pl.DataFrame)
