import abc
import os
from typing import Generator, Tuple
from src.predict_llm import PredictLLMType, PredictLLMFactory
from src.predict import MathProblemPredictorWithStructuredOutput, MathProblemPredictor
from utils.logger_config import setup_logger
import polars as pl
import logging

logger = setup_logger(__name__, logging.StreamHandler, logging.DEBUG)

class LocalPredictions(abc.ABC):
    def __init__(self, model_type: PredictLLMType, model_id: str, reference_csv: str = None):
        self.llm = PredictLLMFactory.create_llm(model_type, model_id)
        self.predictor = MathProblemPredictorWithStructuredOutput(self.llm)
        if reference_csv:
            self.reference_csv = self._check_reference_csv(reference_csv)
        else:
            default_reference_csv = 'src/datasets/reference.csv'
            self.reference_csv = self._check_reference_csv(default_reference_csv)
        

    def _check_reference_csv(self, reference_csv: str) -> str:
        if not os.path.exists(reference_csv):
            raise ValueError(f'Reference CSV {self.reference_csv} does not exist.')

        return reference_csv

    def read_reference_csv(self) -> pl.DataFrame:
        lazy_frame = pl.scan_csv(self.reference_csv)
        df = lazy_frame.collect(streaming=True)
        return df

    def get_answer(self, df: pl.DataFrame, row: dict):
        if "answer" in df.columns:
            return row["answer"]

        return 0

    def get_predict_answer(self, df: pl.DataFrame, predictor: MathProblemPredictor) -> Generator[Tuple[int, int], None, None]:
        for row in df.iter_rows(named=True):
            problem = row["problem"]
            answer = self.get_answer(df, row)
            response = predictor.predict(problem)
            predict_answer = response.answer
            logging.debug(f"answer: {answer}, predict_answer: {predict_answer}")
            yield answer, predict_answer

