import os
import src.kaggle_evaluation.aimo_2_inference_server as server
import polars as pl
import pandas as pd

from predict import MathProblemPredictor

def submit_predictions(predictor: MathProblemPredictor, input_csv: str) -> None:
    def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
        id_ = id_.item(0)
        question = question.item(0)
        # prediction = predictor.predict(question)
        # answer = prediction.answer
        return pl.DataFrame({"id": id_, "answer": 0})


    inference_server = server.AIMO2InferenceServer(predict)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway((input_csv,))
