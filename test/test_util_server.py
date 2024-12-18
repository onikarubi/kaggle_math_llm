from langchain_google_genai import ChatGoogleGenerativeAI
from src.predict import MathProblemPredictor
import src.utils.submit_server as submit_server
import dotenv
import os
import pytest

dotenv.load_dotenv('.env.local')

GOOGLE_GENERATIVE_MODEL = os.getenv("GOOGLE_GENERATIVE_MODEL")
csv_file = "src/datasets/test.csv"

@pytest.fixture
def load_llm():
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_GENERATIVE_MODEL,
        temperature=0.7,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )
    return llm

def test_load_env():
    
    assert os.path.exists(csv_file)
    assert GOOGLE_GENERATIVE_MODEL is not None

def test_load_predictor(load_llm):
    predictor = MathProblemPredictor(model=load_llm)
    assert predictor is not None

def test_submit_predictions(load_llm):
    predictor = MathProblemPredictor(model=load_llm)
    submit_server.submit_predictions(predictor, csv_file)
