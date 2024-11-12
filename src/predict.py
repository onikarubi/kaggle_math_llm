from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from src.schema import ResponseSchema
from langchain_core.language_models import BaseChatModel
import pandas as pd
import polars as pl


class MathProblemPredictor:
    def __init__(self, model: BaseChatModel):
        """
        Create a new MathProblemPredictor instance.

        Args:
            model: The AI model to use for generating responses.

        Attributes:
            model: The AI model to use for generating responses.
            parser: The PydanticOutputParser used to parse the response from the model.
        """
        self.model = model
        self.parser = PydanticOutputParser(pydantic_object=ResponseSchema)

    def submit_answer(self, id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
        """
        Submit an answer to a problem.

        Args:
            id_: The ID of the problem as a Polars DataFrame.
            question: The question as a Polars DataFrame.

        Returns:
            A Polars DataFrame with the answer to the problem.
        """
        id_: str = id_.item(0)
        question: str = question.item(0)
        prediction = self.predict(question)
        return pl.DataFrame({"id": id_, "answer": 0})

    def predict(self, problem: str) -> ResponseSchema:
        """
        Predict the solution to a math problem.

        Args:
            problem: The problem to solve, in LaTeX notation.

        Returns:
            A ResponseSchema object with the solution and steps to the problem.
        """
        chain = self._create_chain()
        response_schema = chain.invoke({"problem": problem, "format_instructions": self.parser.get_format_instructions()})
        return response_schema

    def _create_chain(self) -> RunnableSerializable[dict, any]:
        """
        Create a LangChain chain for generating responses to math problems.

        Returns:
            A LangChain chain that takes a dictionary with the problem and instructions as input, and
            returns a dictionary with the solution and steps to the problem.
        """
        prompt = self._create_prompts()
        chain = prompt | self.model | self.parser
        return chain


    def _create_prompts(self):
        """
        Create a LangChain prompt template for generating responses to math problems.

        The prompt is designed to be used with the `ChatPromptTemplate.from_messages` method, and
        will be formatted with the problem and instructions strings.

        Returns:
            A LangChain prompt template that takes a dictionary with the problem and instructions as
            input, and returns a dictionary with the solution and steps to the problem.
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Solve the following math problem (all in LaTeX notation). In the process of solving the problem, clearly sequence each step and give a brief rationale as to why you chose the method you did. Each step should be presented logically and step-by-step, and the final solution should be presented as a non-negative integer from 0 to 999.",
                ),
                ("user", "{problem}"),
                ("assistant", "{format_instructions}"),
            ]
        )

