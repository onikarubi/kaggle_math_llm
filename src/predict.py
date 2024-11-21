from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from src.schema import ResponseSchema
from langchain_core.language_models import BaseChatModel
import pandas as pd
import polars as pl
import abc

class MathProblemPredictor(metaclass=abc.ABCMeta):
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
        self.PROBLEM_SYSTEM_MSG = "Solve the following math problem (all in LaTeX notation). In the process of solving the problem, clearly sequence each step and give a brief rationale as to why you chose the method you did. Each step should be presented logically and step-by-step, and the final solution should be presented as a non-negative integer from 0 to 999."

    @abc.abstractmethod
    def predict(self, problem: str) -> ResponseSchema:
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def _create_prompts(self) -> ChatPromptTemplate:
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def _create_chain(self) -> RunnableSerializable[dict, any]:
        """
        Create a LangChain chain for generating responses to math problems.

        Returns:
            A LangChain chain that takes a dictionary with the problem and instructions as input, and
            returns a dictionary with the solution and steps to the problem.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class MathProblemPredictorWithOutputParser(MathProblemPredictor):
    def __init__(self, model):
        """
        Create a new MathProblemPredictorWithOutputParser instance.

        Args:
            model: The AI model to use for generating responses.

        Attributes:
            parser: The PydanticOutputParser used to parse the response from the model.
        """
        super().__init__(model)
        self.parser = PydanticOutputParser(pydantic_object=ResponseSchema)

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

    def _create_prompts(self) -> ChatPromptTemplate:
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
                    self.PROBLEM_SYSTEM_MSG,
                ),
                ("user", "{problem}"),
                ("assistant", "{format_instructions}"),
            ]
        )


class MathProblemPredictorWithStructuredOutput(MathProblemPredictor):
    def __init__(self, model):
        """
        Initialize a MathProblemPredictorWithStructuredOutput instance.

        Args:
            model: The AI model to use for generating responses.

        Attributes:
            structured_llm: The AI model with structured output, which is a
                LangChain model that takes a dictionary with the problem and
                instructions as input, and returns a dictionary with the solution
                and steps to the problem.
        """
        
        super().__init__(model)
        self.structured_llm = self.model.with_structured_output(ResponseSchema)

    def predict(self, problem: str) -> ResponseSchema:
        """
        Predict the solution to a math problem.

        Args:
            problem: The problem to solve, in LaTeX notation.

        Returns:
            A ResponseSchema object with the solution and steps to the problem.
        """
        prompts = self._create_prompts()
        chain = prompts | self.structured_llm
        response = chain.invoke({"problem": problem})
        return response

    def _create_prompts(self):
        """
        Create a LangChain prompt template for generating responses to math problems.

        The prompt template includes two messages: one from the system with the problem
        instructions, and one from the user with the problem.

        Returns:
            A ChatPromptTemplate with the two messages.
        """
        return ChatPromptTemplate.from_messages([
            ("system", self.PROBLEM_SYSTEM_MSG),
            ("user", "{problem}"),
        ])

    def _create_chain(self):
        raise NotImplementedError("Not implemented")