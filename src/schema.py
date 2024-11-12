from pydantic import BaseModel, Field, field_validator

class ResponseSchema(BaseModel):
    solution: str = Field(description="The solution to the problem")
    steps: list[str] = Field(description="The steps to solve the problem")
    answer: int = Field(description="The final answer(from 0 to 999)", ge=0, le=999)

    @field_validator("solution")
    @classmethod
    def solution_not_empty(cls, v: str) -> str:        
        """
        Field validator for the solution field.

        Checks if the solution is not empty by stripping out any whitespace and
        checking if the resulting string is not empty. If it is, it raises a
        ValueError with the message "Solution cannot be empty".

        :param v: The value of the solution field as a string
        :return: The validated value of the solution field
        :raises ValueError: If the solution is empty
        """
        if not v.strip():
            raise ValueError("Solution cannot be empty")
        return v

    @field_validator("steps")
    @classmethod
    def steps_not_empty(cls, v: list[str]) -> list[str]:
        """
        Field validator for the steps field.

        Checks that the list of steps is not empty. If it is, raises a ValueError
        with a message indicating that the steps list cannot be empty.

        Returns the original list of steps if it is not empty.
        """
        if len(v) == 0:
            raise ValueError("Steps list cannot be empty")
        return v
