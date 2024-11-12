from src.schema import ResponseSchema
import pytest
from pydantic import ValidationError


@pytest.fixture
def valid_schema():
    return {
        "solution": "This is a valid solution",
        "steps": ["Step 1", "Step 2"],
        "answer": 42,
    }


def test_valid_response_schema(valid_schema):
    
    schema = ResponseSchema(**valid_schema)
    assert schema.model_dump() == valid_schema


@pytest.mark.parametrize(
    "field,value,error_msg",
    [
        ("solution", "", "Solution cannot be empty"),
        ("solution", "   ", "Solution cannot be empty"),
        ("steps", [], "Steps list cannot be empty"),
    ],
)
def test_invalid_fields(valid_schema, field, value, error_msg):
    invalid_data = valid_schema.copy()
    print(f'invalid_data: {invalid_data}')
    invalid_data[field] = value
    with pytest.raises(ValidationError) as exc_info:
        ResponseSchema(**invalid_data)
    assert error_msg in str(exc_info.value)


@pytest.mark.parametrize(
    "answer,is_valid", [(0, True), (999, True), (-1, False), (1000, False)]
)
def test_answer_range(valid_schema, answer, is_valid):
    valid_schema["answer"] = answer
    if is_valid:
        schema = ResponseSchema(**valid_schema)
        assert schema.answer == answer
    else:
        with pytest.raises(ValidationError):
            ResponseSchema(**valid_schema)


@pytest.mark.parametrize(
    "field,invalid_value",
    [
        ("solution", 123),
        ("steps", "Not a list"),
        ("answer", "Not an int"),
    ],
)
def test_invalid_types(valid_schema, field, invalid_value):
    invalid_data = valid_schema.copy()
    invalid_data[field] = invalid_value
    with pytest.raises(ValidationError):
        ResponseSchema(**invalid_data)
