from typing import Any

from pydantic import BaseModel, Field


class Input(BaseModel):
    """
    Form-based input schema for template.
    """
    data: Any = Field(
        ...,
        title="An applicable title is helpful for the navigator.",
        examples=["Examples are helpful for navigator to understand the data."],
        description="Always provide verbose description for the data.",
    )

class Output(BaseModel):
    """
    Form-based output schema for template.
    """
    data: Any = Field(
        ...,
        title="An applicable title is helpful for the navigator.",
        examples=["Examples are helpful for navigator to understand the data."],
        description="Always provide verbose description for the data.",
    )
