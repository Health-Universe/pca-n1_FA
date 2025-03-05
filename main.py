from typing import Annotated
import httpx
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import Input, Output

# a descriptive title and description are helpful for the navigator
app = FastAPI(
    title="Tool Template",
    description=(
        "The tool template provides a starting point for creating new tools. "
    ),
    version="1.0.0",
)

# This is so that other services can make requests to this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# including docstrings for each endpoint is helpful for the navigator
# the line data: Annotated[Input, Form()] ensures that the input data is form-based.
@app.post(
    "/call/",
    response_model=Output,
)
async def call(data: Annotated[Input, Form()]) -> Output | JSONResponse:
    """Process the input data and return the output.

    Args:
        data: The input data.

    Returns:
        The output data.
    """
    # use httpx for external api requests (anything you might use requests for) in order to leverage
    # async capabilities of FastAPI
    async with httpx.AsyncClient() as client:
        try:
            return Output(**data.model_dump())
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            return JSONResponse(
                {"error": f"Request failed: {e}"}, status_code=500
            )

# include a health check endpoint to help kubernetes determine if the service is healthy
@app.get("/health", response_class=JSONResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "Application is running."}
