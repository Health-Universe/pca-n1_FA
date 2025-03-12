from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os

# Initialize FastAPI app
app = FastAPI(
    title="Prostate Cancer Survival Prediction API",
    description=(
        "Predict the overall survival of prostate cancer patients with lymph node-positive status "
        "using a pre-trained Gradient Boosting Survival Analysis (GBSA) model."
    ),
    version="1.0.0",
)

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model at startup
# Make sure 'GBSA12.18.pkl' is available in the working directory
GBSA = joblib.load("GBSA12.18.pkl")

# --------- Pydantic Models for Request and Response ---------

class PredictionOutput(BaseModel):
    survival_table: dict = Field(
        ...,
        description="Survival probabilities at 36, 60, 96, and 119 months",
        example={
            "36-month": 0.89,
            "60-month": 0.82,
            "96-month": 0.75,
            "119-month": 0.70,
        },
    )
    plot_image: str = Field(
        ...,
        description=(
            "Base64 encoded PNG image of the survival probability curve. "
            "Decode and display as an image from the string."
        ),
    )

# --------- The Prediction Endpoint ---------

@app.post("/predict", response_model=PredictionOutput)
async def predict(
    # Use annotated Form parameters for each required feature.
    age: Annotated[
        Literal["≤60", "61-69", "≥70"], Form(..., description="Age group")
    ],
    race: Annotated[
        Literal["White", "Black", "Other"], Form(..., description="Race")
    ],
    marital: Annotated[
        Literal["Married", "Unmarried"], Form(..., description="Marital status")
    ],
    clinical: Annotated[
        Literal["T1-T3a", "T3b", "T4"], Form(..., description="Clinical T stage")
    ],
    psa: Annotated[float, Form(..., description="PSA level")] = Form(...),
    gs: Annotated[
        Literal["≤7(3+4)", "7(4+3)", "8", "≥9"], Form(..., description="Gleason Score")
    ],
    nodes: Annotated[
        Literal["1", "2", "≥3", "No nodes were examined"],
        Form(..., description="Number of positive lymph nodes"),
    ],
    therapy: Annotated[
        Literal["Yes", "No"],
        Form(..., description="Radical prostatectomy: Yes means surgical treatment provided"),
    ],
    radio: Annotated[
        Literal["Yes", "No"],
        Form(..., description="Radiotherapy: Yes if radiotherapy is provided"),
    ],
):
    """
    Accepts patient-specific variables as form parameters, processes them into the
    appropriate one-hot encoded feature order, calls the pre-trained GBSA model to compute
    survival probabilities, and returns both a survival table and an encoded survival curve plot.
    """

    # --- One-hot encode the categorical variables exactly as in your Streamlit app ---
    # Age
    if age == "61-69":
        age_vec = [1, 0, 0]
    elif age == "≤60":
        age_vec = [0, 1, 0]
    else:  # "≥70"
        age_vec = [0, 0, 1]

    # Race (order: Black, Other, White)
    if race == "Black":
        race_vec = [1, 0, 0]
    elif race == "Other":
        race_vec = [0, 1, 0]
    else:  # White
        race_vec = [0, 0, 1]

    # Marital (order: Married, Unmarried)
    if marital == "Married":
        marital_vec = [1, 0]
    else:
        marital_vec = [0, 1]

    # Clinical T stage (order: T1-T3a, T3b, T4)
    if clinical == "T1-T3a":
        clinical_vec = [1, 0, 0]
    elif clinical == "T3b":
        clinical_vec = [0, 1, 0]
    else:  # T4
        clinical_vec = [0, 0, 1]

    # Radiotherapy (in the Streamlit code: if "No" ➔ [1,0], else [0,1])
    if radio == "No":
        radio_vec = [1, 0]
    else:
        radio_vec = [0, 1]

    # Radical prostatectomy (therapy): if "No" ➔ [1,0], else [0,1]
    if therapy == "No":
        therapy_vec = [1, 0]
    else:
        therapy_vec = [0, 1]

    # Positive lymph nodes (order: ≥3, No nodes were examined, 1, 2)
    if nodes == "≥3":
        nodes_vec = [1, 0, 0, 0]
    elif nodes == "1":
        nodes_vec = [0, 0, 1, 0]
    elif nodes == "2":
        nodes_vec = [0, 0, 0, 1]
    else:  # "No nodes were examined"
        nodes_vec = [0, 1, 0, 0]

    # Gleason Score (order: 7(4+3), 8, ≤7(3+4), ≥9)
    if gs == "7(4+3)":
        gs_vec = [1, 0, 0, 0]
    elif gs == "8":
        gs_vec = [0, 1, 0, 0]
    elif gs == "≤7(3+4)":
        gs_vec = [0, 0, 1, 0]
    else:  # "≥9"
        gs_vec = [0, 0, 0, 1]

    # Combine all one-hot vectors in the same order as used in the original code.
    features = (
        age_vec
        + race_vec
        + marital_vec
        + clinical_vec
        + radio_vec
        + therapy_vec
        + nodes_vec
        + gs_vec
    )

    # Build the dataframe for the model prediction.
    # Column names must match what the model expects.
    col_names = [
        "Age_61-69",
        "Age_<=60",
        "Age_>=70",
        "Race_Black",
        "Race_Other",
        "Race_White",
        "Marital_Married",
        "Marital_Unmarried",
        "CS.extension_T1_T3a",
        "CS.extension_T3b",
        "CS.extension_T4",
        "Radiation_None/Unknown",
        "Radiation_Yes",
        "Therapy_None",
        "Therapy_RP",
        "Nodes.positive_>=3",
        "Nodes.positive_None",
        "Nodes.positive_One",
        "Nodes.positive_Two",
        "Gleason.Patterns_4+3",
        "Gleason.Patterns_8",
        "Gleason.Patterns_<=3+4",
        "Gleason.Patterns_>=9",
    ]

    # Create the main feature dataframe and add the PSA column.
    x_df = pd.DataFrame([features], columns=col_names[:-1])  # all features except PSA first
    # Append the PSA (last column) separately. PSA is assumed to be the last column.
    x_df["PSA"] = psa
    # (Note: Ensure the model was trained with a PSA column appended in the same position)

    # --- Make prediction using the model ---
    # You can get an overall model score if needed:
    # score = GBSA.predict(x_df)[0]

    # Get survival probability function(s)
    surv_funcs = GBSA.predict_survival_function(x_df)

    # For simplicity, assume we have one survival function (for our single-row prediction)
    # Extract survival probabilities at timepoints 36, 60, 96, and 119 months.
    time_points = [36, 60, 96, 119]
    survival_probs = {}
    # We iterate through the survival function object.
    for fn in surv_funcs:
        for t in time_points:
            # Evaluate the function at time t and add to dictionary
            survival_probs[f"{t}-month"] = round(fn(t), 4)

    # --- Create and encode a survival curve plot ---
    # Plot the survival curve up to 120 months.
    fig, ax = plt.subplots()
    for fn in surv_funcs:
        # Plot using a step-like function. Plot points 0 to 120.
        t_vals = fn.x[fn.x <= 120]
        y_vals = fn(t_vals)
        ax.step(t_vals, y_vals, where="post", label="Survival Probability")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Survival Time (months)")
    ax.set_ylabel("Survival Probability")
    ax.grid(True)
    ax.legend()

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)  # Close the figure to free memory

    return PredictionOutput(survival_table=survival_probs, plot_image=plot_base64)
