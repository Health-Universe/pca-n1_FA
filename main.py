import os
import io
import base64
from typing import Optional, Dict, Annotated

import joblib
import pandas as pd
import matplotlib.pyplot as plt

import openai

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Set up the OpenAI API key from environment variable (if not, set it programmatically)
openai.api_key = os.getenv("API_KEY")

# Load the pre-trained model (make sure the pkl file is in the same directory)
GBSA = joblib.load("GBSA12.18.pkl")

app = FastAPI(
    title="Prostate Cancer Survival Predictor",
    description=(
        "This API accepts patient information for prostate cancer patients with lymph node-positive status, "
        "predicts survival probabilities at selected time points using a pre-trained Gradient Boosting Survival Analysis model, "
        "generates a survival plot, and returns an explanation using OpenAI's ChatCompletion."
    ),
    version="1.0.0",
)

# Allow CORS for all origins. Adjust for production as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Pydantic Models -----
class PredictionInput(BaseModel):
    age: str = Field(..., description="Age group. Options: '≤60', '61-69', '≥70'.")
    race: str = Field(..., description="Race. Options: 'White', 'Black', 'Other'.")
    marital: str = Field(..., description="Marital Status. Options: 'Married', 'Unmarried'.")
    clinical: str = Field(..., description="Clinical T stage. Options: 'T1-T3a', 'T3b', 'T4'.")
    psa: float = Field(..., description="PSA level (ng/mL).")
    gs: str = Field(..., description="Gleason Score. Options: '≤7(3+4)', '7(4+3)', '8', '≥9'.")
    nodes: str = Field(..., description="Number of positive lymph nodes. Options: '1', '2', '≥3', 'No nodes were examined'.")
    therapy: str = Field(..., description="Radical prostatectomy. Options: 'Yes', 'No'.")
    radio: str = Field(..., description="Radiotherapy. Options: 'Yes', 'No'.")


class PredictionOutput(BaseModel):
    survival_probabilities: Dict[str, float] = Field(
        ...,
        description=(
            "Predicted survival probabilities at 36, 60, 96, and 119 months. "
            "The keys are timepoints (in months) and values are probabilities."
        ),
    )
    explanation: str = Field(..., description="Explanation interpreting the survival probabilities.")
    survival_curve: Optional[str] = Field(
        None, description="Base64 image (PNG) of the survival curve plot."
    )


# ----- Endpoint -----
@app.post("/predict", response_model=PredictionOutput)
async def predict(
    # Use Annotated[ , Form(...)] so that these values can be passed via form-data.
    age: Annotated[str, Form(..., description="Age group. Options: '≤60', '61-69', '≥70'.")],
    race: Annotated[str, Form(..., description="Race. Options: 'White', 'Black', 'Other'.")],
    marital: Annotated[str, Form(..., description="Marital Status. Options: 'Married', 'Unmarried'.")],
    clinical: Annotated[str, Form(..., description="Clinical T stage. Options: 'T1-T3a', 'T3b', 'T4'.")],
    psa: Annotated[float, Form(..., description="PSA level (ng/mL).")],
    gs: Annotated[str, Form(..., description="Gleason Score. Options: '≤7(3+4)', '7(4+3)', '8', '≥9'.")],
    nodes: Annotated[str, Form(..., description="Number of positive lymph nodes. Options: '1', '2', '≥3', 'No nodes were examined'.")],
    therapy: Annotated[str, Form(..., description="Radical prostatectomy. Options: 'Yes', 'No'.")],
    radio: Annotated[str, Form(..., description="Radiotherapy. Options: 'Yes', 'No'.")],
) -> PredictionOutput:
    """
    Evaluate the patient prognosis by accepting clinical variables, predicting survival probabilities,
    generating a survival curve, and providing a detailed explanation.
    """

    # --- Convert input strings to feature vectors ---
    # Age conversion: if '61-69' then [1,0,0], if '≤60' then [0,1,0], else '≥70' -> [0,0,1]
    if age == "61-69":
        age_vector = [1, 0, 0]
    elif age == "≤60":
        age_vector = [0, 1, 0]
    else:
        age_vector = [0, 0, 1]

    # Race
    if race == "Black":
        race_vector = [1, 0, 0]
    elif race == "Other":
        race_vector = [0, 1, 0]
    else:
        race_vector = [0, 0, 1]

    # Marital status
    if marital == "Married":
        marital_vector = [1, 0]
    else:
        marital_vector = [0, 1]

    # Clinical T stage
    if clinical == "T1-T3a":
        clinical_vector = [1, 0, 0]
    elif clinical == "T3b":
        clinical_vector = [0, 1, 0]
    else:
        clinical_vector = [0, 0, 1]

    # Gleason Score
    if gs == "7(4+3)":
        gs_vector = [1, 0, 0, 0]
    elif gs == "8":
        gs_vector = [0, 1, 0, 0]
    elif gs == "≤7(3+4)":
        gs_vector = [0, 0, 1, 0]
    else:  # '≥9'
        gs_vector = [0, 0, 0, 1]

    # Positive lymph nodes
    if nodes == "≥3":
        nodes_vector = [1, 0, 0, 0]
    elif nodes == "1":
        nodes_vector = [0, 0, 1, 0]
    elif nodes == "2":
        nodes_vector = [0, 0, 0, 1]
    else:  # "No nodes were examined"
        nodes_vector = [0, 1, 0, 0]

    # Therapy: Radical prostatectomy
    if therapy == "No":
        therapy_vector = [1, 0]
    else:
        therapy_vector = [0, 1]

    # Radiotherapy
    if radio == "No":
        radio_vector = [1, 0]
    else:
        radio_vector = [0, 1]

    # Combine all features (the order should match what the model expects)
    features = (
        age_vector + race_vector + marital_vector + clinical_vector +
        radio_vector + therapy_vector + nodes_vector + gs_vector
    )

    # Define the dataframe columns (the order must match the pre-trained model)
    columns = [
        "Age_61-69", "Age_<=60", "Age_>=70",
        "Race_Black", "Race_Other", "Race_White",
        "Marital_Married", "Marital_Unmarried",
        "CS.extension_T1_T3a", "CS.extension_T3b", "CS.extension_T4",
        "Radiation_None/Unknown", "Radiation_Yes",
        "Therapy_None", "Therapy_RP",
        "Nodes.positive_>=3", "Nodes.positive_None", "Nodes.positive_One", "Nodes.positive_Two",
        "Gleason.Patterns_4+3", "Gleason.Patterns_8", "Gleason.Patterns_<=3+4", "Gleason.Patterns_>=9"
    ]

    # Build the model input dataframe
    x_df = pd.DataFrame([features], columns=columns)
    # Include the PSA value in a separate DataFrame then concat
    x_psa_df = pd.DataFrame([[psa]], columns=["PSA"])
    x_test = pd.concat([x_df, x_psa_df], axis=1)

    # --- Predict survival probabilities ---
    # Although the model may also provide a "score", here we obtain the survival function
    surv = GBSA.predict_survival_function(x_test)
    # Retrieve survival probabilities at selected timepoints:
    time_points = [36, 60, 96, 119]
    survival_values = []
    for fn in surv:
        # For each time point, call the function (fn is assumed callable as in lifelines' SurvivalFunction)
        for t in time_points:
            survival_values.append(fn(t))

    survival_probabilities = {
        "36-month": survival_values[0],
        "60-month": survival_values[1],
        "96-month": survival_values[2],
        "119-month": survival_values[3],
    }

    # --- Plot the survival curve ---
    fig = plt.figure()
    for fn in surv:
        plt.plot(fn.x, fn(fn.x), label="Survival Curve")
    plt.xlim(0, 120)
    plt.ylabel("Survival probability")
    plt.xlabel("Survival time (months)")
    plt.grid(True)
    # (Optional) You may add a legend if more curves are expected
    plt.legend()

    # Convert figure to PNG-encoded base64 string:
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    survival_curve_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # --- Generate explanation using OpenAI's ChatCompletion ---
    # We use a concise prompt that includes the raw input values and calculated probabilities.
    prompt = f"""
As a medical professional, provide a comprehensive and empathetic interpretation of the survival probabilities for this prostate cancer patient.

Patient Characteristics:
- Age Group: {age}
- Race: {race}
- Marital Status: {marital}
- Clinical T Stage: {clinical}
- PSA Level: {psa} ng/mL
- Gleason Score: {gs}
- Positive Lymph Nodes: {nodes}
- Radical Prostatectomy: {"Yes" if therapy == "Yes" else "No"}
- Radiotherapy: {"Yes" if radio == "Yes" else "No"}

Survival Probabilities:
- 36-month survival: {survival_probabilities["36-month"]:.2%}
- 60-month survival: {survival_probabilities["60-month"]:.2%}
- 96-month survival: {survival_probabilities["96-month"]:.2%}
- 119-month survival: {survival_probabilities["119-month"]:.2%}

Please:
1. Contextualize these probabilities within the patient's clinical profile.
2. Explain the significance of each timepoint.
3. Discuss factors that could be influencing these outcomes.
4. Provide a balanced, hopeful, and medically sound interpretation.
5. Suggest potential next steps for the patient and the healthcare team.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Use the correct model specification
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical professional providing a detailed explanation of cancer survival predictions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        explanation = response.choices[0].message.content.strip()
    except Exception as e:
        # If the OpenAI call fails, return an error message in the explanation field.
        explanation = f"Error obtaining explanation from OpenAI: {e}"

    # --- Return the predictions and explanation ---
    return PredictionOutput(
        survival_probabilities=survival_probabilities,
        explanation=explanation,
        survival_curve=survival_curve_b64,
    )
