import os
from typing import Annotated, List
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import JSONResponse, HTMLResponse
import httpx
import openai

app = FastAPI(
    title="Prostate Cancer Survival Prognostic Model API",
    description="API for predicting survival probabilities for prostate cancer patients based on their characteristics.",
    version="1.0.0",
)

# CORS policy setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model (make sure that 'GBSA12.18.pkl' is in your path or provide absolute path)
GBSA = joblib.load('GBSA12.18.pkl')

# Get the OpenAI API key from the environment
openai.api_key = os.environ.get("API_KEY")
if not openai.api_key:
    raise RuntimeError("OpenAI API_KEY environment variable is not set.")

class PatientInput(BaseModel):
    age: str
    race: str
    marital: str
    clinical: str
    psa: float
    gs: str
    nodes: str
    therapy: str
    radio: str

class SurvivalOutput(BaseModel):
    survival_probabilities: dict
    survival_plot: str
    interpretation: str

@app.post(
    "/predict_survival",
    response_model=SurvivalOutput,
    description="Predict prostate cancer survival based on patient data and return survival probabilities and a plot."
)
async def predict_survival(
    age: Annotated[str, Form()],
    race: Annotated[str, Form()],
    marital: Annotated[str, Form()],
    clinical: Annotated[str, Form()],
    psa: Annotated[float, Form()],
    gs: Annotated[str, Form()],
    nodes: Annotated[str, Form()],
    therapy: Annotated[str, Form()],
    radio: Annotated[str, Form()]
) -> SurvivalOutput:
    
    # Process categorical inputs
    # Age, Race, Marital status, Clinical T Stage processing logic remains the same as in Streamlit
    # Let's assume your preprocessing mapping logic is the same
    age_mapping = {'≤60': [0, 1, 0], '61-69': [1, 0, 0], '≥70': [0, 0, 1]}
    race_mapping = {'White': [0, 0, 1], 'Black': [1, 0, 0], 'Other': [0, 1, 0]}
    marital_mapping = {'Married': [1, 0], 'Unmarried': [0, 1]}
    clinical_mapping = {'T1-T3a': [1, 0, 0], 'T3b': [0, 1, 0], 'T4': [0, 0, 1]}
    gs_mapping = {'≤7(3+4)': [0, 0, 1, 0], '7(4+3)': [1, 0, 0, 0], '8': [0, 1, 0, 0], '≥9': [0, 0, 0, 1]}
    nodes_mapping = {'1': [0, 0, 1, 0], '2': [0, 0, 0, 1], '≥3': [1, 0, 0, 0], 'No nodes were examined': [0, 1, 0, 0]}
    therapy_mapping = {'Yes': [0, 1], 'No': [1, 0]}
    radio_mapping = {'Yes': [0, 1], 'No': [1, 0]}
    
    # Construct input vector for model
    input_vector = age_mapping[age] + race_mapping[race] + marital_mapping[marital] + clinical_mapping[clinical] + \
                   radio_mapping[radio] + therapy_mapping[therapy] + nodes_mapping[nodes] + gs_mapping[gs]
    
    # Create DataFrame for model
    x_df = pd.DataFrame([input_vector], columns=[
        'Age_61-69', 'Age_<=60', 'Age_>=70', 'Race_Black', 'Race_Other', 'Race_White',
        'Marital_Married', 'Marital_Unmarried', 'CS.extension_T1_T3a', 'CS.extension_T3b', 'CS.extension_T4',
        'Radiation_None/Unknown', 'Radiation_Yes', 'Therapy_None', 'Therapy_RP',
        'Nodes.positive_>=3', 'Nodes.positive_None', 'Nodes.positive_One', 'Nodes.positive_Two',
        'Gleason.Patterns_4+3', 'Gleason.Patterns_8', 'Gleason.Patterns_<=3+4', 'Gleason.Patterns_>=9'
    ])
    
    # Add PSA value to DataFrame
    x_psa_df = pd.DataFrame([psa], columns=['PSA'])
    x_test = pd.concat([x_df, x_psa_df], axis=1)
    
    # Get survival probabilities
    prob = GBSA.predict(x_test)
    surv_func = GBSA.predict_survival_function(x_test)

    # Prepare survival probabilities for different time points
    yv = []

    for fn in surv_func:
        for i, month in enumerate([36, 60, 96, 119]):
            if month in fn.x:
                yv.append(fn(fn.x)[i])
    
    survival_probabilities = {
        "36-month": yv[0] if len(yv) > 0 else 0,
        "60-month": yv[1] if len(yv) > 1 else 0,
        "96-month": yv[2] if len(yv) > 2 else 0,
        "119-month": yv[3] if len(yv) > 3 else 0
    }

    # Generate plot
    fig, ax = plt.subplots()
    ax.plot(fn.x[0:120], fn(fn.x)[0:120], label="Survival curve")
    ax.set_xlim(0, 120)
    ax.set_ylabel("Survival probability")
    ax.set_xlabel("Survival time (months)")
    ax.legend()
    ax.grid(True)

    # Encode plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()

    # Generate explanation using OpenAI
    prompt = f"""
    As a medical professional, provide a comprehensive and empathetic interpretation of the survival probabilities for this prostate cancer patient:

    Patient Characteristics:
    - Age Group: {age}
    - Race: {race}
    - Clinical T Stage: {clinical}
    - PSA Level: {psa} ng/mL
    - Gleason Score: {gs}
    - Positive Lymph Nodes: {nodes}
    - Radical Prostatectomy: {'Yes' if therapy_mapping['Yes'] == [0, 1] else 'No'}
    - Radiotherapy: {'Yes' if radio_mapping['Yes'] == [0, 1] else 'No'}

    Survival Probabilities:
    - 36-month survival: {survival_probabilities['36-month']:.2%}
    - 60-month survival: {survival_probabilities['60-month']:.2%}
    - 96-month survival: {survival_probabilities['96-month']:.2%}
    - 119-month survival: {survival_probabilities['119-month']:.2%}

    Do the following:
    1. Contextualize these probabilities within the patient's specific clinical profile.
    2. Explain the significance of each survival timepoint.
    3. Discuss potential factors influencing these survival rates.
    4. Provide a balanced, hopeful, and medically sound interpretation.
    5. Suggest potential next steps or considerations for the patient and healthcare team.

    Write in a clear, compassionate, and professional tone suitable for sharing with both healthcare providers and patients.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical professional providing detailed explanations of cancer survival predictions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    interpretation = response.choices[0].message.content.strip()

    return SurvivalOutput(
        survival_probabilities=survival_probabilities,
        survival_plot=img_str,
        interpretation=interpretation,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
