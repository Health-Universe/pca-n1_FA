import os
import traceback
from typing import Annotated, Literal
from fastapi import FastAPI, Depends, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import base64
import openai
from fastapi.responses import JSONResponse, HTMLResponse
import httpx

app = FastAPI(
    title="Prostate Cancer Survival Prognostic Model API",
    description="API for predicting survival probabilities for prostate cancer patients with lymph node-positive status",
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

# Load model
try:
    GBSA = joblib.load('GBSA12.18.pkl')
except FileNotFoundError:
    print("Error: Model file 'GBSA12.18.pkl' not found.")
    GBSA = None

# OpenAI API setup
openai.api_key = os.getenv("API_KEY")

class PatientInput(BaseModel):
    age: Annotated[Literal['≤60', '61-69', '≥70'], Form()] = '≥70'
    race: Annotated[Literal['White', 'Black', 'Other'], Form()] = 'White'
    marital: Annotated[Literal['Married', 'Unmarried'], Form()] = 'Married'
    clinical: Annotated[Literal['T1-T3a', 'T3b', 'T4'], Form()] = 'T3b'
    psa: Annotated[float, Form()] = 20.0
    gs: Annotated[Literal['≤7(3+4)', '7(4+3)', '8', '≥9'], Form()] = '≥9'
    nodes: Annotated[Literal['1', '2', '≥3', 'No nodes were examined'], Form()] = '≥3'
    therapy: Annotated[Literal['Yes', 'No'], Form()] = 'Yes'
    radio: Annotated[Literal['Yes', 'No'], Form()] = 'No'

    @classmethod
    def as_form(
        cls, 
        age: Annotated[Literal['≤60', '61-69', '≥70'], Form()] = '≥70',
        race: Annotated[Literal['White', 'Black', 'Other'], Form()] = 'White',
        marital: Annotated[Literal['Married', 'Unmarried'], Form()] = 'Married',
        clinical: Annotated[Literal['T1-T3a', 'T3b', 'T4'], Form()] = 'T3b',
        psa: Annotated[float, Form()] = 20.0,
        gs: Annotated[Literal['≤7(3+4)', '7(4+3)', '8', '≥9'], Form()] = '≥9',
        nodes: Annotated[Literal['1', '2', '≥3', 'No nodes were examined'], Form()] = '≥3',
        therapy: Annotated[Literal['Yes', 'No'], Form()] = 'Yes',
        radio: Annotated[Literal['Yes', 'No'], Form()] = 'No'
    ):
        return cls(
            age=age,
            race=race,
            marital=marital,
            clinical=clinical,
            psa=psa,
            gs=gs,
            nodes=nodes,
            therapy=therapy,
            radio=radio
        )

class SurvivalOutput(BaseModel):
    survival_probabilities: dict
    survival_plot: str
    interpretation: str

def map_age(age):
    mapping = {
        '61-69': [1, 0, 0], 
        '≤60': [0, 1, 0], 
        '≥70': [0, 0, 1]
    }
    return mapping.get(age, [0, 0, 1])

def map_race(race):
    mapping = {
        'Black': [1, 0, 0], 
        'Other': [0, 1, 0], 
        'White': [0, 0, 1]
    }
    return mapping.get(race, [0, 0, 1])

def map_marital(marital):
    mapping = {
        'Married': [1, 0],
        'Unmarried': [0, 1]
    }
    return mapping.get(marital, [1, 0])

def map_clinical(clinical):
    mapping = {
        'T1-T3a': [1, 0, 0],
        'T3b': [0, 1, 0],
        'T4': [0, 0, 1]
    }
    return mapping.get(clinical, [0, 1, 0])

def map_gs(gs):
    mapping = {
        '7(4+3)': [1, 0, 0, 0],
        '8': [0, 1, 0, 0],
        '≤7(3+4)': [0, 0, 1, 0],
        '≥9': [0, 0, 0, 1]
    }
    return mapping.get(gs, [0, 0, 0, 1])

def map_nodes(nodes):
    mapping = {
        '≥3': [1, 0, 0, 0],
        '1': [0, 0, 1, 0],
        '2': [0, 0, 0, 1],
        'No nodes were examined': [0, 1, 0, 0]
    }
    return mapping.get(nodes, [1, 0, 0, 0])

def map_therapy(therapy):
    mapping = {
        'No': [1, 0],
        'Yes': [0, 1]
    }
    return mapping.get(therapy, [0, 1])

def map_radio(radio):
    mapping = {
        'No': [1, 0],
        'Yes': [0, 1]
    }
    return mapping.get(radio, [1, 0])

@app.post("/predict_survival", response_model=SurvivalOutput)
async def predict_survival(
    patient_data: PatientInput = Depends(PatientInput.as_form)
):
    try:
        if GBSA is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Convert inputs to one-hot encoded lists
        age_list = map_age(patient_data.age)
        race_list = map_race(patient_data.race)
        marital_list = map_marital(patient_data.marital)
        clinical_list = map_clinical(patient_data.clinical)
        gs_list = map_gs(patient_data.gs)
        nodes_list = map_nodes(patient_data.nodes)
        therapy_list = map_therapy(patient_data.therapy)
        radio_list = map_radio(patient_data.radio)

        # Construct input vector
        input_vector = (
            age_list + race_list + marital_list + clinical_list + 
            radio_list + therapy_list + nodes_list + gs_list
        )
        
        # Create DataFrame for model
        x_df = pd.DataFrame([input_vector], columns=[
            'Age_61-69', 'Age_<=60', 'Age_>=70', 'Race_Black', 'Race_Other', 'Race_White',
            'Marital_Married', 'Marital_Unmarried', 'CS.extension_T1_T3a', 'CS.extension_T3b', 'CS.extension_T4',
            'Radiation_None/Unknown', 'Radiation_Yes', 'Therapy_None', 'Therapy_RP',
            'Nodes.positive_>=3', 'Nodes.positive_None', 'Nodes.positive_One', 'Nodes.positive_Two',
            'Gleason.Patterns_4+3', 'Gleason.Patterns_8', 'Gleason.Patterns_<=3+4', 'Gleason.Patterns_>=9'
        ])
        
        # Add PSA value to DataFrame
        x_psa_df = pd.DataFrame([patient_data.psa], columns=['PSA'])
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
        - Age Group: {patient_data.age}
        - Race: {patient_data.race}
        - Clinical T Stage: {patient_data.clinical}
        - PSA Level: {patient_data.psa} ng/mL
        - Gleason Score: {patient_data.gs}
        - Positive Lymph Nodes: {patient_data.nodes}
        - Radical Prostatectomy: {patient_data.therapy}
        - Radiotherapy: {patient_data.radio}

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

    except Exception as e:
        print(f"Error in prediction: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": GBSA is not None,
        "openai_api_key_set": bool(openai.api_key)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
