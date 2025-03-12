from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Annotated, List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI
import os

app = FastAPI(
    title="Machine Learning Prognostic Model for Prostate Cancer Patients with Lymph Node-positive",
    description="This API predicts survival probabilities for prostate cancer patients with lymph node-positive status using a pre-trained machine learning model.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
GBSA = joblib.load('GBSA12.18.pkl')

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("API_KEY"))

# Data point model for survival curve
class SurvivalTimePoint(BaseModel):
    """
    Data point representing survival probability at a specific time point.
    """
    time: int = Field(
        title="Time (months)",
        description="Time point in months from diagnosis."
    )
    probability: float = Field(
        title="Survival Probability",
        description="Predicted survival probability at the given time point."
    )

# Input Model
class PatientData(BaseModel):
    age: Literal['≤60', '61-69', '≥70'] = Field(..., 
        title="Age", 
        description="Patient's age group")
    
    race: Literal['White', 'Black', 'Other'] = Field(..., 
        title="Race", 
        description="Patient's race")
    
    marital: Literal['Married', 'Unmarried'] = Field(..., 
        title="Marital Status", 
        description="Patient's marital status")
    
    clinical: Literal['T1-T3a', 'T3b', 'T4'] = Field(..., 
        title="Clinical T stage", 
        description="Extent of the tumor")
    
    psa: float = Field(..., 
        title="PSA level", 
        ge=0.1, le=98.0, 
        description="Prostate-Specific Antigen level (ng/mL)")
    
    gs: Literal['≤7(3+4)', '7(4+3)', '8', '≥9'] = Field(..., 
        title="Gleason Score", 
        description="Score used to assess aggressiveness of prostate cancer")
    
    nodes: Literal['1', '2', '≥3', 'No nodes were examined'] = Field(..., 
        title="Number of positive lymph nodes", 
        description="Number of positive lymph nodes found")
    
    therapy: Literal['Yes', 'No'] = Field(..., 
        title="Radical prostatectomy", 
        description="Whether patient has undergone radical prostatectomy")
    
    radio: Literal['Yes', 'No'] = Field(..., 
        title="Radiotherapy", 
        description="Whether patient has received radiotherapy")

# Output Model
class SurvivalPrediction(BaseModel):
    survival_probabilities: Dict[str, float] = Field(..., 
        title="Survival Probabilities", 
        description="Predicted survival probabilities at various time points")
    
    survival_curve: List[SurvivalTimePoint] = Field(..., 
        title="Survival Curve Data", 
        description="Data points for plotting the survival curve")
    
    explanation: str = Field(..., 
        title="Explanation", 
        description="AI-generated explanation of the results")

def encode_one_hot(value, categories):
    """Convert categorical value to one-hot encoding"""
    result = [0] * len(categories)
    if value in categories:
        result[categories.index(value)] = 1
    return result

def get_explanation(patient_data, yv, age_encoding, race_encoding, clinical_encoding, 
                   gs_encoding, nodes_encoding, therapy_value, radio_value):
    """Generate explanation using OpenAI API"""
    
    # Format patient data for prompt
    age_str = patient_data.age
    race_str = patient_data.race
    clinical_str = patient_data.clinical
    gs_str = patient_data.gs
    nodes_str = patient_data.nodes
    
    prompt = f"""
    As a medical professional, provide a comprehensive and empathetic interpretation of the survival probabilities for this prostate cancer patient:

    Patient Characteristics:
    - Age Group: {age_str}
    - Race: {race_str}
    - Clinical T Stage: {clinical_str}
    - PSA Level: {patient_data.psa} ng/mL
    - Gleason Score: {gs_str}
    - Positive Lymph Nodes: {nodes_str}
    - Radical Prostatectomy: {therapy_value}
    - Radiotherapy: {radio_value}

    Survival Probabilities:
    - 36-month survival: {yv[0]:.2%}
    - 60-month survival: {yv[1]:.2%}
    - 96-month survival: {yv[2]:.2%}
    - 119-month survival: {yv[3]:.2%}

    Do the following:
    1. Contextualize these probabilities within the patient's specific clinical profile
    2. Explain the significance of each survival timepoint
    3. Discuss potential factors influencing these survival rates
    4. Provide a balanced, hopeful, and medically sound interpretation
    5. Suggest potential next steps or considerations for the patient and healthcare team

    Write in a clear, compassionate, and professional tone suitable for sharing with both healthcare providers and patients.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical professional providing detailed explanations of cancer survival predictions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    
    return response.choices[0].message.content.strip()

@app.post("/predict_survival", response_model=SurvivalPrediction, 
          summary="Predict Survival Probabilities")
async def predict_survival(
    background_tasks: BackgroundTasks,
    patient_data: PatientData
):
    """Predict survival probabilities for prostate cancer patients with lymph node-positive status."""
    
    # Convert categorical variables to one-hot encoding
    age_encoding = encode_one_hot(patient_data.age, ['≤60', '61-69', '≥70'])
    race_encoding = encode_one_hot(patient_data.race, ['White', 'Black', 'Other'])
    marital_encoding = encode_one_hot(patient_data.marital, ['Married', 'Unmarried'])
    clinical_encoding = encode_one_hot(patient_data.clinical, ['T1-T3a', 'T3b', 'T4'])
    radio_encoding = encode_one_hot(patient_data.radio, ['No', 'Yes'])
    therapy_encoding = encode_one_hot(patient_data.therapy, ['No', 'Yes'])
    nodes_encoding = encode_one_hot(patient_data.nodes, ['1', '2', '≥3', 'No nodes were examined'])
    gs_encoding = encode_one_hot(patient_data.gs, ['≤7(3+4)', '7(4+3)', '8', '≥9'])
    
    # Mapping categorical variables to column names
    column_mapping = {
        '≤60': 'Age_<=60', 
        '61-69': 'Age_61-69', 
        '≥70': 'Age_>=70',
        'White': 'Race_White', 
        'Black': 'Race_Black', 
        'Other': 'Race_Other',
        'Married': 'Marital_Married', 
        'Unmarried': 'Marital_Unmarried',
        'T1-T3a': 'CS.extension_T1_T3a', 
        'T3b': 'CS.extension_T3b', 
        'T4': 'CS.extension_T4',
        'No': 'Radiation_None/Unknown' if patient_data.radio == 'No' else 'Therapy_None',
        'Yes': 'Radiation_Yes' if patient_data.radio == 'Yes' else 'Therapy_RP',
        '1': 'Nodes.positive_One',
        '2': 'Nodes.positive_Two',
        '≥3': 'Nodes.positive_>=3',
        'No nodes were examined': 'Nodes.positive_None',
        '≤7(3+4)': 'Gleason.Patterns_<=3+4',
        '7(4+3)': 'Gleason.Patterns_4+3',
        '8': 'Gleason.Patterns_8',
        '≥9': 'Gleason.Patterns_>=9'
    }
    
    # Initialize dataframe with all columns set to 0
    columns = [
        'Age_61-69', 'Age_<=60', 'Age_>=70',
        'Race_Black', 'Race_Other', 'Race_White',
        'Marital_Married', 'Marital_Unmarried',
        'CS.extension_T1_T3a', 'CS.extension_T3b', 'CS.extension_T4',
        'Radiation_None/Unknown', 'Radiation_Yes',
        'Therapy_None', 'Therapy_RP',
        'Nodes.positive_>=3', 'Nodes.positive_None', 'Nodes.positive_One', 'Nodes.positive_Two',
        'Gleason.Patterns_4+3', 'Gleason.Patterns_8', 'Gleason.Patterns_<=3+4', 'Gleason.Patterns_>=9',
        'PSA'
    ]
    x_test = pd.DataFrame(0, index=[0], columns=columns)
    
    # Set values based on patient data
    x_test[column_mapping[patient_data.age]] = 1
    x_test[column_mapping[patient_data.race]] = 1
    x_test[column_mapping[patient_data.marital]] = 1
    x_test[column_mapping[patient_data.clinical]] = 1
    x_test[column_mapping[patient_data.radio]] = 1
    x_test[column_mapping[patient_data.therapy]] = 1
    x_test[column_mapping[patient_data.nodes]] = 1
    x_test[column_mapping[patient_data.gs]] = 1
    x_test['PSA'] = patient_data.psa
    
    # Make prediction
    surv = GBSA.predict_survival_function(x_test)
    
    # Extract survival probabilities at specific time points
    yv = []
    fn = next(iter(surv))
    time_points = [36, 60, 96, 119]
    
    for time_point in time_points:
        closest_idx = min(range(len(fn.x)), key=lambda i: abs(fn.x[i] - time_point))
        yv.append(fn(fn.x)[closest_idx])
    
    # Get explanation
    explanation = get_explanation(
        patient_data, yv, age_encoding, race_encoding, clinical_encoding, 
        gs_encoding, nodes_encoding, patient_data.therapy, patient_data.radio
    )
    
    # Prepare survival probabilities for specific timepoints
    survival_probs = {
        "36-month": float(yv[0]),
        "60-month": float(yv[1]),
        "96-month": float(yv[2]),
        "119-month": float(yv[3])
    }
    
    # Prepare survival curve data points
    survival_curve_points = []
    for i in range(len(fn.x)):
        survival_curve_points.append(
            SurvivalTimePoint(
                time=int(fn.x[i]),
                probability=float(fn(fn.x)[i])
            )
        )
    
    return SurvivalPrediction(
        survival_probabilities=survival_probs,
        survival_curve=survival_curve_points,
        explanation=explanation
    )
