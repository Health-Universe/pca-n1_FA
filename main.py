from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Annotated, List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import openai
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

# Set OpenAI API key
openai.api_key = os.getenv("API_KEY")

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
    survival_probabilities: str = Field(..., 
        title="Survival Probabilities", 
        description="Predicted survival probabilities at various time points")
    
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
    age_str = "≤60" if age_encoding[1] == 1 else "61-69" if age_encoding[0] == 1 else "≥70"
    race_str = "Black" if race_encoding[0] == 1 else "Other" if race_encoding[1] == 1 else "White"
    clinical_str = "T1-T3a" if clinical_encoding[0] == 1 else "T3b" if clinical_encoding[1] == 1 else "T4"
    
    gs_str = "7(4+3)" if gs_encoding[0] == 1 else "8" if gs_encoding[1] == 1 else "≤7(3+4)" if gs_encoding[2] == 1 else "≥9"
    
    nodes_str = "≥3" if nodes_encoding[0] == 1 else "No nodes examined" if nodes_encoding[1] == 1 else "1" if nodes_encoding[2] == 1 else "2"
    
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
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Using the model from your original code
        messages=[
            {"role": "system", "content": "You are a medical professional providing detailed explanations of cancer survival predictions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    
    return response.choices[0].message.content.strip()

@app.post("/predict_survival", response_model=SurvivalPrediction, 
          summary="Predict Survival Probabilities",
          openapi_extra={"x-chart-type": "line_chart"})
async def predict_survival(
    background_tasks: BackgroundTasks,
    patient_data: Annotated[PatientData, Form()]
):
    """Predict survival probabilities for prostate cancer patients with lymph node-positive status."""
    
    # Convert categorical variables to one-hot encoding
    age_encoding = encode_one_hot(patient_data.age, ['61-69', '≤60', '≥70'])
    race_encoding = encode_one_hot(patient_data.race, ['Black', 'Other', 'White'])
    marital_encoding = encode_one_hot(patient_data.marital, ['Married', 'Unmarried'])
    clinical_encoding = encode_one_hot(patient_data.clinical, ['T1-T3a', 'T3b', 'T4'])
    radio_encoding = encode_one_hot(patient_data.radio, ['No', 'Yes'])
    therapy_encoding = encode_one_hot(patient_data.therapy, ['No', 'Yes'])
    nodes_encoding = encode_one_hot(patient_data.nodes, ['≥3', 'No nodes were examined', '1', '2'])
    gs_encoding = encode_one_hot(patient_data.gs, ['7(4+3)', '8', '≤7(3+4)', '≥9'])
    
    # Combine all features
    features = []
    features.extend(age_encoding)
    features.extend(race_encoding)
    features.extend(marital_encoding)
    features.extend(clinical_encoding)
    features.extend(radio_encoding)
    features.extend(therapy_encoding)
    features.extend(nodes_encoding)
    features.extend(gs_encoding)
    
    # Create feature dataframe
    x_df = pd.DataFrame([features], columns=[
        'Age_61-69', 'Age_<=60', 'Age_>=70',
        'Race_Black', 'Race_Other', 'Race_White',
        'Marital_Married', 'Marital_Unmarried',
        'CS.extension_T1_T3a', 'CS.extension_T3b', 'CS.extension_T4',
        'Radiation_None/Unknown', 'Radiation_Yes',
        'Therapy_None', 'Therapy_RP',
        'Nodes.positive_>=3', 'Nodes.positive_None', 'Nodes.positive_One', 'Nodes.positive_Two',
        'Gleason.Patterns_4+3', 'Gleason.Patterns_8', 'Gleason.Patterns_<=3+4', 'Gleason.Patterns_>=9'
    ])
    
    # Add PSA level
    x_psa_df = pd.DataFrame([patient_data.psa], columns=['PSA'])
    x_test = pd.concat([x_df, x_psa_df], axis=1)
    
    # Make prediction
    prob = GBSA.predict(x_test)
    surv = GBSA.predict_survival_function(x_test)
    
    # Extract survival probabilities at specific time points
    yv = []
    fn = next(iter(surv))
    for i in range(0, len(fn.x)):
        if fn.x[i] in (36, 60, 96, 119):
            yv.append(fn(fn.x)[i])
    
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
    
    survival_probabilities_str = ", ".join([f"{key}: {value}%" for key, value in survival_probs.items()]) 
    
    # Prepare survival curve data points (similar to the caloric calculator approach)
    survival_curve_points = []
    for i in range(0, 120):  # 0 to 119 months
        if i < len(fn.x):
            survival_curve_points.append(
                SurvivalTimePoint(
                    time=int(fn.x[i]),
                    probability=float(fn(fn.x)[i])
                )
            )
    
    return SurvivalPrediction(
        survival_probabilities=survival_probabilities_str,
        explanation=explanation
    )
