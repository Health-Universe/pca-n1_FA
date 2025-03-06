import io
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Form, BackgroundTasks, HTTPException
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
try:
    GBSA = joblib.load('GBSA12.18.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Error loading the model")

# Set OpenAI API key
openai.api_key = os.getenv("API_KEY")

# Data point model for survival curve
class SurvivalTimePoint(BaseModel):
    time: int = Field(..., title="Time (months)", description="Time point in months from diagnosis.")
    probability: float = Field(..., title="Survival Probability", description="Predicted survival probability at the given time point.")

# Input model for survival prediction
class SurvivalPrediction(BaseModel):
    survival_probabilities: Dict[str, float] = Field(..., title="Survival Probabilities", description="Predicted survival probabilities at various time points.")
    survival_curve: List[SurvivalTimePoint] = Field(..., title="Survival Curve Data", description="Data points for plotting the survival curve.")
    explanation: str = Field(..., title="Explanation", description="AI-generated explanation of the results.")

# Prediction function for survival probabilities
def encode_one_hot(value, categories):
    result = [0] * len(categories)
    if value in categories:
        result[categories.index(value)] = 1
    return result

def get_explanation(patient_data, yv, age_encoding, race_encoding, clinical_encoding, gs_encoding, nodes_encoding, therapy_value, radio_value):
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

@app.post("/predict_survival", response_model=SurvivalPrediction, summary="Predict Survival Probabilities and Plot Survival Curve")
async def predict_survival(background_tasks: BackgroundTasks, patient_data: Annotated[PatientData, Form()]):
    try:
        age_encoding = encode_one_hot(patient_data.age, ['61-69', '≤60', '≥70'])
        race_encoding = encode_one_hot(patient_data.race, ['Black', 'Other', 'White'])
        marital_encoding = encode_one_hot(patient_data.marital, ['Married', 'Unmarried'])
        clinical_encoding = encode_one_hot(patient_data.clinical, ['T1-T3a', 'T3b', 'T4'])
        radio_encoding = encode_one_hot(patient_data.radio, ['No', 'Yes'])
        therapy_encoding = encode_one_hot(patient_data.therapy, ['No', 'Yes'])
        nodes_encoding = encode_one_hot(patient_data.nodes, ['≥3', 'No nodes were examined', '1', '2'])
        gs_encoding = encode_one_hot(patient_data.gs, ['7(4+3)', '8', '≤7(3+4)', '≥9'])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing patient data: {e}")
    
    try:
        features = []
        features.extend(age_encoding)
        features.extend(race_encoding)
        features.extend(marital_encoding)
        features.extend(clinical_encoding)
        features.extend(radio_encoding)
        features.extend(therapy_encoding)
        features.extend(nodes_encoding)
        features.extend(gs_encoding)
        
        x_df = pd.DataFrame([features], columns=[
            'Age_61-69', 'Age_<=60', 'Age_>=70', 'Race_Black', 'Race_Other', 'Race_White', 
            'Marital_Married', 'Marital_Unmarried', 'CS.extension_T1_T3a', 'CS.extension_T3b', 
            'CS.extension_T4', 'Radiation_None/Unknown', 'Radiation_Yes', 'Therapy_None', 
            'Therapy_RP', 'Nodes.positive_>=3', 'Nodes.positive_None', 'Nodes.positive_One', 
            'Nodes.positive_Two', 'Gleason.Patterns_4+3', 'Gleason.Patterns_8', 
            'Gleason.Patterns_<=3+4', 'Gleason.Patterns_>=9'
        ])
        
        x_psa_df = pd.DataFrame([patient_data.psa], columns=['PSA'])
        x_test = pd.concat([x_df, x_psa_df], axis=1)
        
        prob = GBSA.predict(x_test)
        surv = GBSA.predict_survival_function(x_test)
        
        yv = []
        fn = next(iter(surv))
        for i in range(0, len(fn.x)):
            if fn.x[i] in (36, 60, 96, 119):
                yv.append(fn(fn.x)[i])
        
        explanation = get_explanation(
            patient_data, yv, age_encoding, race_encoding, clinical_encoding, 
            gs_encoding, nodes_encoding, patient_data.therapy, patient_data.radio
        )
        
        survival_probs = {
            "36-month": float(yv[0]),
            "60-month": float(yv[1]),
            "96-month": float(yv[2]),
            "119-month": float(yv[3])
        }
        
        survival_curve_points = []
        for i in range(0, 120):
            if i < len(fn.x):
                survival_curve_points.append(SurvivalTimePoint(time=int(fn.x[i]), probability=float(fn(fn.x)[i])))

        # Plot the survival curve
        plt.figure(figsize=(8, 6))
        plt.plot(fn.x, fn(fn.x), label="Survival Probability", color='blue')
        plt.title("Survival Curve")
        plt.xlabel("Time (months)")
        plt.ylabel("Survival Probability")
        plt.grid(True)
        plt.axhline(0.5, color='red', linestyle='--', label='50% Survival Probability')

        # Save the plot to a byte stream
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        
        # Return the data and the image in the response
        return {
            "survival_probabilities": survival_probs,
            "survival_curve": survival_curve_points,
            "explanation": explanation,
            "survival_curve_image": StreamingResponse(buf, media_type="image/png")
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction process: {e}")
