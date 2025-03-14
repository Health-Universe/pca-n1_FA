from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Annotated
import pandas as pd
import numpy as np
import joblib
import os
import openai
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.figure import Figure

# Set up OpenAI API
openai.api_key = os.getenv("API_KEY")

# Configure the FastAPI app
app = FastAPI(
    title="Machine Learning Prognostic Model for the Overall Survival of Prostate Cancer Patients with Lymph Node-positive",
    description="This API provides survival predictions for prostate cancer patients with lymph node-positive status.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
GBSA = joblib.load('GBSA12.18.pkl')

# Define the prediction result model
class PredictionResult(BaseModel):
    qualification_result: str = Field(..., title="Prediction Result", description="Result of the prediction process")
    survival_probabilities: Dict[str, float] = Field(..., title="Survival Probabilities", description="Predicted survival probabilities at different time points")
    survival_curve: str = Field(..., title="Survival Curve", description="Base64-encoded image of the survival curve")
    explanation: str = Field(..., title="Explanation", description="Medical explanation of the prediction results")

# Helper function to encode the survival curve as a base64 string
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

# Helper function to generate explanation using OpenAI
def generate_explanation(age_text, race, clinical_t_stage, psa_level, gleason_score, positive_lymph_nodes, rp_text, rt_text, yv):
    prompt = f"""
    As a medical professional, provide a comprehensive and empathetic interpretation of the survival probabilities for this prostate cancer patient:

    Patient Characteristics:
    - Age Group: {age_text}
    - Race: {race}
    - Clinical T Stage: {clinical_t_stage}
    - PSA Level: {psa_level} ng/mL
    - Gleason Score: {gleason_score}
    - Positive Lymph Nodes: {positive_lymph_nodes}
    - Radical Prostatectomy: {rp_text}
    - Radiotherapy: {rt_text}

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
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Using GPT-4o as in the original code
            messages=[
                {"role": "system", "content": "You are a medical professional providing detailed explanations of cancer survival predictions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        explanation = response.choices[0].message.content.strip()
    except Exception as e:
        explanation = f"Error generating explanation: {str(e)}"
    
    return explanation

# Single endpoint for prediction
@app.post("/predict/", response_model=PredictionResult)
async def predict(
    age_group: Annotated[str, Form()],
    race: Annotated[str, Form()],
    marital_status: Annotated[str, Form()],
    clinical_t_stage: Annotated[str, Form()],
    psa_level: Annotated[float, Form()],
    gleason_score: Annotated[str, Form()],
    positive_lymph_nodes: Annotated[str, Form()],
    radical_prostatectomy: Annotated[str, Form()],
    radiotherapy: Annotated[str, Form()]
):
    try:
        # Age encoding
        if age_group == '61-69':
            age = [1, 0, 0]
            age_text = "61-69 years"
        elif age_group == '≤60':
            age = [0, 1, 0]
            age_text = "≤60 years"
        else:  # ≥70
            age = [0, 0, 1]
            age_text = "≥70 years"
        
        # Race encoding
        if race == 'Black':
            race_encoded = [1, 0, 0]
        elif race == 'Other':
            race_encoded = [0, 1, 0]
        else:  # White
            race_encoded = [0, 0, 1]
        
        # Marital status encoding
        if marital_status == 'Married':
            marital = [1, 0]
        else:  # Unmarried
            marital = [0, 1]
        
        # Clinical T stage encoding
        if clinical_t_stage == 'T1-T3a':
            clinical = [1, 0, 0]
        elif clinical_t_stage == 'T3b':
            clinical = [0, 1, 0]
        else:  # T4
            clinical = [0, 0, 1]
        
        # Gleason score encoding
        if gleason_score == '7(4+3)':
            gs = [1, 0, 0, 0]
        elif gleason_score == '8':
            gs = [0, 1, 0, 0]
        elif gleason_score == '≤7(3+4)':
            gs = [0, 0, 1, 0]
        else:  # ≥9
            gs = [0, 0, 0, 1]
        
        # Positive lymph nodes encoding
        if positive_lymph_nodes == '≥3':
            nodes = [1, 0, 0, 0]
        elif positive_lymph_nodes == '1':
            nodes = [0, 0, 1, 0]
        elif positive_lymph_nodes == '2':
            nodes = [0, 0, 0, 1]
        else:  # No nodes were examined
            nodes = [0, 1, 0, 0]
        
        # Radical prostatectomy encoding
        if radical_prostatectomy == 'No':
            therapy = [1, 0]
            rp_text = "No"
        else:  # Yes
            therapy = [0, 1]
            rp_text = "Yes"
        
        # Radiotherapy encoding
        if radiotherapy == 'No':
            radio = [1, 0]
            rt_text = "No"
        else:  # Yes
            radio = [0, 1]
            rt_text = "Yes"
        
        # Combine all features
        features = []
        features.extend(age)
        features.extend(race_encoded)
        features.extend(marital)
        features.extend(clinical)
        features.extend(radio)
        features.extend(therapy)
        features.extend(nodes)
        features.extend(gs)
        
        # Create DataFrame
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
        x_psa_df = pd.DataFrame([psa_level], columns=['PSA'])
        
        # Combine all features
        x_test = pd.concat([x_df, x_psa_df], axis=1)
        
        # Make prediction
        try:
            prob = GBSA.predict(x_test)
            surv = GBSA.predict_survival_function(x_test)
            
            # Extract survival probabilities at specific time points
            yv = []
            for fn in surv:
                for i in range(0, len(fn.x)):
                    if fn.x[i] in (36, 60, 96, 119):
                        yv.append(fn(fn.x)[i])
            
            # Create survival curve plot
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.plot(fn.x[0:120], fn(fn.x)[0:120])
            ax.set_xlim(0, 120)
            ax.set_ylabel("Survival probability")
            ax.set_xlabel("Survival time (mo)")
            ax.grid(True)
            
            # Convert plot to base64
            survival_curve_base64 = plot_to_base64(fig)
            
            # Generate explanation
            explanation = generate_explanation(
                age_text, race, clinical_t_stage, psa_level, 
                gleason_score, positive_lymph_nodes, rp_text, rt_text, yv
            )
            
            # Create response
            result = PredictionResult(
                qualification_result="Prediction Successful",
                survival_probabilities={
                    "36-month": float(yv[0]),
                    "60-month": float(yv[1]),
                    "96-month": float(yv[2]),
                    "119-month": float(yv[3])
                },
                survival_curve=survival_curve_base64,
                explanation=explanation
            )
            
            return result
            
        except Exception as e:
            # For demonstration purposes, return mock data if model is not available
            print(f"Error making prediction: {str(e)}")
            mock_yv = [0.85, 0.72, 0.61, 0.52]
            
            # Create mock survival curve
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            x = np.arange(0, 120)
            y = 1 * np.exp(-0.005 * x)
            ax.plot(x, y)
            ax.set_xlim(0, 120)
            ax.set_ylabel("Survival probability")
            ax.set_xlabel("Survival time (mo)")
            ax.grid(True)
            
            # Convert plot to base64
            survival_curve_base64 = plot_to_base64(fig)
            
            # Generate explanation
            explanation = generate_explanation(
                age_text, race, clinical_t_stage, psa_level, 
                gleason_score, positive_lymph_nodes, rp_text, rt_text, mock_yv
            )
            
            # Create response
            result = PredictionResult(
                qualification_result="Prediction Completed (Mock Data)",
                survival_probabilities={
                    "36-month": float(mock_yv[0]),
                    "60-month": float(mock_yv[1]),
                    "96-month": float(mock_yv[2]),
                    "119-month": float(mock_yv[3])
                },
                survival_curve=survival_curve_base64,
                explanation=explanation
            )
            
            return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Add root endpoint to guide users
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Prostate Cancer Survival Prediction API",
        "usage": "Send a POST request to /predict/ with the required patient data"
    }
