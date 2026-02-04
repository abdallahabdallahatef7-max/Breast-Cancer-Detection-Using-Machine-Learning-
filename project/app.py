from fastapi import FastAPI
import pickle
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import numpy as np


app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open("model/log.pkl", "rb") as f:
    model = pickle.load(f)


FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "features": FEATURES})

@app.post("/predict")
async def predict(request: Request):

    form_data = await request.form()
    
   
    input_values = [float(form_data[feature]) for feature in FEATURES]
    
  
    final_features = np.array(input_values).reshape(1, -1)
    
    prediction = model.predict(final_features)
    result = "Cancerous (خبيث)" if prediction[0] == 0 else "Benign (حميد)"

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "prediction_text": result,
        "features": FEATURES 
    })