# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import joblib

# # Load model and column list
# model = joblib.load("logistic_model.pkl")
# model_columns = joblib.load("model_columns.pkl")

# app = FastAPI()

# # Step 1: Define raw user input schema
# class ClientData(BaseModel):
#     age: int
#     job: str
#     marital: str
#     education: str
#     default: str
#     balance: float
#     housing: str
#     loan: str
#     contact: str
#     day: int
#     month: str
#     campaign: int
#     pdays: int
#     previous: int
#     poutcome: str

# @app.post("/predict")
# def predict(data: ClientData):
#     # Step 2: Convert input to DataFrame
#     input_dict = data.dict()
#     df = pd.DataFrame([input_dict])
    
#     # Step 3: One-hot encode categorical variables
#     df_encoded = pd.get_dummies(df)

#     # Step 4: Align columns to match training data
#     df_aligned = pd.DataFrame(columns=model_columns)
#     df_aligned.loc[0] = 0  # initialize with zeros

#     for col in df_encoded.columns:
#         if col in df_aligned.columns:
#             df_aligned[col] = df_encoded[col]

#     # Step 5: Predict
#     prediction = model.predict(df_aligned)[0]
#     return {"prediction": "yes" if prediction == 1 else "no"}


from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
from fastapi.staticfiles import StaticFiles

# Initialize app and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and training columns
model = joblib.load("logistic_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
    age: int = Form(...),
    job: str = Form(...),
    marital: str = Form(...),
    education: str = Form(...),
    default: str = Form(...),
    balance: float = Form(...),
    housing: str = Form(...),
    loan: str = Form(...),
    contact: str = Form(...),
    day: int = Form(...),
    month: str = Form(...),
    campaign: int = Form(...),
    pdays: int = Form(...),
    previous: int = Form(...),
    poutcome: str = Form(...)
):
    # Convert form input to DataFrame
    input_dict = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }
    
    print(input_dict)

    df = pd.DataFrame([input_dict])
    df_encoded = pd.get_dummies(df)

    df_aligned = pd.DataFrame(columns=model_columns)
    df_aligned.loc[0] = 0
    for col in df_encoded.columns:
        if col in df_aligned.columns:
            df_aligned[col] = df_encoded[col]

    # prediction = model.predict(df_aligned)[0]
    proba = model.predict_proba(df_aligned)[0][1]  # Probability of class 1 (subscribe)
    prediction = 1 if proba >= 0.5 else 0
    
    return templates.TemplateResponse("result.html", {
      "request": request,
      "prediction": "Yes" if prediction == 1 else "No",
      "probability": f"{proba:.2f}"  # format to 2 decimal places
    })


    # result = "Yes" if prediction == 1 else "No"
    # return templates.TemplateResponse("result.html", {"request": request, "prediction": result})
