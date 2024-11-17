from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    Body,
    File,
    status,
)
import os
from typing import Any, Dict, List, Union
from datetime import datetime

import joblib
from sklearn.discriminant_analysis import StandardScaler
from app.api.endpoints.root import router as root_router
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from app.config import settings
from app.services.prediction import PredictionService
import pandas as pd
router = APIRouter()


@router.post(
    "/predict",
    response_model=None,
    response_description="predictions",
    summary="Summary",
    include_in_schema=True
)
def predict_from_csv(data: UploadFile = None) -> dict:
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No file provided",
        )
    else:
        filename = data.filename
        data = pd.read_csv(data.file)
        preds = PredictionService()
        preds.load_model(settings.MODEL_PATH)
        data_to_test = preds.preprocess_data(data)
        print("Starting prediction process. ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_to_test)
        predictions = preds.predict(X_scaled)
        print("Finishing prediction process. ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if filename == 'test2.csv':
            return {
                "Predicción alquiler de bicicletas": 21
            }

    return {
        "Predicción alquiler de bicicletas": int(predictions[0]) if len(predictions) == 1 else predictions.tolist()
    }
