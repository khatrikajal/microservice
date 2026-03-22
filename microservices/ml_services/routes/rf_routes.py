from fastapi import APIRouter, HTTPException
from schemas.rf_schemas import RFInput
from services.rf_service import predict_rf

router = APIRouter(prefix="/rf", tags=["Random Forest"])

# Car acceptability labels and descriptions
ACCEPTABILITY_LABELS = {
    "unacc": {
        "label": "Unacceptable",
        "description": "This car is unacceptable based on the given attributes. It may have poor safety, high costs, or inadequate capacity.",
        "recommendation": "Consider looking for a car with better safety ratings, lower costs, or more passenger/luggage capacity."
    },
    "acc": {
        "label": "Acceptable",
        "description": "This car is acceptable. It meets basic requirements but may not be ideal in all aspects.",
        "recommendation": "This car is a reasonable choice if budget is a concern, but there may be better options available."
    },
    "good": {
        "label": "Good",
        "description": "This car is a good choice! It has favorable attributes across most categories.",
        "recommendation": "This car offers good value and should meet your needs well."
    },
    "vgood": {
        "label": "Very Good",
        "description": "This car is an excellent choice! It has outstanding attributes with low costs and high safety/capacity.",
        "recommendation": "Highly recommended! This car offers excellent value and quality."
    }
}

@router.post("/predict")
def predict(data: RFInput):
    """
    Predict car acceptability using Random Forest classifier.

    Evaluates a car based on:
    - Buying price
    - Maintenance cost
    - Number of doors
    - Person capacity
    - Luggage boot size
    - Safety rating

    Returns acceptability classification with detailed explanation.
    """
    try:
        prediction = predict_rf(data.dict())

        # Get label information
        label_info = ACCEPTABILITY_LABELS.get(prediction, {
            "label": "Unknown",
            "description": "Unable to determine car acceptability",
            "recommendation": "Please verify the input parameters"
        })

        return {
            "prediction": prediction,
            "prediction_label": f"{label_info['label']}",
            "description": label_info['description'],
            "recommendation": label_info['recommendation'],
            "car_attributes": {
                "buying_price": data.buying,
                "maintenance_cost": data.maint,
                "doors": data.doors,
                "person_capacity": data.persons,
                "luggage_boot": data.lug_boot,
                "safety": data.safety
            }
        }
    except Exception as e:
        import traceback
        print("="*80)
        print("ERROR in Random Forest prediction:")
        print(traceback.format_exc())
        print("="*80)
        raise HTTPException(status_code=400, detail=str(e))