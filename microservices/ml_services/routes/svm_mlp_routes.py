from enum import Enum
import traceback
from fastapi import APIRouter, HTTPException, Query
from schemas.svm_mlp_schemas import SVM_MLP_InputData
from services.svm_mlp_service import predict_svm_mlp


class ModelType(str, Enum):
    """Available model types for rain prediction"""
    svm = "svm"
    mlp = "mlp"


router = APIRouter(prefix="/svm_mlp", tags=["SVM_MLP"])


@router.post("/predict")
def predict(
    data: SVM_MLP_InputData,
    model: ModelType = Query(
        default=ModelType.svm,
        description="Select the model to use for prediction: SVM (Support Vector Machine) or MLP (Multilayer Perceptron)"
    )
):
    """
    Predict whether it will rain tomorrow using either SVM or MLP model.

    - **SVM**: Support Vector Machine - linear classifier with good generalization
    - **MLP**: Multilayer Perceptron - neural network with hidden layers

    Both models are trained on Australian weather data to predict rain tomorrow.
    """
    try:
        # Convert Pydantic model to dict for prediction
        data_dict = data.model_dump()

        # Make prediction using the selected model
        prediction = predict_svm_mlp(data_dict, model_name=model.value)

        # Convert prediction to Yes/No
        result = 'Yes' if prediction == 1 else 'No'

        return {
            "prediction": result,
            "model_used": model.value,
            "prediction_value": prediction
        }
    except Exception as e:
        print("="*80)
        print("ERROR in SVM/MLP prediction:")
        print(traceback.format_exc())
        print("="*80)
        raise HTTPException(status_code=400, detail=str(e))
    
    