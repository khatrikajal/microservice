#integrete the all ml service routes in main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.rf_routes import router as rf_router  # Temporarily disabled due to pandas compatibility issue
from routes.svm_mlp_routes import router as svm_mlp_router
from routes.cluster_routes import router as cluster_router

app = FastAPI(
    title="ML Services API",
    description="Machine Learning prediction services: Rain Forecasting (SVM/MLP), Car Evaluation (Random Forest), and Social Media Clustering (K-means)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rf_router)  # Temporarily disabled
app.include_router(svm_mlp_router)
app.include_router(cluster_router)      

#RUN ON THE PORT 8005

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)