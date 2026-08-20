from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prediction import PredictionPipeline
import uvicorn
import os

app = FastAPI(
    title="RenalScan AI - Inference Service",
    description="Isolated API wrapper around the VGG16 Kidney Classification model",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    classifier = PredictionPipeline()
except Exception as e:
    print(f"Warning: Failed to instantiate PredictionPipeline: {e}")
    classifier = None

@app.get("/")
def read_root():
    return {"status": "online", "model": "VGG16 Transfer Learning"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Prediction model pipeline initialization failed.")
    
    # Verify file extension
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if file.filename and ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file format. Only JPG, JPEG, and PNG are allowed.")

    try:
        # Read file bytes in memory
        content = await file.read()
        prediction, confidence = classifier.predict(content)
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4)
        }
    except Exception as err:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {err}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
