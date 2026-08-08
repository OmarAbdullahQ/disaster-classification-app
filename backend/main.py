from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from model_loader import get_prediction

app = FastAPI(title="Disaster Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_FORMATS = {"image/jpeg", "image/jpg", "image/png"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Validate format
    if image.content_type not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image format. Please upload JPG, JPEG, or PNG."
        )
    
    # Read file content for size validation
    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size is 10 MB."
        )
    
    # Generate placeholder response matching the PRD
    result = get_prediction(contents)
    return result
