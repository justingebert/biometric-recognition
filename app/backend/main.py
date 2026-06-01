from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from model_utils import run_verify

app = FastAPI(title="Face Verify")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return run_verify(image_bytes)
