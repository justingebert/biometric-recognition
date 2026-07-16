import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model_utils import (
    TEST_PROBES_DIR,
    list_persons,
    list_test_images,
    run_verify,
    run_verify_test,
)

app = FastAPI(title="Face Verify")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve staged probe images so the frontend can render thumbnails.
os.makedirs(TEST_PROBES_DIR, exist_ok=True)
app.mount("/test-images-static", StaticFiles(directory=TEST_PROBES_DIR), name="test-images-static")


@app.get("/persons")
def persons():
    return {"persons": list_persons()}


@app.get("/test-images")
def test_images():
    return {"images": list_test_images()}


@app.post("/verify")
async def verify(
    person: str = Form(...),
    file: UploadFile = File(...),
    detection_threshold: float | None = Form(None),
    verification_threshold: float | None = Form(None),
):
    image_bytes = await file.read()
    try:
        return run_verify(person, image_bytes, detection_threshold, verification_threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class VerifyTestRequest(BaseModel):
    person: str
    name: str
    detection_threshold: float | None = None
    verification_threshold: float | None = None


@app.post("/verify-test")
def verify_test(req: VerifyTestRequest):
    try:
        return run_verify_test(
            req.person,
            req.name,
            req.detection_threshold,
            req.verification_threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
