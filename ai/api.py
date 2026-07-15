from pathlib import Path

from django.http import JsonResponse
from fastapi import Form
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import pickle
import cv2
import numpy as np
from datetime import datetime
import os

from insightface.app import FaceAnalysis

from ai.utils.embeddings_crud_utils import load_embeddings, save_embeddings
from utils.cosine_similarity_check import cosine_similarity


# ----------------------------------------
# Configuration
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_FILE = BASE_DIR / "embeddings" / "embeddings.pkl"

THRESHOLD = 0.65

# -----------------------------------------
# Create FastAPI application
# -----------------------------------------

app = FastAPI(
    title="Face Recognition AI Service",
    version="1.0.0",
    description="AI service for face recognition attendance system."
)


# ----------------------------------------
# Load Embeddings
# ----------------------------------------

with open(EMBEDDING_FILE,"rb") as file:

    embedding_database = pickle.load(file)

print(f"Loaded {len(embedding_database)} embeddings.")


# ----------------------------------------
# Load InsightFace
# ----------------------------------------

face_model = FaceAnalysis(
    name="buffalo_l"
)

face_model.prepare(
    ctx_id=-1,
    det_size=(160,160)
)


# -----------------------------------------
# Home Route
# -----------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Service is running.",
        "status": "success"
    }


# ----------------------------------------
# Recognition API
# ----------------------------------------

@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await image.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return JSONResponse(
            status_code=400,
            content={
                "message":"Invalid image."
            }
        )

    # Detect face
    faces = face_model.get(frame)

    if len(faces) != 1:

        return JSONResponse(
            status_code=400,
            content={
                "status":"failed",
                "message":"Exactly one face required."
            }
        )

    embedding = faces[0].embedding

    best_score = -1

    best_student = "Unknown"

    # Compare
    for record in embedding_database:

        score = cosine_similarity(
            embedding,
            record["embedding"]
        )

        if score > best_score:

            best_score = score
            best_student = record["student_id"]

    if best_score < THRESHOLD:
        best_student = "Unknown"

    return JSONResponse(
        status_code=200,
        content={
            "status":"success",
            "student_id":best_student,
            "confidence":round(float(best_score),4)
        }
    )

# ----------------------------------------
# Register API
# ----------------------------------------

@app.post("/register")
async def register(student_id: str = Form(...), image: UploadFile = File(...)):

    image_bytes = await image.read()

    image_array = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:

        return JsonResponse(
            status_code=400,
            content={
                "status":"failed",
                "message":"Invalid image."
            }
        )

    faces = face_model.get(frame)

    if len(faces) != 1:

        return JsonResponse(
            status_code=400,
            content={
                "status":"failed",
                "message":"Exactly one face required."
            }
        )   

    embedding = faces[0].embedding

    database = load_embeddings()

    if student_id not in database:
        database[student_id] = []

    database[student_id].append({
        "embedding": embedding,
        "model": "buffalo_l",
        "created_at": datetime.utcnow().isoformat()
    })

    save_embeddings(database)

    return JsonResponse(
        status_code=200,
        content={
            "status":"success",
            "student_id": student_id,
            "message":"Embedding saved successfully."
        }
    )