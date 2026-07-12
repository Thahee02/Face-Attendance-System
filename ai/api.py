from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import pickle
import cv2
import numpy as np

from insightface.app import FaceAnalysis

from ai.utils.cosine_similarity_check import cosine_similarity

# ----------------------------------------
# Configuration
# ----------------------------------------

EMBEDDING_FILE = "embeddings/embeddings.pkl"

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