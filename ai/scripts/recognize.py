import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

from insightface.app import FaceAnalysis

SCRIPT_DIR = Path(__file__).resolve().parent
AI_ROOT = SCRIPT_DIR.parent

if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from utils.cosine_similarity_check import cosine_similarity


# -----------------------------------------
# Configuration
# -----------------------------------------

EMBEDDING_FILE = "../embeddings/embeddings.pkl"
THRESHOLD = 0.65


# -----------------------------------------
# Load Embeddings
# -----------------------------------------

with open(EMBEDDING_FILE,"rb") as file:

    embedding_database = pickle.load(file)

print(f"Loaded {len(embedding_database)} embeddings.")


# -----------------------------------------
# Load AI Model
# -----------------------------------------

app = FaceAnalysis(name="buffalo_l")

app.prepare(
    ctx_id=-1,
    det_size=(640,640)
)


# -----------------------------------------
# Webcam
# -----------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Cannot open webcam.")
    exit()

print("Recognition started.")


# -----------------------------------------
# Main Loop
# -----------------------------------------

while True:

    success, frame = camera.read()

    if not success:
        break

    faces = app.get(frame)

    for face in faces:

        embedding = face.embedding

        best_score = -1

        best_student = "Unknown"

        # Compare with database
        for record in embedding_database:

            score = cosine_similarity(
                embedding,
                record["embedding"]
            )

            if score > best_score:

                best_score = score

                best_student = record["student_id"]

        # Check threshold
        if best_score < THRESHOLD:
            best_student = "Unknown"

        # Face box
        x1, y1, x2, y2 = map(
            int,
            face.bbox
        )

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"{best_student} ({best_score:.2f})",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,0),
            2
        )

    cv2.imshow(
        "Face Recognition",
        frame
    )

    if cv2.waitKey(1) & (0xFF == ord("q") or 0xFF == ord("Q")):
        break


camera.release()
cv2.destroyAllWindows()
