import os
import pickle
import cv2
from insightface.app import FaceAnalysis

# -----------------------------------------
# Configuration
# -----------------------------------------

DATASET_PATH = "../dataset"
OUTPUT_PATH = "../embeddings/embeddings.pkl"


# -----------------------------------------
# Initialize InsightFace
# -----------------------------------------

print("Loading InsightFace model...")

app = FaceAnalysis(name="buffalo_l")

# Use -1 for CPU.
# Change to 0 if you have a CUDA-compatible GPU configured.
app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("Model loaded successfully.")


# -----------------------------------------
# Create Output Folder
# -----------------------------------------

os.makedirs(
    "../embeddings",
    exist_ok=True
)


# -----------------------------------------
# Store Embeddings
# -----------------------------------------

embedding_database = []


# -----------------------------------------
# Read Student Folders
# -----------------------------------------

for student_id in os.listdir(DATASET_PATH):

    student_folder = os.path.join(
        DATASET_PATH,
        student_id
    )

    if not os.path.isdir(student_folder):
        continue

    print(f"\nProcessing {student_id}")

    # -----------------------------
    # Read Images
    # -----------------------------

    for image_name in os.listdir(student_folder):

        image_path = os.path.join(
            student_folder,
            image_name
        )

        image = cv2.imread(image_path)

        print(f"Processing {image_name}")

        if image is None:

            print(f"Cannot read {image_name}")
            continue

        # -----------------------------
        # Detect Face
        # -----------------------------

        faces = app.get(image)

        print(f"Detected {len(faces)} faces in {image_name}")
        
        if len(faces) != 1:

            print(f"Skipped {image_name}")
            continue

        face = faces[0]

        embedding_database.append({
            "student_id": student_id,
            "image": image_name,
            "embedding": face.embedding
        })

        print(f"Added {image_name}")


# -----------------------------------------
# Save Embeddings
# -----------------------------------------

with open(OUTPUT_PATH,"wb") as file:

    pickle.dump(
        embedding_database,
        file
    )

print("\n===================================")
print("Embedding generation completed.")
print(f"Total embeddings : {len(embedding_database)}")
print(f"Saved to : {OUTPUT_PATH}")
print("===================================")