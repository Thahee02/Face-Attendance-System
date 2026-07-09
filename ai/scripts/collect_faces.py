import os
import sys
import time
from pathlib import Path
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
AI_ROOT = SCRIPT_DIR.parent

if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from utils.image_quality_check_utils import is_blurry


# ----------------------------------------
# Configuration
# ----------------------------------------

DATASET_PATH = "../dataset"

IMAGE_LIMIT = 30

CAPTURE_DELAY = 1


# ----------------------------------------
# Student ID
# ----------------------------------------

student_id = input(
    "Enter Student ID: "
)

student_folder = os.path.join(
    DATASET_PATH,
    student_id
)

os.makedirs(
    student_folder,
    exist_ok=True
)


# ----------------------------------------
# Face Detector
# ----------------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ----------------------------------------
# Webcam
# ----------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Cannot open webcam")
    exit()

count = 0
last_capture_time = 0

print("Starting collection...")


# ----------------------------------------
# Main Loop
# ----------------------------------------

while True:

    success, frame = camera.read()

    if not success:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100,100)
    )

    # Only one face allowed

    if len(faces) == 1:

        x,y,w,h = faces[0]

        face = frame[
            y:y+h,
            x:x+w
        ]

        face = cv2.resize(
            face,
            (160,160)
        )

        current_time = time.time()

        # Check delay
        if (current_time - last_capture_time>= CAPTURE_DELAY):

            # Check blur
            if not is_blurry(face):

                count += 1

                image_path = os.path.join(
                    student_folder,
                    f"{count}.jpg"
                )

                cv2.imwrite(
                    image_path,
                    face
                )

                last_capture_time = current_time

                print(f"Saved {count}/{IMAGE_LIMIT}")


        # Draw rectangle
        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Only one person allowed",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )


    # Progress
    cv2.putText(
        frame,
        f"Images: {count}/{IMAGE_LIMIT}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )


    cv2.imshow(
        "Face Dataset Collector",
        frame
    )

    # Finish
    if count >= IMAGE_LIMIT:
        print("Dataset completed.")
        break

    # Exit on 'q' key or 'Q' key
    if cv2.waitKey(1) & (0xFF == ord("q") or 0xFF == ord("Q")):
        break


camera.release()
cv2.destroyAllWindows()

print("Camera closed.")