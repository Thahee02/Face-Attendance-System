import cv2
import os

# ----------------------------------------
# Configuration
# ----------------------------------------

DATASET_PATH = "../dataset"

IMAGE_LIMIT = 30

# ----------------------------------------
# Get Student ID
# ----------------------------------------

student_id = input(
    "Enter Student ID: "
)

student_folder = os.path.join(
    DATASET_PATH,
    student_id
)

# Create folder if not exists
os.makedirs(
    student_folder,
    exist_ok=True
)

print(
    f"Saving images to {student_folder}"
)

# ----------------------------------------
# Load Face Detector
# ----------------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Open Webcam
camera = cv2.VideoCapture(0)

# Check whether webcam opened successfully
if not camera.isOpened():

    print("Error : Cannot access webcam.")
    exit()

# Image counter
count = 0

print("Starting face collection...")

# ----------------------------------------
# Start Infinite Loop
# ----------------------------------------

while True:

    # Read one frame
    success, frame = camera.read()

    # If frame cannot be read

    if not success:

        print("Cannot receive frame.")
        break

    # ------------------------------------
    # Convert to Grayscale
    # ------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # ------------------------------------
    # Detect Faces
    # ------------------------------------

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # ------------------------------------
    # Draw Rectangle
    # ------------------------------------

    for (x, y, w, h) in faces:

        # Crop face
        face = frame[
            y:y+h,
            x:x+w
        ]

        # Resize face
        face = cv2.resize(
            face,
            (160,160)
        )

        # Save image
        count += 1

        image_path = os.path.join(
            student_folder,
            f"{count}.jpg"
        )

        cv2.imwrite(
            image_path,
            face
        )

        print(
            f"Saved {count}/{IMAGE_LIMIT}"
        )

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


    # ------------------------------------
    # Show Face Count
    # ------------------------------------

    cv2.putText(
        frame,
        f"Faces : {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Show frame
    cv2.imshow(
        "Student Face Collection - Press 'q' to Quit",
        frame
    )

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------------------
# Release Resources
# ----------------------------------------

camera.release()

cv2.destroyAllWindows()

print("Program Closed.")