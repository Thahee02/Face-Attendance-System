import cv2

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

print("Webcam started successfully.")

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