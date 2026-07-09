import cv2

# ----------------------------------------
# Blur Detection Function
# ----------------------------------------

def is_blurry(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    variance = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return variance < 100