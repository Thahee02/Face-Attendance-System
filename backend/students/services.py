import requests

AI_SERVICE_URL = "http://127.0.0.1:8001"


def register_face(student_id, image_path):
    """
    Send a face image to the AI service.
    """

    with open(image_path, "rb") as image:

        files = {
            "image": image
        }

        data = {
            "student_id": student_id
        }

        response = requests.post(
            f"{AI_SERVICE_URL}/register",
            files=files,
            data=data,
            timeout=60
        )

    return response.json()