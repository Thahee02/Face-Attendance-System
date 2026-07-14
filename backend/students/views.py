import tempfile

from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Student
from .serializers import StudentSerializer
from .services import register_face


class StudentViewSet(viewsets.ModelViewSet):

    queryset = Student.objects.all().order_by("student_id")

    serializer_class = StudentSerializer



@api_view(["POST"])
def upload_face(request):

    """
    Upload face image and forward it to the AI service.
    """

    student_id = request.data.get("student_id")

    image = request.FILES.get("image")

    if not student_id or not image:

        return Response(
            {
                "status": "failed",
                "message": "Student ID and image are required."
            },
            status=400
        )

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:

        for chunk in image.chunks():

            temp.write(chunk)

        temp_path = temp.name

    result = register_face(
        student_id,
        temp_path
    )

    return Response(result)
