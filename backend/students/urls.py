from django.urls import path, include

from rest_framework.routers import DefaultRouter

from .views import StudentViewSet, upload_face


router = DefaultRouter()

router.register(r"students", StudentViewSet, basename="students")

urlpatterns = [
    path("students/upload-face/", upload_face),
    path("", include(router.urls)),
]