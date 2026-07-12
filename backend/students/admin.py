from django.contrib import admin

from .models import Student

# Register your models here.
@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):

    list_display = (
        "student_id",
        "first_name",
        "last_name",
        "department",
        "email",
    )

    search_fields = (
        "student_id",
        "first_name",
        "last_name",
        "email",
    )

    ordering = (
        "student_id",
    )