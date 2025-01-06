from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("emotion/", views.emotion, name="emotion"),  # SER sayfası için
    path("contact/", views.contact, name="contact"),
    path("about/", views.about, name="about"),
    path("emotion_process/", views.emotion_analysis, name="emotion_process"),
    path("test-endpoint/", views.test_endpoint, name="test_endpoint"),
    path("audio-process/", views.upload_audio, name="audio_process"),
]
