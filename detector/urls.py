"""
URL configuration for detector app
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('model-info/', views.model_info, name='model_info'),
]
