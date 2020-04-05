from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('hasil/', views.hasil, name='hasil'),
    path('hasil/hasil', views.hasil, name='hasil')
]