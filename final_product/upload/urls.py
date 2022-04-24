from django.urls import path
from django.conf.urls import include
from upload import views

urlpatterns = [
    
    path('', views.home, name = "home"),
    path("signup/", views.SignUp.as_view(), name="signup"),
    path('index/', views.index, name='index'),
]