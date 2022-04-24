from django.http import HttpResponse 
from django.shortcuts import render, redirect 
from upload.models import UploadedImage
from django.urls import reverse_lazy
from django.contrib.auth.forms import UserCreationForm
from django.views.generic.edit import CreateView
# from upload.models import MyModel
from .forms import *
from .enhance import enhance
# from .enhance import enhance_engan

# Create your views here.
def index(request):
    alert_message = False
    if request.method == 'POST': 
        submitted_form = UploadImageForm(request.POST, request.FILES)

        if submitted_form.is_valid():
            submitted_form.save()
            alert_message = {
                'status': True,
                'message': 'Successfully saved the image'
            }
        else:
            alert_message = {
                'status': False,
                'message': 'Form data is invalid. Please check if your image / title is repeated'
            }
    
    if request.method == 'POST' and 'run_script' in request.POST:
        
        enhance() 
        # enhance_engan()


    
    form = UploadImageForm()
    context = {
        'alert_data': alert_message,
        'form': form,
        'images': UploadedImage.objects.all
    }
    return render(request, 'upload/index.html', context=context)

def home(request):
    return render(request,"upload/home.html")

class SignUp(CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy("login")
    template_name = "registration/signup.html"
