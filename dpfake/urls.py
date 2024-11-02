from django.contrib import admin
from django.urls import path,include
from django.conf import settings
from django.contrib.staticfiles.urls import static,staticfiles_urlpatterns
from dpfake import views
app_name='dpfake'
urlpatterns = [   
    path('',views.upload_and_predict,name='upload_and_predict'),
]
urlpatterns+=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
urlpatterns+=static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)
