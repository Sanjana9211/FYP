from django.urls import path
from . import views

urlpatterns = [
    path('', views.index,name='home'),
    path('about', views.about,name='about'),
    path('prediction', views.prediction,name='prediction'),
    path('yprediction', views.yprediction,name='yprediction'),
    path('frecommend', views.frecommend,name='frecommend'),
    path('crecommend', views.crecommend,name='crecommend'),


    path('schemes', views.schemes,name='schemes'),
    #path('latestnews', views.latestnews,name='latestnews'),
    path('latestnews', views.latestnews,name='latestnews'),

    path('livefeedpage', views.livefeedpage,name='livefeedpage'),
    path('community', views.community,name='community'),
    path('contact', views.contact,name='contact'),
    path('calculate', views.index1, name="calculate"),
    path('yieldcalc', views.yieldcalc, name="yieldcalc"),
    path('fertrec', views.fertrec, name="fertrec"),
    path('croprec', views.croprec, name="croprec"),



]

