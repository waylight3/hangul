from django.conf.urls import url
from ai import views

urlpatterns = [
	url(r'^$', views.view_index),
]