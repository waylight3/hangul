from django.conf.urls import url
from dic import views

urlpatterns = [
	url(r'^(?P<link_dic>[^\/]+)$', views.view_index),
	url(r'^(?P<link_dic>[^\/]+)/(?P<link_word>.+)$', views.view_word)
]