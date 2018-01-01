from django.conf.urls import include, url
from django.conf.urls.static import static
from django.conf import settings
from django.contrib import admin
from user import views as user_views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$', user_views.index),
    url(r'^사전/', include('dic.urls', namespace='dic'))
]