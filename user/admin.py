from django.contrib import admin
from user.models import *

class UserInfoAdmin(admin.ModelAdmin):
	list_display = ('user', 'name', 'nickname', 'email', 'point', 'register_datetime')

admin.site.register(UserInfo, UserInfoAdmin)