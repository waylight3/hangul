from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
import datetime, random

class UserInfo(models.Model):
	user = models.OneToOneField(User, default=None)
	name = models.CharField(max_length=64, default='이름')
	name_public = models.BooleanField(default=True)
	nickname = models.CharField(max_length=64, default='별명')
	email = models.EmailField(default='default@email.com')
	email_public = models.BooleanField(default=True)
	point = models.IntegerField(default=0)
	comment = models.CharField(max_length=256, default='소개')
	register_datetime = models.DateTimeField(default=datetime.datetime.now)
	last_login_time = models.DateTimeField(default=datetime.datetime.now)
	last_login_ip = models.CharField(max_length=64, default='255.255.255.255')

	def __str__(self):
		return self.user.username