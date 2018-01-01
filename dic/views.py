from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.core.urlresolvers import reverse
from django.views.generic import TemplateView
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from user.models import *
from dic.models import *
from datetime import datetime, date, timezone
from ipware.ip import get_ip
import html, difflib
from django.core.mail import send_mail

def view_index(request, link_dic):
	userinfo = None
	if request.user.is_authenticated():
		userinfo = UserInfo.objects.get(user=request.user)
	dic = get_object_or_404(Dictionary, link=link_dic)
	data = {
		'userinfo':userinfo,
		'dic':dic,
	}
	return render(request, 'dic/index.html', data)

def view_word(request, link_dic, link_word):
	userinfo = None
	if request.user.is_authenticated():
		userinfo = UserInfo.objects.get(user=request.user)
	dic = get_object_or_404(Dictionary, link=link_dic)
	word = get_object_or_404(Word, word=link_word)
	data = {
		'userinfo':userinfo,
		'dic':dic,
		'word':word,
	}
	return render(request, 'dic/word.html', data)