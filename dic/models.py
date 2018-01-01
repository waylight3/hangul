from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
import datetime, random

class Category(models.Model):
	name = models.CharField(max_length=64, default='분류')  # category name in korean
	link = models.CharField(max_length=64, default='분류')  # link for category

	def __str__(self):
		return self.name

class Dictionary(models.Model):
	name = models.CharField(max_length=16, default='언어')  # name in korean
	iso = models.CharField(max_length=16, default='aaa')    # iso 639 code
	link = models.CharField(max_length=16, default='aaa')   # 한글.com/{{link}}
	admin = models.ForeignKey(User, blank=True, null=True, default=None, related_name='admin_dics') # admin of the dictionary

	def __str__(self):
		return self.name

class Word(models.Model):
	dictionary = models.ForeignKey(Dictionary, blank=True, null=True, default=None, related_name='dictionary_words')
	word = models.CharField(max_length=128, default='단어') # ex. 한글
	pos = models.CharField(max_length=128, default='품사')  # ex. 명사

	def __str__(self):
		return self.word

class Meaning(models.Model):
	meaning = models.TextField(blank=True, null=True) # ex. 한국어를 표기하는 대한민국의 고유문자.
	word = models.ForeignKey(Word, blank=True, null=True, default=None, related_name='word_meanings')        # ex. 한글
	category = models.ManyToManyField(Category, blank=True, default=None, related_name='category_meanings')  # ex. <국문학>
	writer = models.ForeignKey(User, blank=True, null=True, default=None, related_name='writer_meanings')    # ex. waylight3
	
	def __str__(self):
		return self.meaning

class Example(models.Model):
	example = models.TextField(blank=True, null=True, default='예문') # ex. 한글은 자음 19자, 모음 21자, 받침 27자로 구성된다.
	meaning = models.ForeignKey(Meaning, blank=True, null=True, default=None, related_name='meaning_examples')  # ex. 한국어를 표기하는 대한민국의 고유문자.
	writer = models.ForeignKey(User, blank=True, null=True, default=None, related_name='writer_examples')       # ex. waylight3

	def __str__(self):
		return self.example