from django.contrib import admin
from dic.models import *

class CategoryAdmin(admin.ModelAdmin):
	list_display = ('name', 'link')

class DictionaryAdmin(admin.ModelAdmin):
	list_display = ('name', 'iso', 'link', 'admin')

class WordAdmin(admin.ModelAdmin):
	list_display = ('word', 'pos')

class MeaningAdmin(admin.ModelAdmin):
	list_display = ('meaning', 'word', 'writer')
	filter_horizontal = ('category', )

class ExampleAdmin(admin.ModelAdmin):
	list_display = ('example', 'meaning', 'writer')

admin.site.register(Category, CategoryAdmin)
admin.site.register(Dictionary, DictionaryAdmin)
admin.site.register(Word, WordAdmin)
admin.site.register(Meaning, MeaningAdmin)
admin.site.register(Example, ExampleAdmin)