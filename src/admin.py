from django.contrib import admin
from .models import Producer, UserProfile, Review

admin.site.register(Producer)
admin.site.register(UserProfile)
admin.site.register(Review)