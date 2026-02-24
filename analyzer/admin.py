from django.contrib import admin

# Register your models here.
# analyzer/admin.py
from django.contrib import admin
from .models import Dataset, UserProfile

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'uploaded_at', 'file')
    list_filter = ('uploaded_at', 'user')
    search_fields = ('name',)

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'location', 'has_image', 'bio_preview')
    list_filter = ('location',)
    search_fields = ('user__username', 'user__email', 'bio', 'location')
    readonly_fields = ('user',)
    
    def has_image(self, obj):
        """Check if profile has an image"""
        return bool(obj.image)
    has_image.boolean = True
    has_image.short_description = 'Has Image'
    
    def bio_preview(self, obj):
        """Show first 50 characters of bio"""
        return obj.bio[:50] + '...' if len(obj.bio) > 50 else obj.bio
    bio_preview.short_description = 'Bio Preview'