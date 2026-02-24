# analyzer/models.py
from django.db import models
from django.db.models import Max
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    image = models.ImageField(upload_to='profile_pics/', default='profile_pics/default.png', blank=True)
    bio = models.TextField(max_length=500, blank=True)
    location = models.CharField(max_length=100, blank=True)
    
    def __str__(self):
        return f'{self.user.username} Profile'
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        
        # Resize image if it exists and PIL is available
        try:
            from PIL import Image
            if self.image and hasattr(self.image, 'path'):
                img = Image.open(self.image.path)
                if img.height > 300 or img.width > 300:
                    output_size = (300, 300)
                    img.thumbnail(output_size)
                    img.save(self.image.path)
        except (ImportError, IOError, AttributeError):
            # PIL not available or image processing failed, skip resizing
            pass

class Dataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=255, blank=True)  # optional custom name
    # Simple, integer-based versioning per user. Newer uploads get higher versions.
    version = models.PositiveIntegerField(default=1)
    # Marks which dataset is currently active on the dashboard.
    is_current = models.BooleanField(default=False)
    # Stores a simple schema signature (e.g., sorted lowercased column names) for
    # validating that multiple datasets share a compatible structure.
    schema_signature = models.CharField(max_length=1024, blank=True)

    def __str__(self):
        return self.file.name.split('/')[-1]

    def save(self, *args, **kwargs):
        if not self.name:
            self.name = self.file.name.split('/')[-1]
        # Auto-assign version on first save if not set explicitly.
        if not self.pk and not self.version:
            if self.user_id:
                last_version = (
                    Dataset.objects.filter(user_id=self.user_id)
                    .aggregate(max_ver=Max("version"))
                    .get("max_ver")
                ) or 0
                self.version = last_version + 1
            else:
                # Fallback for datasets without a user (should be rare).
                last_version = (
                    Dataset.objects.all().aggregate(max_ver=Max("version")).get("max_ver")
                ) or 0
                self.version = last_version + 1
        super().save(*args, **kwargs)

class SavedChart(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.SET_NULL, null=True, blank=True)
    title = models.CharField(max_length=255)
    chart_type = models.CharField(max_length=50)
    x_axis = models.CharField(max_length=100)
    y_axis = models.CharField(max_length=100)
    aggregation = models.CharField(max_length=20, default='sum')
    limit = models.CharField(max_length=10, default='10') # '10', '20', 'all'
    filter_op = models.CharField(max_length=10, blank=True, null=True)
    filter_value = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} ({self.user.username})"