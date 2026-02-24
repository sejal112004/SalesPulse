# analyzer/management/commands/migrate_existing_datasets.py
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from analyzer.models import Dataset
from django.core.files import File as DjangoFile
import os
from django.conf import settings

class Command(BaseCommand):
    help = 'Migrates existing uploaded files to Dataset model instances'

    def handle(self, *args, **options):
        upload_dir = os.path.join(settings.BASE_DIR, 'analyzer', 'uploads')
        
        if not os.path.exists(upload_dir):
            self.stdout.write(self.style.WARNING('Upload directory does not exist.'))
            return
        
        # Get all files in upload directory
        files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
        
        if not files:
            self.stdout.write(self.style.WARNING('No files found in upload directory.'))
            return
        
        self.stdout.write(f'Found {len(files)} files to migrate...')
        
        migrated = 0
        skipped = 0
        
        for filename in files:
            # Skip if already in database
            if Dataset.objects.filter(name=filename).exists():
                self.stdout.write(self.style.WARNING(f'  Skipping {filename} (already exists)'))
                skipped += 1
                continue
            
            file_path = os.path.join(upload_dir, filename)
            
            # Try to find the user who uploaded it (if we can determine from filename or use first user)
            # For now, we'll assign to the first user or create a generic entry
            user = User.objects.first()
            
            if not user:
                self.stdout.write(self.style.ERROR('No users found. Please create a user first.'))
                return
            
            try:
                with open(file_path, 'rb') as f:
                    dataset = Dataset.objects.create(
                        user=user,
                        name=filename,
                        file=DjangoFile(f, name=filename)
                    )
                self.stdout.write(self.style.SUCCESS(f'  ✓ Migrated: {filename}'))
                migrated += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'  ✗ Error migrating {filename}: {str(e)}'))
        
        self.stdout.write(self.style.SUCCESS(
            f'\n✓ Migration complete!\n'
            f'  Migrated: {migrated}\n'
            f'  Skipped: {skipped}\n'
            f'  Total: {len(files)}'
        ))
