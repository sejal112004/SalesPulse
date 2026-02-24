# analyzer/management/commands/create_admin.py
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
import getpass

class Command(BaseCommand):
    help = 'Creates a superuser/admin account for the admin panel'

    def add_arguments(self, parser):
        parser.add_argument(
            '--username',
            type=str,
            help='Username for the admin account',
        )
        parser.add_argument(
            '--email',
            type=str,
            help='Email for the admin account',
        )
        parser.add_argument(
            '--password',
            type=str,
            help='Password for the admin account (not recommended, use interactive mode)',
        )

    def handle(self, *args, **options):
        username = options.get('username')
        email = options.get('email')
        password = options.get('password')

        # Interactive mode if username not provided
        if not username:
            username = input('Enter username: ').strip()
            if not username:
                self.stdout.write(self.style.ERROR('Username is required!'))
                return

        # Check if user already exists
        if User.objects.filter(username=username).exists():
            self.stdout.write(self.style.WARNING(f'User "{username}" already exists!'))
            update = input('Do you want to make this user a superuser? (y/n): ').strip().lower()
            if update == 'y':
                user = User.objects.get(username=username)
                user.is_superuser = True
                user.is_staff = True
                user.is_active = True
                user.save()
                self.stdout.write(self.style.SUCCESS(f'User "{username}" is now a superuser!'))
            return

        # Get email
        if not email:
            email = input('Enter email (optional): ').strip()

        # Get password
        if not password:
            password = getpass.getpass('Enter password: ')
            password_confirm = getpass.getpass('Confirm password: ')
            if password != password_confirm:
                self.stdout.write(self.style.ERROR('Passwords do not match!'))
                return
            if len(password) < 8:
                self.stdout.write(self.style.WARNING('Password is too short (minimum 8 characters).'))
                continue_anyway = input('Continue anyway? (y/n): ').strip().lower()
                if continue_anyway != 'y':
                    return

        # Create superuser
        try:
            user = User.objects.create_user(
                username=username,
                email=email if email else '',
                password=password
            )
            user.is_superuser = True
            user.is_staff = True
            user.is_active = True
            user.save()

            self.stdout.write(self.style.SUCCESS(
                f'\n✓ Successfully created admin user: "{username}"\n'
                f'  Email: {email if email else "Not provided"}\n'
                f'  You can now login at: /admin-panel/login/\n'
            ))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error creating user: {str(e)}'))
