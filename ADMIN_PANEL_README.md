# Admin Panel Setup Guide

## Overview
A custom admin panel has been created for SalesPulse to manage users, datasets, and system settings.

## Access the Admin Panel
- **URL**: `http://localhost:8000/admin-panel/login/`
- **Django Admin**: `http://localhost:8000/admin/` (default Django admin)

## Creating Admin Credentials

### Method 1: Using Management Command (Recommended)
Run this command in your terminal:

```bash
python manage.py create_admin
```

This will prompt you to enter:
- Username
- Email (optional)
- Password (will be hidden)
- Password confirmation

### Method 2: Using Django Shell
```bash
python manage.py shell
```

Then in the shell:
```python
from django.contrib.auth.models import User
user = User.objects.create_user('admin', 'admin@example.com', 'your_password')
user.is_superuser = True
user.is_staff = True
user.is_active = True
user.save()
```

### Method 3: Using Django's createsuperuser
```bash
python manage.py createsuperuser
```

## Admin Panel Features

### 1. Dashboard
- View total users, active users, datasets
- See recent activity and statistics
- Quick access to all admin functions

### 2. User Management
- View all users
- Search and filter users
- View user details
- Activate/Deactivate users
- Delete users (except superusers)
- View user datasets

### 3. Dataset Management
- View all uploaded datasets
- Search datasets
- View dataset details
- Delete datasets

### 4. System Settings
- View system information
- Check storage usage
- View debug mode status

## Security Notes

1. **Only superusers can access the admin panel**
2. Make sure to set `DEBUG=False` in production
3. Configure `ALLOWED_HOSTS` in `settings.py` for production
4. Use strong passwords for admin accounts
5. Regularly review user permissions

## File Structure

```
analyzer/
├── admin_panel/
│   ├── __init__.py
│   ├── views.py          # Admin views
│   ├── urls.py           # Admin URLs
│   ├── templates/
│   │   └── admin/
│   │       ├── base.html
│   │       ├── login.html
│   │       ├── dashboard.html
│   │       ├── manage_users.html
│   │       ├── user_detail.html
│   │       ├── manage_datasets.html
│   │       └── system_settings.html
│   └── static/
│       └── css/
│           └── admin.css
└── management/
    └── commands/
        └── create_admin.py
```

## URLs

- `/admin-panel/login/` - Admin login
- `/admin-panel/dashboard/` - Admin dashboard
- `/admin-panel/users/` - Manage users
- `/admin-panel/datasets/` - Manage datasets
- `/admin-panel/settings/` - System settings

## Troubleshooting

### Can't access admin panel?
1. Make sure you created a superuser account
2. Check that `is_superuser=True` and `is_staff=True`
3. Verify you're logged in with the correct account

### Static files not loading?
Run:
```bash
python manage.py collectstatic
```

### Templates not found?
Check that `analyzer/admin_panel/templates` is in `TEMPLATES['DIRS']` in `settings.py`

## Support

For issues or questions, check the Django documentation or project README.
