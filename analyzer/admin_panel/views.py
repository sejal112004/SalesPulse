# analyzer/admin_panel/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.contrib.auth.models import User
from django.db.models import Count, Q
from django.utils import timezone
from datetime import timedelta
from analyzer.models import Dataset
import os
from django.conf import settings

def is_admin(user):
    """Check if user is admin/superuser"""
    return user.is_authenticated and user.is_superuser

@user_passes_test(is_admin, login_url='/admin-panel/login/')
def admin_dashboard(request):
    """Admin Dashboard"""
    # Statistics
    total_users = User.objects.count()
    active_users = User.objects.filter(is_active=True).count()
    total_datasets = Dataset.objects.count()
    
    # Recent users (last 7 days)
    week_ago = timezone.now() - timedelta(days=7)
    recent_users = User.objects.filter(date_joined__gte=week_ago).count()
    
    # Recent datasets (last 7 days)
    recent_datasets = Dataset.objects.filter(uploaded_at__gte=week_ago).count()
    
    # Users by registration date (last 30 days)
    month_ago = timezone.now() - timedelta(days=30)
    users_this_month = User.objects.filter(date_joined__gte=month_ago).count()
    
    # Top users by dataset count
    top_users = User.objects.annotate(
        dataset_count=Count('dataset')
    ).order_by('-dataset_count')[:5]
    
    context = {
        'total_users': total_users,
        'active_users': active_users,
        'total_datasets': total_datasets,
        'recent_users': recent_users,
        'recent_datasets': recent_datasets,
        'users_this_month': users_this_month,
        'top_users': top_users,
    }
    
    return render(request, 'admin/dashboard.html', context)

def admin_login(request):
    """Admin Login Page"""
    if request.user.is_authenticated and request.user.is_superuser:
        return redirect('admin_panel:dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None and user.is_superuser:
            login(request, user)
            messages.success(request, f"Welcome, {user.username}!")
            return redirect('admin_panel:dashboard')
        else:
            messages.error(request, "Invalid credentials or you don't have admin access.")
    
    return render(request, 'admin/login.html')

@user_passes_test(is_admin, login_url='/admin-panel/login/')
def admin_logout(request):
    """Admin Logout"""
    logout(request)
    messages.success(request, "You have been logged out successfully!")
    return redirect('admin_panel:login')

@user_passes_test(is_admin, login_url='/admin-panel/login/')
def manage_users(request):
    """Manage Users"""
    users = User.objects.all().order_by('-date_joined')
    
    # Search functionality
    search_query = request.GET.get('search', '')
    if search_query:
        users = users.filter(
            Q(username__icontains=search_query) |
            Q(email__icontains=search_query) |
            Q(first_name__icontains=search_query) |
            Q(last_name__icontains=search_query)
        )
    
    # Filter by status
    status_filter = request.GET.get('status', '')
    if status_filter == 'active':
        users = users.filter(is_active=True)
    elif status_filter == 'inactive':
        users = users.filter(is_active=False)
    elif status_filter == 'staff':
        users = users.filter(is_staff=True)
    
    context = {
        'users': users,
        'search_query': search_query,
        'status_filter': status_filter,
    }
    
    return render(request, 'admin/manage_users.html', context)

@user_passes_test(is_admin, login_url='/admin-panel/login/')
def user_detail(request, user_id):
    """User Detail View"""
    try:
        user = User.objects.get(id=user_id)
        datasets = Dataset.objects.filter(user=user).order_by('-uploaded_at')
        
        context = {
            'user_obj': user,
            'datasets': datasets,
        }
        return render(request, 'admin/user_detail.html', context)
    except User.DoesNotExist:
        messages.error(request, "User not found.")
        return redirect('admin_panel:manage_users')

@user_passes_test(is_admin, login_url='/admin-panel/login/')
def toggle_user_status(request, user_id):
    """Toggle user active status"""
    try:
        user = User.objects.get(id=user_id)
        user.is_active = not user.is_active
        user.save()
        
        status = "activated" if user.is_active else "deactivated"
        messages.success(request, f"User {user.username} has been {status}.")
    except User.DoesNotExist:
        messages.error(request, "User not found.")
    
    return redirect('admin_panel:manage_users')

@user_passes_test(is_admin, login_url='/admin-panel/login/')
def delete_user(request, user_id):
    """Delete User"""
    try:
        user = User.objects.get(id=user_id)
        username = user.username
        user.delete()
        messages.success(request, f"User {username} has been deleted.")
    except User.DoesNotExist:
        messages.error(request, "User not found.")
    
    return redirect('admin_panel:manage_users')

@user_passes_test(is_admin, login_url='/admin-panel/login/')
def manage_datasets(request):
    """Manage Datasets"""
    datasets = Dataset.objects.all().order_by('-uploaded_at')
    
    # Search functionality
    search_query = request.GET.get('search', '')
    if search_query:
        datasets = datasets.filter(
            Q(name__icontains=search_query) |
            Q(user__username__icontains=search_query)
        )
    
    context = {
        'datasets': datasets,
        'search_query': search_query,
    }
    
    return render(request, 'admin/manage_datasets.html', context)

@user_passes_test(is_admin, login_url='/admin-panel/login/')
def delete_dataset(request, dataset_id):
    """Delete Dataset"""
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        dataset_name = dataset.name
        
        # Delete the file if it exists
        if dataset.file:
            file_path = os.path.join(settings.MEDIA_ROOT, dataset.file.name)
            if os.path.exists(file_path):
                os.remove(file_path)
        
        dataset.delete()
        messages.success(request, f"Dataset {dataset_name} has been deleted.")
    except Dataset.DoesNotExist:
        messages.error(request, "Dataset not found.")
    
    return redirect('admin_panel:manage_datasets')

@user_passes_test(is_admin, login_url='/admin-panel/login/')
def system_settings(request):
    """System Settings"""
    # Get system information
    total_files = 0
    total_size = 0
    
    upload_dir = os.path.join(settings.MEDIA_ROOT)
    if os.path.exists(upload_dir):
        for root, dirs, files in os.walk(upload_dir):
            total_files += len(files)
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    
    # Convert to MB
    total_size_mb = total_size / (1024 * 1024)
    
    context = {
        'total_files': total_files,
        'total_size_mb': round(total_size_mb, 2),
        'debug_mode': settings.DEBUG,
    }
    
    return render(request, 'admin/system_settings.html', context)
