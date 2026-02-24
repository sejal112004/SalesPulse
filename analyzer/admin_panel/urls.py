# analyzer/admin_panel/urls.py
from django.urls import path
from . import views

app_name = 'admin_panel'

urlpatterns = [
    path('login/', views.admin_login, name='login'),
    path('logout/', views.admin_logout, name='logout'),
    path('dashboard/', views.admin_dashboard, name='dashboard'),
    path('users/', views.manage_users, name='manage_users'),
    path('users/<int:user_id>/', views.user_detail, name='user_detail'),
    path('users/<int:user_id>/toggle/', views.toggle_user_status, name='toggle_user_status'),
    path('users/<int:user_id>/delete/', views.delete_user, name='delete_user'),
    path('datasets/', views.manage_datasets, name='manage_datasets'),
    path('datasets/<int:dataset_id>/delete/', views.delete_dataset, name='delete_dataset'),
    path('settings/', views.system_settings, name='system_settings'),
]
