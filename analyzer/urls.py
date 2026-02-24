# analyzer/urls.py
from django.urls import path
from .views import test_view
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_file, name='upload'),
    path('upload/activate/<int:dataset_id>/', views.activate_dataset, name='activate_dataset'),
    path('upload/delete/<int:dataset_id>/', views.delete_dataset, name='delete_dataset'),
    path('upload/view/<int:dataset_id>/', views.view_dataset, name='view_dataset'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('signup/', views.signup, name='signup'),
    path('products/', views.products, name='products'),
    path('customers/', views.customers, name='customers'),
    path('profile/', views.profile_view, name='profile'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('alerts/', views.alerts_view, name='alerts'),
    path('forecasts/', views.forecasts, name='forecasts'),
    path('reports/', views.reports_view, name='reports'),
    path('regions/', views.regions, name='regions'),
    path('dashboard/explain/', views.explain_sales, name='explain_sales'),
    path('dashboard/create/', views.create_dashboard, name='create_dashboard'),
    path('dashboard/save/', views.save_chart, name='save_chart'),
    path('dashboard/delete/<int:chart_id>/', views.delete_chart, name='delete_chart'),
    path('dashboard/my/', views.my_dashboard, name='my_dashboard'),
    path('test/', test_view),
]
