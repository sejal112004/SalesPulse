# analyzer/views.py
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from analyzer.models import Dataset
from django.http import JsonResponse
import pandas as pd
import os
import json
import hashlib
from datetime import datetime
from django.db.models import Max
from .forecasting import aggregate_data, linear_regression_forecast
from .cleaning import clean_dataset
import numpy as np
from .services.datasets import (
    validate_dataset_schema,
    compute_basic_kpis,
    compare_values,
    infer_core_columns,
    read_data_file,
)

def get_or_load_active_dataset(request):
    """
    Helper to resolve and load the active dataset.
    Prioritizes session data, falls back to database 'is_current', then most recent.
    Updates session if a dataset is found in DB.
    Returns: (df, filename, file_path, dataset_id)
    """
    file_path = request.session.get('data_path')
    filename = request.session.get('filename')
    dataset_id = request.session.get('dataset_id')
    
    # 1. Validate Session Path
    if file_path and os.path.exists(file_path):
        try:
            df = read_data_file(file_path)
            if df is not None:
                return df, filename, file_path, dataset_id
        except Exception:
            # Session path failed, fall back to DB
            pass

    # 2. DB Lookups
    dataset = None
    if request.user.is_authenticated:
        # Try current first
        dataset = Dataset.objects.filter(user=request.user, is_current=True).first()
        # Fallback to most recent
        if not dataset:
            dataset = Dataset.objects.filter(user=request.user).order_by('-uploaded_at').first()
    
    if dataset:
        try:
            # Resolve path
            try:
                fpath = dataset.file.path
            except (ValueError, AttributeError):
                fpath = os.path.join(settings.MEDIA_ROOT, dataset.file.name)
            
            if os.path.exists(fpath):
                # Update Session
                request.session['data_path'] = fpath
                request.session['dataset_id'] = dataset.id
                request.session['filename'] = dataset.name
                request.session['data_upload_at'] = dataset.uploaded_at.isoformat()
                request.session.save()
                
                # Load DF
                # Load DF
                df = read_data_file(fpath)
                return df, dataset.name, fpath, dataset.id
        except Exception:
            pass

    return None, None, None, None


# -------------------------------------------------
# 1. MAIN LANDING PAGE (/)
# -------------------------------------------------
def index(request):
    if request.user.is_authenticated:
        return redirect('analyzer:dashboard')
    return render(request, 'index.html')


# -------------------------------------------------
# 2. LOGIN PAGE (/login/)
# -------------------------------------------------
def login_view(request):
    if request.method == 'POST':
        username_or_email = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        user = None

        # Try login with email first if input contains '@'
        if '@' in username_or_email:
            user_obj = User.objects.filter(email=username_or_email).first()
            if user_obj:
                user = authenticate(request, username=user_obj.username, password=password)

        # If login by email failed or input is username, try username login
        if user is None:
            user = authenticate(request, username=username_or_email, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect('analyzer:dashboard')
        else:
            messages.error(request, "Invalid username/email or password")

    return render(request, 'auth/login.html')


# -------------------------------------------------
# 3. LOGOUT
# -------------------------------------------------
def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out successfully!")
    return redirect('analyzer:login')


# -------------------------------------------------
# 4. SIGNUP PAGE (/signup/)
# -------------------------------------------------
def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password1 = request.POST.get('password1', '')
        password2 = request.POST.get('password2', '')

        if password1 != password2:
            messages.error(request, "Passwords do not match!")
            return redirect('analyzer:signup')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken!")
            return redirect('analyzer:signup')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered!")
            return redirect('analyzer:signup')

        user = User.objects.create_user(username=username, email=email, password=password1)
        user.save()
        messages.success(request, "Account created successfully! Please login.")
        return redirect('analyzer:login')

    return render(request, 'auth/signup.html')


# -------------------------------------------------
# 5. DASHBOARD PAGE (/dashboard/)
# -------------------------------------------------
@login_required
def dashboard(request):
    try:
        file_path = request.session.get('data_path')
        filename = request.session.get('filename', 'No file uploaded')
        dataset_id = request.session.get('dataset_id')
        session_upload_time = request.session.get('data_uploaded_at')
        
        # Always check for the most recent dataset first, preferring the one
        # marked as current so the dashboard automatically reflects new uploads.
        most_recent_dataset = (
            Dataset.objects.filter(user=request.user, is_current=True)
            .order_by('-uploaded_at')
            .first()
        ) or Dataset.objects.filter(user=request.user).order_by('-uploaded_at').first()
        need_to_load = False
        
        # Priority: Always use most recent dataset if available
        if most_recent_dataset:
            # Check if session dataset matches most recent
            if not dataset_id or dataset_id != most_recent_dataset.id:
                # Different dataset or no session - load most recent
                need_to_load = True
                dataset_id = most_recent_dataset.id
            else:
                # Same dataset_id - verify file path is correct and exists
                try:
                    dataset = Dataset.objects.get(id=dataset_id, user=request.user)
                    try:
                        expected_path = dataset.file.path
                    except (ValueError, AttributeError):
                        expected_path = os.path.join(settings.MEDIA_ROOT, dataset.file.name)
                    
                    # Verify file exists and path matches
                    if not os.path.exists(file_path) or file_path != expected_path:
                        if os.path.exists(expected_path):
                            file_path = expected_path
                            need_to_load = True
                        else:
                            # File doesn't exist - reload from database
                            need_to_load = True
                except Dataset.DoesNotExist:
                    # Dataset was deleted - load most recent
                    need_to_load = True
                    dataset_id = most_recent_dataset.id
        elif not file_path or not os.path.exists(file_path):
            # No datasets and no valid file path
            need_to_load = False  # Will show empty state

        # Load data if needed
        if need_to_load and most_recent_dataset:
            # First, try to get the active dataset from session dataset_id
            if dataset_id:
                try:
                    active_dataset = Dataset.objects.get(id=dataset_id, user=request.user)
                    # Use Django's file.path property - this is the correct way
                    try:
                        file_path = active_dataset.file.path
                    except (ValueError, AttributeError):
                        # Fallback if file.path doesn't work
                        file_path = os.path.join(settings.MEDIA_ROOT, active_dataset.file.name)
                    
                    if os.path.exists(file_path):
                        try:
                            df = read_data_file(file_path)
                            
                            if df is not None and len(df) > 0:
                                request.session['data_path'] = file_path
                                request.session['data_loaded'] = True
                                request.session['row_count'] = len(df)
                                request.session['filename'] = active_dataset.name
                                request.session['dataset_id'] = active_dataset.id
                                request.session['data_uploaded_at'] = active_dataset.uploaded_at.isoformat() if hasattr(active_dataset, 'uploaded_at') else None
                                request.session.save()
                                request.session.modified = True
                                filename = active_dataset.name
                                need_to_load = False
                        except Exception as e:
                            # If loading fails, try most recent dataset
                            import traceback
                            print(f"Error loading dataset {dataset_id}: {str(e)}")
                            print(traceback.format_exc())
                            need_to_load = True
                except Dataset.DoesNotExist:
                    # Dataset was deleted, try most recent
                    need_to_load = True
            
            # If still need to load, try the most recent dataset
            if need_to_load:
                user_datasets = Dataset.objects.filter(user=request.user).order_by('-uploaded_at').first()
                if user_datasets:
                    # Use Django's file.path property - this is the correct way
                    try:
                        file_path = user_datasets.file.path
                    except (ValueError, AttributeError):
                        # Fallback if file.path doesn't work
                        file_path = os.path.join(settings.MEDIA_ROOT, user_datasets.file.name)
                    
                    if os.path.exists(file_path):
                        try:
                            df = read_data_file(file_path)
                            
                            if df is not None and len(df) > 0:
                                request.session['data_path'] = file_path
                                request.session['data_loaded'] = True
                                request.session['row_count'] = len(df)
                                request.session['filename'] = user_datasets.name
                                request.session['dataset_id'] = user_datasets.id
                                request.session['data_uploaded_at'] = user_datasets.uploaded_at.isoformat() if hasattr(user_datasets, 'uploaded_at') else None
                                request.session.save()
                                request.session.modified = True
                                filename = user_datasets.name
                                need_to_load = False
                        except Exception as e:
                            # Log error but continue
                            import traceback
                            print(f"Error loading most recent dataset: {str(e)}")
                            print(traceback.format_exc())
                            pass

        if not file_path or not os.path.exists(file_path):
            # Final attempt: check if we have any datasets and load the most recent one
            final_dataset = Dataset.objects.filter(user=request.user).order_by('-uploaded_at').first()
            if final_dataset:
                try:
                    # Use Django's file.path property - this is the correct way
                    try:
                        final_path = final_dataset.file.path
                    except (ValueError, AttributeError):
                        # Fallback if file.path doesn't work
                        final_path = os.path.join(settings.MEDIA_ROOT, final_dataset.file.name)
                    
                    if os.path.exists(final_path):
                        file_path = final_path
                        filename = final_dataset.name
                        dataset_id = final_dataset.id
                        request.session['data_path'] = file_path
                        request.session['dataset_id'] = dataset_id
                        request.session['filename'] = filename
                        request.session['data_uploaded_at'] = final_dataset.uploaded_at.isoformat() if hasattr(final_dataset, 'uploaded_at') else None
                        request.session.save()
                        request.session.modified = True
                except Exception:
                    pass
            
            # If still no file, show empty state
            if not file_path or not os.path.exists(file_path):
                context = {
                    'has_data': False,
                    'filename': filename,
                    'total_revenue': 0,
                    'total_orders': 0,
                    'total_customers': 0,
                    'revenue_growth': 0,
                    'orders_growth': 0,
                    'customers_growth': 0,
                    'sales_trend_data': json.dumps({"labels": [], "datasets": []}),
                    'top_products': [],
                    'recent_activity': [],
                    'low_stock_items': [],
                }
                return render(request, 'dashboard.html', context)

        # Load the data - ensure file exists
        if not os.path.exists(file_path):
            messages.error(request, f"Data file not found. Please upload your dataset again.")
            context = {
                'has_data': False,
                'filename': filename,
                'total_revenue': 0,
                'total_orders': 0,
                'total_customers': 0,
                'revenue_growth': 0,
                'orders_growth': 0,
                'customers_growth': 0,
                'sales_trend_data': json.dumps({"labels": [], "datasets": []}),
                'top_products': [],
                'recent_activity': [],
                'low_stock_items': [],
            }
            return render(request, 'dashboard.html', context)
        
        # Load the data
        try:
            # Load the data using robust helper
            df = read_data_file(file_path)
            
            if df is None:
                messages.error(request, "Unsupported file format or error reading file.")
                context = {
                    'has_data': False,
                    'filename': filename,
                    'total_revenue': 0,
                    'total_orders': 0,
                    'total_customers': 0,
                    'revenue_growth': 0,
                    'orders_growth': 0,
                    'customers_growth': 0,
                    'sales_trend_data': json.dumps({"labels": [], "datasets": []}),
                    'top_products': [],
                    'recent_activity': [],
                    'low_stock_items': [],
                }
                return render(request, 'dashboard.html', context)
            
            if df is None or len(df) == 0:
                messages.warning(request, "Dataset is empty. Please upload a dataset with data.")
                context = {
                    'has_data': False,
                    'filename': filename,
                    'total_revenue': 0,
                    'total_orders': 0,
                    'total_customers': 0,
                    'revenue_growth': 0,
                    'orders_growth': 0,
                    'customers_growth': 0,
                    'sales_trend_data': json.dumps({"labels": [], "datasets": []}),
                    'top_products': [],
                    'recent_activity': [],
                    'low_stock_items': [],
                }
                return render(request, 'dashboard.html', context)
        except Exception as e:
            messages.error(request, f"Error reading data file: {str(e)}")
            import traceback
            print(f"Error reading file {file_path}: {str(e)}")
            print(traceback.format_exc())
            context = {
                'has_data': False,
                'filename': filename,
                'total_revenue': 0,
                'total_orders': 0,
                'total_customers': 0,
                'revenue_growth': 0,
                'orders_growth': 0,
                'customers_growth': 0,
                'sales_trend_data': json.dumps({"labels": [], "datasets": []}),
                'top_products': [],
                'recent_activity': [],
                'low_stock_items': [],
            }
            return render(request, 'dashboard.html', context)

        # Infer columns using standard helper
        inferred = infer_core_columns(df)
        order_date_col = inferred.get('date')
        customer_col = inferred.get('customer')
        product_col = inferred.get('product')
        category_col = inferred.get('category')
        qty_col = inferred.get('quantity')
        price_col = inferred.get('price')
        order_id_col = inferred.get('order_id')
        stock_col = inferred.get('stock')
        region_col = inferred.get('region')

        # Calculate revenue if we have quantity and price
        # Ensure data types are numeric for accurate calculations
        if qty_col and price_col:
            df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
            df['Revenue'] = df[qty_col] * df[price_col]
        else:
            df['Revenue'] = 0

        # ============================================
        # APPLY FILTERS (Date Range, Region, Product/Category)
        # ============================================
        original_df_size = len(df)
        date_range = request.GET.get('date_range', 'all')  # all, 7, 30, 90
        filter_region = request.GET.get('region', '')
        filter_product = request.GET.get('product', '')
        filter_category = request.GET.get('category', '')
        
        # Date range filter
        if order_date_col and date_range != 'all':
            try:
                df[order_date_col] = pd.to_datetime(df[order_date_col], errors='coerce')
                df = df.dropna(subset=[order_date_col])
                
                # Calculate max date from the dataset to handle historical data
                max_date = df[order_date_col].max()
                
                if date_range == '7':
                    cutoff_date = max_date - pd.Timedelta(days=7)
                elif date_range == '30':
                    cutoff_date = max_date - pd.Timedelta(days=30)
                elif date_range == '90':
                    cutoff_date = max_date - pd.Timedelta(days=90)
                else:
                    cutoff_date = None
                
                if cutoff_date:
                    df = df[df[order_date_col] >= cutoff_date]
            except Exception:
                pass
        
        # Region filter
        if filter_region and region_col:
            df = df[df[region_col].astype(str).str.lower() == filter_region.lower()]
        
        # Product filter
        if filter_product and product_col:
            df = df[df[product_col].astype(str).str.lower() == filter_product.lower()]
        
        # Category filter
        if filter_category and category_col:
            df = df[df[category_col].astype(str).str.lower() == filter_category.lower()]
        
        # Store filter info for template
        filters_applied = {
            'date_range': date_range,
            'region': filter_region,
            'product': filter_product,
            'category': filter_category,
            'rows_filtered': original_df_size - len(df) if original_df_size > len(df) else 0
        }

        # Ensure date column is datetime for calculations
        if order_date_col:
            try:
                df[order_date_col] = pd.to_datetime(df[order_date_col], errors='coerce')
                df = df.dropna(subset=[order_date_col])
            except Exception:
                pass

        # Calculate metrics with accurate data
        total_revenue = float(df['Revenue'].sum())
        total_orders = int(df[order_id_col].nunique() if order_id_col else len(df))
        total_customers = int(df[customer_col].nunique() if customer_col else 0)

        # --------------------------------------------
        # BASELINE DATASET & COMPARISON (current vs last)
        # --------------------------------------------
        baseline_dataset = (
            Dataset.objects.filter(user=request.user)
            .exclude(id=dataset_id)
            .order_by('-uploaded_at')
            .first()
        )

        revenue_comparison = None
        orders_comparison = None

        if baseline_dataset:
            try:
                try:
                    baseline_path = baseline_dataset.file.path
                except (ValueError, AttributeError):
                    baseline_path = os.path.join(settings.MEDIA_ROOT, baseline_dataset.file.name)

                baseline_df = None # Initialize baseline_df
                if os.path.exists(baseline_path):
                    baseline_df = read_data_file(baseline_path)

                    if baseline_df is not None and len(baseline_df) > 0:
                        # Reuse column inference for baseline.
                        b_inferred = infer_core_columns(baseline_df)
                        b_date_col = b_inferred.get('date')
                        b_qty_col = b_inferred.get('quantity')
                        b_price_col = b_inferred.get('price')
                        
                        if b_date_col and b_qty_col and b_price_col:
                            baseline_kpis = compute_basic_kpis(
                                baseline_df, b_date_col, b_qty_col, b_price_col
                            )
                            revenue_comparison = compare_values(
                                current=total_revenue,
                                baseline=baseline_kpis["total_revenue"],
                                threshold_pct=10.0,
                            )
                            orders_comparison = compare_values(
                                current=total_orders,
                                baseline=baseline_kpis["total_orders"],
                                threshold_pct=10.0,
                            )
            except Exception:
                revenue_comparison = None
                orders_comparison = None

        # ============================================
        # NEW KPIs: Top Product, Top Region, Returning Customers %
        # ============================================
        top_product_name = "N/A"
        top_product_revenue = 0
        top_region_name = "N/A"
        top_region_revenue = 0
        returning_customers_pct = 0
        revenue_yoy_growth = 0

        # Top Product
        if product_col and qty_col and price_col:
            product_revenue = df.groupby(product_col)['Revenue'].sum().sort_values(ascending=False)
            if not product_revenue.empty:
                top_product_name = str(product_revenue.index[0])
                top_product_revenue = float(product_revenue.iloc[0])

        # Top Region
        if region_col:
            region_revenue = df.groupby(region_col)['Revenue'].sum().sort_values(ascending=False)
            if not region_revenue.empty:
                top_region_name = str(region_revenue.index[0])
                top_region_revenue = float(region_revenue.iloc[0])

        # Returning Customers %
        if customer_col and order_id_col:
            try:
                # Count customers with more than one order
                customer_order_count = df.groupby(customer_col)[order_id_col].nunique()
                returning_customers = (customer_order_count > 1).sum()
                total_unique_customers = len(customer_order_count)
                if total_unique_customers > 0:
                    returning_customers_pct = (returning_customers / total_unique_customers) * 100
            except Exception:
                returning_customers_pct = 0

        # Calculate month-over-month and year-over-year growth
        revenue_growth_mom = 0
        revenue_growth_yoy = 0
        orders_growth = 0
        customers_growth = 0

        if order_date_col and len(df) > 0:
            try:
                # Get current month and previous month data
                current_date = pd.Timestamp.now()
                current_month_start = current_date.replace(day=1)
                previous_month_start = (current_month_start - pd.DateOffset(months=1))
                previous_month_end = current_month_start - pd.Timedelta(days=1)
                
                # Year-over-year: same month last year
                previous_year_start = current_month_start - pd.DateOffset(years=1)
                previous_year_end = current_month_start - pd.Timedelta(days=1)

                current_month_data = df[df[order_date_col] >= current_month_start]
                previous_month_data = df[
                    (df[order_date_col] >= previous_month_start) & 
                    (df[order_date_col] <= previous_month_end)
                ]
                previous_year_data = df[
                    (df[order_date_col] >= previous_year_start) & 
                    (df[order_date_col] <= previous_year_end)
                ]

                # Month-over-Month Growth
                if len(previous_month_data) > 0:
                    prev_revenue = previous_month_data['Revenue'].sum()
                    prev_orders = previous_month_data[order_id_col].nunique() if order_id_col else len(previous_month_data)
                    prev_customers = previous_month_data[customer_col].nunique() if customer_col else 0

                    curr_revenue = current_month_data['Revenue'].sum()
                    curr_orders = current_month_data[order_id_col].nunique() if order_id_col else len(current_month_data)
                    curr_customers = current_month_data[customer_col].nunique() if customer_col else 0

                    if prev_revenue > 0:
                        revenue_growth_mom = ((curr_revenue - prev_revenue) / prev_revenue) * 100
                    if prev_orders > 0:
                        orders_growth = ((curr_orders - prev_orders) / prev_orders) * 100
                    if prev_customers > 0:
                        customers_growth = ((curr_customers - prev_customers) / prev_customers) * 100

                # Year-over-Year Growth
                if len(previous_year_data) > 0:
                    prev_year_revenue = previous_year_data['Revenue'].sum()
                    curr_revenue = current_month_data['Revenue'].sum()
                    if prev_year_revenue > 0:
                        revenue_growth_yoy = ((curr_revenue - prev_year_revenue) / prev_year_revenue) * 100

                # Sales trend for last 12 months (or filtered period)
                df['YearMonth'] = df[order_date_col].dt.to_period('M')
                monthly_sales = df.groupby('YearMonth')['Revenue'].sum().reset_index()
                monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
                
                # Get last 12 months or all available if filtered
                if date_range == 'all':
                    monthly_sales = monthly_sales.tail(12)
                else:
                    # Show all months in filtered period
                    monthly_sales = monthly_sales.tail(12)
                
                sales_trend_data = {
                    "labels": monthly_sales['YearMonth'].tolist(),
                    "datasets": [{
                        "label": "Monthly Revenue",
                        "data": monthly_sales['Revenue'].round(2).tolist(),
                        "borderColor": "#36A2EB",
                        "backgroundColor": "rgba(54, 162, 235, 0.1)",
                        "fill": True,
                        "tension": 0.4
                    }]
                }
            except Exception as e:
                sales_trend_data = {"labels": [], "datasets": []}
        else:
            sales_trend_data = {"labels": [], "datasets": []}

        # Top products (for table)
        top_products = []
        if product_col and qty_col and price_col:
            group_cols = [product_col]
            if category_col:
                group_cols.append(category_col)
            
            product_stats = df.groupby(group_cols).agg({
                qty_col: 'sum',
                'Revenue': 'sum'
            }).reset_index()

            rename_map = {
                product_col: 'product_name',
                qty_col: 'units_sold'
            }
            if category_col:
                rename_map[category_col] = 'category'
            
            product_stats = product_stats.rename(columns=rename_map)
            
            if not category_col:
                product_stats['category'] = 'N/A'
            top_products_df = product_stats.nlargest(10, 'Revenue')
            top_products = top_products_df.to_dict('records')

        # Get filter options for dropdowns
        filter_options = {
            'regions': [],
            'products': [],
            'categories': []
        }
        
        if region_col:
            filter_options['regions'] = sorted(df[region_col].dropna().unique().tolist())
        if product_col:
            filter_options['products'] = sorted(df[product_col].dropna().unique().tolist())
        if category_col:
            filter_options['categories'] = sorted(df[category_col].dropna().unique().tolist())

        # Recent activity (latest orders)
        recent_activity = []
        if order_date_col and customer_col:
            try:
                recent_orders = df.nlargest(5, order_date_col) if order_date_col else df.tail(5)
                for _, row in recent_orders.iterrows():
                    customer_name = str(row.get(customer_col, 'Unknown'))
                    product_name = str(row.get(product_col, 'Product')) if product_col else 'Order'
                    order_date = row[order_date_col] if order_date_col else pd.Timestamp.now()
                    
                    # Calculate time ago
                    if isinstance(order_date, pd.Timestamp):
                        time_diff = pd.Timestamp.now() - order_date
                        if time_diff.days > 0:
                            time_ago = f"{time_diff.days} day{'s' if time_diff.days > 1 else ''} ago"
                        elif time_diff.seconds // 3600 > 0:
                            hours = time_diff.seconds // 3600
                            time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
                        else:
                            minutes = time_diff.seconds // 60
                            time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
                    else:
                        time_ago = "Recently"

                    revenue = row.get('Revenue', 0)
                    recent_activity.append({
                        'type': 'order',
                        'icon': 'cart-plus-fill',
                        'icon_color': 'primary',
                        'title': f'Order from {customer_name}',
                        'description': f'{product_name} - ₹{revenue:,.0f}',
                        'time_ago': time_ago
                    })
            except Exception as e:
                recent_activity = []

        # Add dataset upload activity
        if filename:
            recent_activity.insert(0, {
                'type': 'upload',
                'icon': 'upload',
                'icon_color': 'info',
                'title': 'Dataset uploaded successfully',
                'description': f'{filename}',
                'time_ago': 'Recently'
            })

        # Get low stock items for inventory insights
        low_stock_items = []
        if product_col and stock_col and stock_col in df.columns:
            stock_df = df.groupby(product_col)[stock_col].min().reset_index()
            stock_df = stock_df.rename(columns={product_col: 'product_name', stock_col: 'current_stock'})
            low_stock_df = stock_df[stock_df['current_stock'] <= 10].sort_values('current_stock')
            low_stock_items = low_stock_df.to_dict('records')

        # All datasets for the current user, used for the dashboard selector.
        user_datasets = Dataset.objects.filter(user=request.user).order_by('-uploaded_at')

        context = {
            'has_data': True,
            'filename': filename,
            'total_revenue': round(total_revenue, 2),
            'total_orders': total_orders,
            'total_customers': total_customers,
            'revenue_growth': round(revenue_growth_mom, 1),  # MoM growth
            'revenue_growth_yoy': round(revenue_growth_yoy, 1),  # YoY growth
            'orders_growth': round(orders_growth, 1),
            'customers_growth': round(customers_growth, 1),
            'top_product_name': top_product_name,
            'top_product_revenue': round(top_product_revenue, 2),
            'top_region_name': top_region_name,
            'top_region_revenue': round(top_region_revenue, 2),
            'returning_customers_pct': round(returning_customers_pct, 1),
            'sales_trend_data': json.dumps(sales_trend_data),
            'top_products': top_products,
            'recent_activity': recent_activity[:5],  # Limit to 5 items
            'low_stock_items': low_stock_items,  # Inventory insights
            'filters_applied': filters_applied,
            'filter_options': filter_options,
            # Dataset management & comparisons
            'datasets': user_datasets,
            'current_dataset_id': dataset_id,
            'baseline_dataset': baseline_dataset,
            'revenue_comparison': revenue_comparison,
            'orders_comparison': orders_comparison,
        }

    except Exception as e:
        messages.error(request, f"Error processing dashboard data: {str(e)}")
        context = {
            'has_data': False,
            'filename': request.session.get('filename', 'No file uploaded'),
            'total_revenue': 0,
            'total_orders': 0,
            'total_customers': 0,
            'revenue_growth': 0,
            'revenue_growth_yoy': 0,
            'orders_growth': 0,
            'customers_growth': 0,
            'top_product_name': 'N/A',
            'top_product_revenue': 0,
            'top_region_name': 'N/A',
            'top_region_revenue': 0,
            'returning_customers_pct': 0,
            'sales_trend_data': json.dumps({"labels": [], "datasets": []}),
            'top_products': [],
            'recent_activity': [],
            'low_stock_items': [],  # Empty when no data
            'filters_applied': {'date_range': 'all', 'region': '', 'product': '', 'category': '', 'rows_filtered': 0},
            'filter_options': {'regions': [], 'products': [], 'categories': []},
            'datasets': Dataset.objects.filter(user=request.user).order_by('-uploaded_at'),
            'current_dataset_id': request.session.get('dataset_id'),
            'baseline_dataset': None,
            'revenue_comparison': None,
            'orders_comparison': None,
        }

    return render(request, 'dashboard.html', context)


# -------------------------------------------------
# EXPLAIN MY SALES - AI-Powered Summary
# -------------------------------------------------
@login_required
def explain_sales(request):
    """
    Generate a natural language explanation of sales performance.
    Returns JSON response with AI-generated summary.
    """
    try:
        # Use helper
        df, _, file_path, _ = get_or_load_active_dataset(request)
        
        if df is None:
            return JsonResponse({
                'success': False,
                'summary': 'No data available. Please upload a dataset first.'
            })

        # Get filter parameters
        date_range = request.GET.get('date_range', 'all')
        filter_region = request.GET.get('region', '')
        filter_product = request.GET.get('product', '')
        filter_category = request.GET.get('category', '')

        # Load data (already loaded)
        # if file_path.endswith('.csv'): ...

        if df is None or len(df) == 0:
            return JsonResponse({
                'success': False,
                'summary': 'Dataset is empty. Please upload a dataset with data.'
            })

        # Infer columns using standard helper
        inferred = infer_core_columns(df)
        order_date_col = inferred.get('date')
        customer_col = inferred.get('customer')
        product_col = inferred.get('product')
        category_col = inferred.get('category')
        qty_col = inferred.get('quantity')
        price_col = inferred.get('price')
        order_id_col = inferred.get('order_id')
        region_col = inferred.get('region')

        # Calculate revenue
        if qty_col and price_col:
            df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
            df['Revenue'] = df[qty_col] * df[price_col]
        else:
            return JsonResponse({
                'success': False,
                'summary': 'Unable to calculate revenue. Please ensure your dataset has Quantity and Price columns.'
            })

        # Apply filters (same as dashboard)
        if order_date_col and date_range != 'all':
            try:
                df[order_date_col] = pd.to_datetime(df[order_date_col], errors='coerce')
                df = df.dropna(subset=[order_date_col])
                if date_range == '7':
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=7)
                elif date_range == '30':
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=30)
                elif date_range == '90':
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=90)
                else:
                    cutoff_date = None
                if cutoff_date:
                    df = df[df[order_date_col] >= cutoff_date]
            except Exception:
                pass

        if filter_region and region_col:
            df = df[df[region_col].astype(str).str.lower() == filter_region.lower()]
        if filter_product and product_col:
            df = df[df[product_col].astype(str).str.lower() == filter_product.lower()]
        if filter_category and category_col:
            df = df[df[category_col].astype(str).str.lower() == filter_category.lower()]

        # Calculate key metrics
        total_revenue = float(df['Revenue'].sum())
        total_orders = int(df[order_id_col].nunique() if order_id_col else len(df))
        total_customers = int(df[customer_col].nunique() if customer_col else 0)

        # Calculate growth
        revenue_growth_mom = 0
        revenue_growth_yoy = 0
        
        if order_date_col and len(df) > 0:
            try:
                df[order_date_col] = pd.to_datetime(df[order_date_col], errors='coerce')
                df = df.dropna(subset=[order_date_col])
                
                current_date = pd.Timestamp.now()
                current_month_start = current_date.replace(day=1)
                previous_month_start = (current_month_start - pd.DateOffset(months=1))
                previous_month_end = current_month_start - pd.Timedelta(days=1)
                previous_year_start = current_month_start - pd.DateOffset(years=1)
                previous_year_end = current_month_start - pd.Timedelta(days=1)

                current_month_data = df[df[order_date_col] >= current_month_start]
                previous_month_data = df[
                    (df[order_date_col] >= previous_month_start) & 
                    (df[order_date_col] <= previous_month_end)
                ]
                previous_year_data = df[
                    (df[order_date_col] >= previous_year_start) & 
                    (df[order_date_col] <= previous_year_end)
                ]

                if len(previous_month_data) > 0:
                    prev_revenue = previous_month_data['Revenue'].sum()
                    curr_revenue = current_month_data['Revenue'].sum()
                    if prev_revenue > 0:
                        revenue_growth_mom = ((curr_revenue - prev_revenue) / prev_revenue) * 100

                if len(previous_year_data) > 0:
                    prev_year_revenue = previous_year_data['Revenue'].sum()
                    curr_revenue = current_month_data['Revenue'].sum()
                    if prev_year_revenue > 0:
                        revenue_growth_yoy = ((curr_revenue - prev_year_revenue) / prev_year_revenue) * 100
            except Exception:
                pass

        # Get top contributors
        top_region_name = "N/A"
        top_region_pct = 0
        if region_col:
            region_revenue = df.groupby(region_col)['Revenue'].sum().sort_values(ascending=False)
            if not region_revenue.empty:
                top_region_name = str(region_revenue.index[0])
                top_region_revenue = float(region_revenue.iloc[0])
                if total_revenue > 0:
                    top_region_pct = (top_region_revenue / total_revenue) * 100

        top_product_name = "N/A"
        top_product_pct = 0
        if product_col:
            product_revenue = df.groupby(product_col)['Revenue'].sum().sort_values(ascending=False)
            if not product_revenue.empty:
                top_product_name = str(product_revenue.index[0])
                top_product_revenue = float(product_revenue.iloc[0])
                if total_revenue > 0:
                    top_product_pct = (top_product_revenue / total_revenue) * 100

        # Generate natural language summary
        summary_parts = []
        
        # Revenue overview
        if revenue_growth_mom != 0:
            if revenue_growth_mom > 0:
                summary_parts.append(f"Sales increased by {abs(revenue_growth_mom):.1f}% this month")
            else:
                summary_parts.append(f"Sales decreased by {abs(revenue_growth_mom):.1f}% this month")
        else:
            summary_parts.append(f"Total sales are ₹{total_revenue:,.0f}")

        # Top region contribution
        if top_region_name != "N/A" and top_region_pct > 0:
            if revenue_growth_mom > 0:
                summary_parts.append(f", mainly driven by Region {top_region_name} ({top_region_pct:.1f}% of total revenue)")
            else:
                summary_parts.append(f", with Region {top_region_name} contributing {top_region_pct:.1f}% of total revenue")

        # Top product contribution
        if top_product_name != "N/A" and top_product_pct > 0 and top_product_pct < 50:
            summary_parts.append(f". Top product '{top_product_name}' accounts for {top_product_pct:.1f}% of sales")

        # YoY comparison
        if revenue_growth_yoy != 0:
            if revenue_growth_yoy > 0:
                summary_parts.append(f". Year-over-year growth is {revenue_growth_yoy:.1f}%")
            else:
                summary_parts.append(f". Year-over-year shows a decline of {abs(revenue_growth_yoy):.1f}%")

        # Customer insights
        if total_customers > 0:
            avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
            summary_parts.append(f". You have {total_customers:,} customers with an average order value of ₹{avg_order_value:,.0f}")

        # Combine summary
        summary = "".join(summary_parts) + "."

        return JsonResponse({
            'success': True,
            'summary': summary,
            'metrics': {
                'total_revenue': round(total_revenue, 2),
                'revenue_growth_mom': round(revenue_growth_mom, 1),
                'revenue_growth_yoy': round(revenue_growth_yoy, 1),
                'top_region': top_region_name,
                'top_product': top_product_name,
            }
        })

    except Exception as e:
        import traceback
        print(f"Error generating sales explanation: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'summary': f'Error generating explanation: {str(e)}'
        })


# -------------------------------------------------
# 6. UPLOAD FILE PAGE (/upload/)
# -------------------------------------------------
@login_required
def activate_dataset(request, dataset_id):
    """Activate a dataset for dashboard visualization"""
    try:
        dataset = Dataset.objects.get(id=dataset_id, user=request.user)
        
        # Get the file path using Django's file.path property - this is the correct way
        try:
            file_path = dataset.file.path
        except (ValueError, AttributeError):
            # Fallback if file.path doesn't work
            file_path = os.path.join(settings.MEDIA_ROOT, dataset.file.name)
        
        if not os.path.exists(file_path):
            messages.error(request, f"File for dataset '{dataset.name}' not found on server at {file_path}.")
            return redirect('analyzer:upload')
        
        # Mark this dataset as the current version for the user and clear others.
        Dataset.objects.filter(user=request.user, is_current=True).update(is_current=False)
        dataset.is_current = True
        dataset.save(update_fields=["is_current"])

        # Load data to get row count
        try:

            df = read_data_file(file_path)
            
            if df is None:
                messages.error(request, "Unsupported file format or error reading file.")
                return redirect('analyzer:upload')
            
            if df is None or len(df) == 0:
                messages.error(request, f"Dataset '{dataset.name}' is empty.")
                return redirect('analyzer:upload')
            
            # Set session data
            request.session['data_path'] = file_path
            request.session['data_loaded'] = True
            request.session['row_count'] = len(df)
            request.session['filename'] = dataset.name
            request.session['dataset_id'] = dataset.id
            request.session['data_uploaded_at'] = dataset.uploaded_at.isoformat() if hasattr(dataset, 'uploaded_at') else None
            request.session.save()  # Ensure session is saved
            request.session.modified = True  # Force session modification
            
            messages.success(request, f"Dataset '{dataset.name}' activated! View dashboard to see analytics.")
            return redirect('analyzer:dashboard')
            
        except Exception as e:
            import traceback
            error_msg = f"Error loading dataset: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            messages.error(request, error_msg)
            return redirect('analyzer:upload')
            
    except Dataset.DoesNotExist:
        messages.error(request, "Dataset not found.")
        return redirect('analyzer:upload')


@login_required
def delete_dataset(request, dataset_id):
    """Delete a specific dataset"""
    try:
        dataset = Dataset.objects.get(id=dataset_id, user=request.user)
        dataset_name = dataset.name
        
        # Delete the file if it exists
        if dataset.file:
            file_path = os.path.join(settings.MEDIA_ROOT, dataset.file.name)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    messages.warning(request, f"File deleted from database but file system error: {str(e)}")
        
        # Also check if this was the active dataset in session
        if request.session.get('dataset_id') == dataset_id:
            request.session.pop('data_path', None)
            request.session.pop('data_loaded', None)
            request.session.pop('filename', None)
            request.session.pop('dataset_id', None)
            request.session.pop('row_count', None)
        
        dataset.delete()
        messages.success(request, f"Dataset '{dataset_name}' has been deleted successfully.")
    except Dataset.DoesNotExist:
        messages.error(request, "Dataset not found or you don't have permission to delete it.")
    except Exception as e:
        messages.error(request, f"Error deleting dataset: {str(e)}")
    
    return redirect('analyzer:upload')


@login_required
def upload_file(request):
    # Handle activate dataset request
    if request.method == 'POST' and 'activate_dataset' in request.POST:
        dataset_id = request.POST.get('dataset_id')
        if dataset_id:
            return activate_dataset(request, dataset_id)
    
    # Handle delete request
    if request.method == 'POST' and 'delete_dataset' in request.POST:
        dataset_id = request.POST.get('dataset_id')
        if dataset_id:
            return delete_dataset(request, dataset_id)
    
    if request.method == 'POST':
        # Support multiple file uploads (2-3 files)
        uploaded_files = request.FILES.getlist('sales_file')
        
        if not uploaded_files:
            messages.error(request, "Please select at least one file to upload.")
            return redirect('analyzer:upload')
        
        if len(uploaded_files) > 3:
            messages.warning(request, "Maximum 3 files allowed. Only the first 3 will be processed.")
            uploaded_files = uploaded_files[:3]
        
        upload_dir = os.path.join(settings.BASE_DIR, 'analyzer', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        # Existing schema signature from the user's current dataset, if any,
        # to enforce consistency across versions.
        existing_current = (
            Dataset.objects.filter(user=request.user, is_current=True)
            .order_by('-uploaded_at')
            .first()
        )
        existing_signature = existing_current.schema_signature if existing_current else None

        created_datasets = []

        for file in uploaded_files:
            filename = file.name
            file_path = os.path.join(upload_dir, filename)

            try:
                # Save file temporarily to validate and read it.
                with open(file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

                # Validate file format and read data
                # Validate file format and read data
                df = read_data_file(file_path)
                
                if df is None:
                    messages.error(request, f"Unsupported file format or error reading '{filename}'. Use .csv or .xlsx.")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    continue

                # Schema + data validation
                try:
                    schema_signature, _meta = validate_dataset_schema(df, existing_signature=existing_signature)
                except ValueError as ve:
                    messages.error(request, f"'{filename}' rejected: {ve}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    continue

                # Create Dataset model instance to track in database
                from django.core.files import File as DjangoFile

                with open(file_path, 'rb') as f:
                    dataset = Dataset(
                        user=request.user,
                        name=filename,
                    )
                    # Let FileField handle storage under MEDIA_ROOT/datasets/
                    dataset.file.save(filename, DjangoFile(f, name=filename), save=False)

                    # Assign schema signature and a monotonic version number per user.
                    dataset.schema_signature = schema_signature
                    last_version = (
                        Dataset.objects.filter(user=request.user)
                        .aggregate(max_ver=Max("version"))
                        .get("max_ver")
                    ) or 0
                    dataset.version = last_version + 1
                    dataset.is_current = False  # will be set on the latest below
                    dataset.save()

                created_datasets.append(dataset)

            except Exception as e:
                messages.error(request, f"Error reading file '{filename}': {str(e)}")
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
                continue

        if not created_datasets:
            messages.error(request, "No files were successfully processed.")
            return redirect('analyzer:upload')

        # Mark the newest uploaded dataset as current, previous as baseline.
        # (Baseline logic is handled later when computing comparisons.)
        Dataset.objects.filter(user=request.user, is_current=True).update(is_current=False)
        current_dataset = max(created_datasets, key=lambda d: d.version)
        current_dataset.is_current = True
        current_dataset.save(update_fields=["is_current"])

        # Verify the current dataset file and push it into the session
        try:
            file_path_to_use = current_dataset.file.path
        except (ValueError, AttributeError):
            file_path_to_use = os.path.join(settings.MEDIA_ROOT, current_dataset.file.name)

        if not os.path.exists(file_path_to_use):
            messages.error(request, f"File not found at {file_path_to_use}. Please try uploading again.")
            return redirect('analyzer:upload')

        try:

            df = read_data_file(file_path_to_use)
            verify_df = df # Alias for compatibility

            if verify_df is None or len(verify_df) == 0:
                messages.error(request, "Uploaded file is empty. Please upload a file with data.")
                return redirect('analyzer:upload')

            request.session['data_path'] = file_path_to_use
            request.session['data_loaded'] = True
            request.session['row_count'] = len(verify_df)
            request.session['filename'] = current_dataset.name
            request.session['dataset_id'] = current_dataset.id
            request.session['data_uploaded_at'] = current_dataset.uploaded_at.isoformat() if hasattr(current_dataset, 'uploaded_at') else None
            request.session.save()
            request.session.modified = True

            if len(created_datasets) == 1:
                messages.success(request, f"Success! '{current_dataset.name}' uploaded → {len(verify_df):,} rows loaded.")
            else:
                messages.success(request, f"Success! {len(created_datasets)} datasets uploaded. Current version: v{current_dataset.version}.")

            return redirect('analyzer:dashboard')
        except Exception as e:
            messages.error(request, f"Error reading uploaded file: {str(e)}")
            return redirect('analyzer:upload')
    
    # Get all user's datasets for display
    user_datasets = Dataset.objects.filter(user=request.user).order_by('-uploaded_at')
    
    # Get current active dataset info
    active_dataset_id = request.session.get('dataset_id')
    active_filename = request.session.get('filename', 'No file uploaded')
    
    context = {
        'datasets': user_datasets,
        'active_dataset_id': active_dataset_id,
        'active_filename': active_filename,
        'total_datasets': user_datasets.count(),
    }
    
    return render(request, 'upload.html', context)


# -------------------------------------------------
# 7. PRODUCTS PAGE (/products/) - DATA-DRIVEN
# -------------------------------------------------
@login_required
def products(request):
    """
    Product-level insights using the uploaded dataset.
    Expects at least: Product, Category (optional), Quantity, Price, and optionally Profit or Cost.
    """
    # file_path = request.session.get('data_path')
    # filename = request.session.get('filename', 'No file uploaded')
    
    # Use helper to get data
    df, filename, file_path, _ = get_or_load_active_dataset(request)

    total_products = 0
    low_stock_count = 0  # only set if we have a Stock column
    avg_margin = None
    top_product = "N/A"
    top_products = []
    category_data = {"labels": [], "datasets": []}
    margin_data = {"labels": [], "datasets": []}
    has_profit_data = False

    try:
        if df is None or df.empty:
            messages.info(request, "Upload a dataset first to see product insights.")
        else:
            # df is already loaded
            # if file_path.endswith('.csv'): ...


            # Infer columns using standard helper
            inferred = infer_core_columns(df)
            product_col = inferred.get('product')
            category_col = inferred.get('category')
            qty_col = inferred.get('quantity')
            price_col = inferred.get('price')
            profit_col = inferred.get('profit')
            cost_col = inferred.get('cost')
            stock_col = inferred.get('stock')

            # If price is missing but we have Derived_Price (from normalization), use that
            if not price_col and 'Derived_Price' in df.columns:
                price_col = 'Derived_Price'

            if not product_col or not qty_col or not price_col:
                messages.warning(
                    request,
                    "Products page requires at least Product, Quantity and Price columns in your file."
                )
            else:
                # Ensure numeric types for accurate calculations
                df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
                df[price_col] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
                
                # Revenue
                df['Revenue'] = df[qty_col] * df[price_col]

                # Profit margin if we have profit or cost
                if profit_col:
                    df['Profit'] = pd.to_numeric(df[profit_col], errors='coerce').fillna(0)
                elif cost_col:
                    df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce').fillna(0)
                    df['Profit'] = df['Revenue'] - (df[cost_col] * df[qty_col])
                else:
                    df['Profit'] = None

                # Aggregate per product
                group_cols = [product_col]
                if category_col:
                    group_cols.append(category_col)

                prod_stats = (
                    df.groupby(group_cols)
                    .agg(
                        units_sold=(qty_col, 'sum'),
                        revenue=('Revenue', 'sum'),
                        profit=('Profit', 'sum'),
                    )
                    .reset_index()
                )

                # Rename columns for template
                rename_map = {
                    product_col: 'product_name',
                    qty_col: 'units_sold'
                }
                if category_col:
                    rename_map[category_col] = 'category'
                
                prod_stats = prod_stats.rename(columns=rename_map)

                if not category_col:
                    prod_stats['category'] = 'N/A'

                total_products = int(prod_stats['product_name'].nunique())

                # Top product by revenue
                if not prod_stats.empty:
                    top_row = prod_stats.sort_values('revenue', ascending=False).iloc[0]
                    top_product = str(top_row['product_name'])

                # Low stock (only if stock_col exists)
                if stock_col and stock_col in df.columns:
                    stock_stats = (
                        df.groupby(product_col)[stock_col].min().reset_index()
                    )
                    # consider low stock if <= 10 units
                    low_stock_count = int((stock_stats[stock_col] <= 10).sum())

                # Profit margin calculation - ensure accurate calculations
                if prod_stats['profit'].notna().any() and (prod_stats['profit'] != 0).any():
                    has_profit_data = True
                    # avoid division by zero - use safe division
                    prod_stats['margin_pct'] = prod_stats.apply(
                        lambda r: (r['profit'] / r['revenue'] * 100)
                        if r['revenue'] > 0 and r['profit'] is not None and pd.notna(r['profit'])
                        else 0,
                        axis=1,
                    )
                    # Filter out invalid margins for average calculation
                    valid_margins = prod_stats[prod_stats['margin_pct'].notna() & (prod_stats['revenue'] > 0)]['margin_pct']
                    avg_margin = float(valid_margins.mean()) if len(valid_margins) > 0 else 0
                else:
                    prod_stats['margin_pct'] = 0
                    avg_margin = 0

                # Top 10 products by revenue for table
                top_products_df = prod_stats.sort_values('revenue', ascending=False).head(10)
                top_products = [
                    {
                        "rank": idx + 1,
                        "product_name": str(row['product_name']),
                        "category": str(row.get('category', 'N/A')),
                        "units_sold": float(row['units_sold']),
                        "revenue": float(row['revenue']),
                        "profit": float(row['profit']) if row['profit'] is not None else None,
                        "margin_pct": float(row['margin_pct']),
                        "stock": int(df[df[product_col] == row['product_name']][stock_col].min())
                        if stock_col and stock_col in df.columns
                        else None,
                    }
                    for idx, row in top_products_df.iterrows()
                ]

                # Revenue by category chart
                if category_col:
                    cat_stats = (
                        df.groupby(category_col)['Revenue']
                        .sum()
                        .reset_index()
                        .rename(columns={category_col: 'category', 'Revenue': 'revenue'})
                    )
                    category_data = {
                        "labels": cat_stats['category'].astype(str).tolist(),
                        "datasets": [
                            {
                                "label": "Revenue by Category",
                                "data": cat_stats['revenue'].round(0).tolist(),
                                "backgroundColor": [
                                    "#FF6384",
                                    "#36A2EB",
                                    "#FFCE56",
                                    "#4BC0C0",
                                    "#9966FF",
                                    "#FF9F40",
                                ][: len(cat_stats)]
                                or ["#36A2EB"],
                            }
                        ],
                    }

                # Top 10 by profit margin for bar chart (if we have profit)
                if has_profit_data:
                    margin_top = (
                        prod_stats.sort_values('margin_pct', ascending=False)
                        .head(10)
                    )
                    margin_data = {
                        "labels": margin_top['product_name'].astype(str).tolist(),
                        "datasets": [
                            {
                                "label": "Profit Margin %",
                                "data": margin_top['margin_pct'].round(1).tolist(),
                                "backgroundColor": "#00bfa5",
                            }
                        ],
                    }

    except Exception as e:
        messages.error(request, f"Error building product insights: {str(e)}")

    context = {
        'filename': filename,
        'total_products': total_products,
        'top_product': top_product,
        'low_stock_count': low_stock_count,
        'avg_margin': round(avg_margin, 1) if avg_margin is not None else None,
        'top_products': top_products,
        'category_data': json.dumps(category_data),
        'margin_data': json.dumps(margin_data),
        'has_profit_data': has_profit_data,
    }

    return render(request, 'products.html', context)


# -------------------------------------------------
# 8. CUSTOMERS PAGE (/customers/)
# -------------------------------------------------
@login_required
def customers(request):
    try:
        # file_path = request.session.get('data_path')
        # filename = request.session.get('filename', 'No file uploaded')
        
        # Use helper
        df, filename, file_path, _ = get_or_load_active_dataset(request)

        if df is None or df.empty:
            # No file or empty
            context = {
                'filename': filename,
                'total_customers': 0,
                'total_customer_revenue': 0,
                'avg_customer_value': 0,
                'top_customer': "N/A",
                'top_customers': [],
                'top_customers_data': json.dumps({"labels": [], "datasets": [{"data": []}]}),
                'region_data': json.dumps({"labels": [], "datasets": [{"data": []}]}),
                'has_region': False,
            }
            return render(request, 'customers.html', context)
        else:
            # df is loaded
            # Infer columns using standard helper
            inferred = infer_core_columns(df)
            cust_col = inferred.get('customer')
            region_col = inferred.get('region')
            qty_col = inferred.get('quantity')
            price_col = inferred.get('price')

            # If price is missing but we have Derived_Price (from normalization), use that
            if not price_col and 'Derived_Price' in df.columns:
                price_col = 'Derived_Price'

            if not cust_col or not qty_col or not price_col:
                messages.warning(request, "Some required columns (Customer, Quantity, Price) are missing in your file.")
                context = {
                    'filename': filename,
                    'total_customers': 0,
                    'total_customer_revenue': 0,
                    'avg_customer_value': 0,
                    'top_customer': "N/A",
                    'top_customers': [],
                    'top_customers_data': json.dumps({"labels": [], "datasets": [{"data": []}]}),
                    'region_data': json.dumps({"labels": [], "datasets": [{"data": []}]}),
                    'has_region': False,
                }
                return render(request, 'customers.html', context)

            df['Revenue'] = df[qty_col] * df[price_col]

            customer_df = df.groupby(cust_col).agg({'Revenue': 'sum', qty_col: 'count'}).rename(columns={qty_col: 'order_count'}).reset_index()
            customer_df = customer_df.rename(columns={cust_col: 'customer_name'})
            customer_df['avg_order_value'] = customer_df['Revenue'] / customer_df['order_count']

            top_customers = customer_df.nlargest(20, 'Revenue').round(0)
            top_customers_list = top_customers.to_dict('records')

            top10 = customer_df.nlargest(10, 'Revenue')
            top_customers_data = {
                "labels": top10['customer_name'].tolist(),
                "datasets": [{"label": "Total Revenue", "data": top10['Revenue'].round(0).tolist(),
                              "backgroundColor": "#36A2EB"}]
            }

            has_region = bool(region_col)
            if has_region:
                region_sales = df.groupby(region_col)['Revenue'].sum().round(0)
                region_data = {
                    "labels": region_sales.index.tolist(),
                    "datasets": [{"data": region_sales.values.tolist(),
                                  "backgroundColor": ["#FF6384", "#4BC0C0", "#FFCE56", "#9966FF", "#FF9F40"]}]
                }
            else:
                region_data = {"labels": [], "datasets": [{"data": []}]}

            context = {
                'filename': filename,
                'total_customers': customer_df['customer_name'].nunique(),
                'total_customer_revenue': round(customer_df['Revenue'].sum()),
                'avg_customer_value': round(customer_df['Revenue'].mean()),
                'top_customer': top_customers.iloc[0]['customer_name'] if not top_customers.empty else "N/A",
                'top_customers': top_customers_list,
                'top_customers_data': json.dumps(top_customers_data),
                'region_data': json.dumps(region_data),
                'has_region': has_region,
            }

    except Exception as e:
        messages.error(request, f"Error processing customer data: {str(e)}")
        context = {
            'filename': filename if 'filename' in locals() else "No file uploaded",
            'total_customers': 0,
            'total_customer_revenue': 0,
            'avg_customer_value': 0,
            'top_customer': "N/A",
            'top_customers': [],
            'top_customers_data': json.dumps({"labels": [], "datasets": [{"data": []}]}),
            'region_data': json.dumps({"labels": [], "datasets": [{"data": []}]}),
            'has_region': False,
        }

    return render(request, 'customers.html', context)


# -------------------------------------------------
# 9. PROFILE, ALERTS
# -------------------------------------------------
@login_required
def profile_view(request):
    """
    Profile view with photo upload and bio management.
    Production-ready with proper error handling.
    """
    from analyzer.models import UserProfile
    from django.db import IntegrityError, transaction
    
    try:
        # Get or create user profile with proper error handling
        try:
            profile, created = UserProfile.objects.get_or_create(
                user=request.user,
                defaults={
                    'bio': '',
                    'location': '',
                }
            )
        except IntegrityError:
            # Handle race condition where profile might be created between check and create
            try:
                profile = UserProfile.objects.get(user=request.user)
            except UserProfile.DoesNotExist:
                # If still doesn't exist, create it manually
                profile = UserProfile.objects.create(
                    user=request.user,
                    bio='',
                    location=''
                )
        
        if request.method == 'POST':
            # Handle photo upload
            if 'photo' in request.FILES:
                try:
                    photo_file = request.FILES['photo']
                    # Validate file size (max 5MB)
                    if photo_file.size > 5 * 1024 * 1024:
                        messages.error(request, "Image file too large. Maximum size is 5MB.")
                        return redirect('analyzer:profile')
                    
                    # Validate file type
                    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp']
                    if photo_file.content_type not in allowed_types:
                        messages.error(request, "Invalid file type. Please upload a JPEG, PNG, GIF, or WebP image.")
                        return redirect('analyzer:profile')
                    
                    profile.image = photo_file
                    profile.save()
                    messages.success(request, "Profile photo updated successfully!")
                    return redirect('analyzer:profile')
                except Exception as e:
                    messages.error(request, f"Error uploading photo: {str(e)}")
                    return redirect('analyzer:profile')
            
            # Handle profile update (bio and location)
            if 'bio' in request.POST or 'location' in request.POST:
                try:
                    bio = request.POST.get('bio', '').strip()
                    location = request.POST.get('location', '').strip()
                    
                    # Validate bio length
                    if len(bio) > 500:
                        messages.error(request, "Bio must be 500 characters or less.")
                        return redirect('analyzer:profile')
                    
                    # Validate location length
                    if len(location) > 100:
                        messages.error(request, "Location must be 100 characters or less.")
                        return redirect('analyzer:profile')
                    
                    profile.bio = bio
                    profile.location = location
                    profile.save()
                    messages.success(request, "Profile updated successfully!")
                    return redirect('analyzer:profile')
                except Exception as e:
                    messages.error(request, f"Error updating profile: {str(e)}")
                    return redirect('analyzer:profile')
        
        # Prepare context with safe defaults
        context = {
            'profile': profile,
            'user': request.user,
        }
        return render(request, 'includes/profile.html', context)
        
    except Exception as e:
        # Catch-all error handler for unexpected issues
        import traceback
        print(f"Error in profile_view: {str(e)}")
        print(traceback.format_exc())
        messages.error(request, "An error occurred while loading your profile. Please try again.")
        
        # Return a safe context even if profile doesn't exist
        context = {
            'profile': None,
            'user': request.user,
        }
        return render(request, 'includes/profile.html', context)


@login_required
def alerts_view(request):
    """
    Dynamic alerts based on uploaded data analysis.
    Generates alerts for: low stock, declining sales, low margins, top performers, etc.
    Supports removing alerts and adding custom alerts.
    """
    
    # Handle POST requests for adding/removing alerts
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'remove_alert':
            alert_id = request.POST.get('alert_id')
            dismissed = request.session.get('dismissed_alerts', [])
            if alert_id and alert_id not in dismissed:
                dismissed.append(alert_id)
                request.session['dismissed_alerts'] = dismissed
                messages.success(request, "Alert removed successfully.")
        
        elif action == 'add_alert':
            title = request.POST.get('title', '').strip()
            message = request.POST.get('message', '').strip()
            alert_type = request.POST.get('alert_type', 'info')
            icon = request.POST.get('icon', 'info-circle-fill')
            
            if title and message:
                custom_alerts = request.session.get('custom_alerts', [])
                alert_id = f"custom_{hashlib.md5(f"{title}{message}{datetime.now()}".encode()).hexdigest()[:8]}"
                
                # Map alert_type to icon_color
                type_to_color = {
                    'info': 'info',
                    'success': 'success',
                    'warning': 'warning',
                    'danger': 'danger'
                }
                
                custom_alerts.append({
                    'id': alert_id,
                    'type': alert_type,
                    'icon': icon,
                    'icon_color': type_to_color.get(alert_type, 'info'),
                    'title': title,
                    'message': message,
                    'time_ago': 'Just now',
                    'priority': 'low',
                    'is_custom': True
                })
                request.session['custom_alerts'] = custom_alerts
                messages.success(request, "Custom alert added successfully.")
            else:
                messages.error(request, "Please provide both title and message.")
        
        return redirect('analyzer:alerts')
    
        return redirect('analyzer:alerts')
    
    # Use helper
    df, filename, file_path, _ = get_or_load_active_dataset(request)
    
    alerts = []
    dismissed_alerts = request.session.get('dismissed_alerts', [])
    custom_alerts = request.session.get('custom_alerts', [])

    try:
        if df is None or df.empty:
            alert_id = "no_data"
            if alert_id not in dismissed_alerts:
                alerts.append({
                    'id': alert_id,
                    'type': 'info',
                    'icon': 'upload',
                    'icon_color': 'info',
                    'title': 'No Data Uploaded',
                    'message': 'Upload a sales dataset to receive intelligent alerts and insights.',
                    'time_ago': 'Just now',
                    'action_url': '/upload/',
                    'action_text': 'Upload Data'
                })
        else:
            # Load data (already loaded)
            # if file_path.endswith('.csv'): ...


            # Infer columns using standard helper
            inferred = infer_core_columns(df)
            product_col = inferred.get('product')
            category_col = inferred.get('category')
            qty_col = inferred.get('quantity')
            price_col = inferred.get('price')
            stock_col = inferred.get('stock')
            profit_col = inferred.get('profit')
            date_col = inferred.get('date')
            region_col = inferred.get('region')

            # Calculate revenue
            if qty_col and price_col:
                df['Revenue'] = df[qty_col] * df[price_col]

            # 1. Low Stock Alerts
            if stock_col and stock_col in df.columns:
                low_stock_threshold = 10
                low_stock_products = df[df[stock_col] <= low_stock_threshold]
                if not low_stock_products.empty:
                    group_cols = [product_col]
                    if region_col:
                        group_cols.append(region_col)
                        
                    for _, row in low_stock_products.groupby(group_cols).first().iterrows():
                        product_name = str(row.get(product_col, 'Unknown Product'))
                        stock_level = int(row[stock_col])
                        region = str(row.get(region_col, '')) if region_col else ''
                        region_text = f" in Region {region}" if region else ""
                        
                        alert_id = f"low_stock_{hashlib.md5(f"{product_name}{region}".encode()).hexdigest()[:8]}"
                        if alert_id not in dismissed_alerts:
                            alerts.append({
                                'id': alert_id,
                                'type': 'danger',
                                'icon': 'exclamation-triangle-fill',
                                'icon_color': 'danger',
                                'title': f'Low Stock Alert: {product_name}',
                                'message': f'Only {stock_level} unit(s) remaining{region_text}. Consider restocking soon.',
                                'time_ago': 'Recent',
                                'priority': 'high'
                            })

            # 2. Top Performer Alert
            if product_col and qty_col and price_col:
                product_revenue = df.groupby(product_col)['Revenue'].sum().sort_values(ascending=False)
                if not product_revenue.empty:
                    top_product = product_revenue.index[0]
                    top_revenue = float(product_revenue.iloc[0])
                    
                    alert_id = f"top_performer_{hashlib.md5(str(top_product).encode()).hexdigest()[:8]}"
                    if alert_id not in dismissed_alerts:
                        alerts.append({
                            'id': alert_id,
                            'type': 'success',
                            'icon': 'star-fill',
                            'icon_color': 'success',
                            'title': 'Top Performer',
                            'message': f'"{top_product}" is your best-selling product with ₹{top_revenue:,.0f} in revenue!',
                            'time_ago': 'Today',
                            'priority': 'medium'
                        })

            # 3. Low Profit Margin Alert
            if profit_col and qty_col and price_col:
                df['Profit'] = df[profit_col]
                df['Margin_Pct'] = (df['Profit'] / df['Revenue'] * 100).fillna(0)
                low_margin_threshold = 5  # Less than 5% margin
                
                product_margins = df.groupby(product_col)['Margin_Pct'].mean()
                low_margin_products = product_margins[product_margins < low_margin_threshold]
                
                if not low_margin_products.empty:
                    worst_product = low_margin_products.idxmin()
                    worst_margin = float(low_margin_products.min())
                    
                    alert_id = f"low_margin_{hashlib.md5(str(worst_product).encode()).hexdigest()[:8]}"
                    if alert_id not in dismissed_alerts:
                        alerts.append({
                            'id': alert_id,
                            'type': 'warning',
                            'icon': 'graph-down-arrow',
                            'icon_color': 'warning',
                            'title': 'Low Profit Margin Alert',
                            'message': f'"{worst_product}" has a low profit margin of {worst_margin:.1f}%. Review pricing strategy.',
                            'time_ago': 'Recent',
                            'priority': 'medium'
                        })

            # 4. Declining Sales Trend Alert
            if date_col and qty_col and price_col:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df.dropna(subset=[date_col])
                    df['YearMonth'] = df[date_col].dt.to_period('M')
                    monthly_revenue = df.groupby('YearMonth')['Revenue'].sum().sort_index()
                    
                    if len(monthly_revenue) >= 2:
                        recent_months = monthly_revenue.tail(2)
                        if len(recent_months) == 2:
                            prev_revenue = float(recent_months.iloc[0])
                            curr_revenue = float(recent_months.iloc[1])
                            
                            if prev_revenue > 0:
                                decline_pct = ((curr_revenue - prev_revenue) / prev_revenue) * 100
                                
                                if decline_pct < -10:  # More than 10% decline
                                    alert_id = "sales_decline"
                                    if alert_id not in dismissed_alerts:
                                        alerts.append({
                                            'id': alert_id,
                                            'type': 'warning',
                                            'icon': 'arrow-down-circle-fill',
                                            'icon_color': 'warning',
                                            'title': 'Sales Decline Detected',
                                            'message': f'Sales decreased by {abs(decline_pct):.1f}% compared to previous month. Investigate causes.',
                                            'time_ago': 'Recent',
                                            'priority': 'high'
                                        })
                                elif curr_revenue > prev_revenue * 1.2:  # More than 20% growth
                                    alert_id = "sales_growth"
                                    if alert_id not in dismissed_alerts:
                                        alerts.append({
                                            'id': alert_id,
                                            'type': 'success',
                                            'icon': 'arrow-up-circle-fill',
                                            'icon_color': 'success',
                                            'title': 'Strong Sales Growth!',
                                            'message': f'Sales increased by {decline_pct:.1f}% compared to previous month. Great performance!',
                                            'time_ago': 'Recent',
                                            'priority': 'medium'
                                        })
                except Exception:
                    pass  # Skip if date parsing fails

            # 5. Dataset Upload Success Alert
            if filename and filename != 'No file uploaded':
                alert_id = f"upload_{hashlib.md5(filename.encode()).hexdigest()[:8]}"
                if alert_id not in dismissed_alerts:
                    alerts.append({
                        'id': alert_id,
                        'type': 'info',
                        'icon': 'check-circle-fill',
                        'icon_color': 'info',
                        'title': 'Dataset Uploaded Successfully',
                        'message': f'Your sales data file "{filename}" has been processed and analyzed.',
                        'time_ago': 'Recently',
                        'priority': 'low'
                    })

            # 6. Missing Critical Columns Alert
            missing_cols = []
            if not product_col:
                missing_cols.append('Product')
            if not qty_col:
                missing_cols.append('Quantity')
            if not price_col:
                missing_cols.append('Price')
            
            if missing_cols:
                alert_id = "missing_columns"
                if alert_id not in dismissed_alerts:
                    alerts.append({
                        'id': alert_id,
                        'type': 'warning',
                        'icon': 'exclamation-circle-fill',
                        'icon_color': 'warning',
                        'title': 'Missing Data Columns',
                        'message': f'Your dataset is missing: {", ".join(missing_cols)}. Some features may be limited.',
                        'time_ago': 'Recent',
                        'priority': 'medium'
                    })

    except Exception as e:
        alert_id = "error_processing"
        if alert_id not in dismissed_alerts:
            alerts.append({
                'id': alert_id,
                'type': 'danger',
                'icon': 'exclamation-triangle-fill',
                'icon_color': 'danger',
                'title': 'Error Processing Alerts',
                'message': f'An error occurred while analyzing your data: {str(e)}',
                'time_ago': 'Just now',
                'priority': 'high'
            })

    # Add custom alerts
    for custom_alert in custom_alerts:
        if custom_alert.get('id') not in dismissed_alerts:
            alerts.append(custom_alert)

    # Sort alerts by priority (high -> medium -> low), then custom alerts last
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    alerts.sort(key=lambda x: (
        priority_order.get(x.get('priority', 'low'), 2),
        1 if x.get('is_custom') else 0
    ))

    context = {
        'alerts': alerts,
        'filename': filename,
        'has_data': file_path and os.path.exists(file_path) if file_path else False
    }

    return render(request, 'alerts.html', context)


@login_required
def reports_view(request):
    """
    Dynamic sales reports based on uploaded data.
    Generates comprehensive sales analytics and insights.
    """
    # Use helper
    df, filename, file_path, _ = get_or_load_active_dataset(request)

    # Default values
    total_sales = 0
    avg_monthly_sales = 0
    best_month = "N/A"
    best_month_revenue = 0
    expected_growth = 0
    product_performance = []
    category_performance = []
    monthly_sales_data = {"labels": [], "datasets": []}
    has_data = False

    try:
        if df is None or df.empty:
            messages.info(request, "Upload a dataset first to see detailed reports.")
        else:
            has_data = True
            # Load data (already loaded)
            # if file_path.endswith('.csv'): ...

            # Infer columns using standard helper
            inferred = infer_core_columns(df)
            product_col = inferred.get('product')
            category_col = inferred.get('category')
            qty_col = inferred.get('quantity')
            price_col = inferred.get('price')
            date_col = inferred.get('date')
            profit_col = inferred.get('profit')

            if qty_col and price_col:
                df['Revenue'] = df[qty_col] * df[price_col]
                total_sales = float(df['Revenue'].sum())

                # Monthly Sales Analysis
                if date_col:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.dropna(subset=[date_col])
                        df['YearMonth'] = df[date_col].dt.to_period('M')
                        monthly_revenue = df.groupby('YearMonth')['Revenue'].sum().sort_index()
                        
                        if not monthly_revenue.empty:
                            # Average monthly sales
                            avg_monthly_sales = float(monthly_revenue.mean())
                            
                            # Best performing month
                            best_month_period = monthly_revenue.idxmax()
                            best_month = str(best_month_period)
                            best_month_revenue = float(monthly_revenue.max())
                            
                            # Monthly sales chart data
                            monthly_sales_data = {
                                "labels": [str(period) for period in monthly_revenue.index],
                                "datasets": [{
                                    "label": "Monthly Revenue",
                                    "data": [float(val) for val in monthly_revenue.values],
                                    "backgroundColor": "#36A2EB",
                                    "borderColor": "#36A2EB",
                                    "borderWidth": 2
                                }]
                            }
                    except Exception:
                        pass

                # Product Performance
                if product_col:
                    product_stats = df.groupby(product_col).agg({
                        'Revenue': 'sum',
                        qty_col: 'sum'
                    }).reset_index()
                    product_stats = product_stats.rename(columns={product_col: 'product_name', qty_col: 'units_sold'})
                    product_stats = product_stats.sort_values('Revenue', ascending=False).head(10)
                    
                    product_performance = [
                        {
                            'product': str(row['product_name']),
                            'revenue': float(row['Revenue']),
                            'units_sold': int(row['units_sold']),
                            'performance': 'High' if row['Revenue'] > product_stats['Revenue'].quantile(0.7) else 
                                          'Medium' if row['Revenue'] > product_stats['Revenue'].quantile(0.3) else 'Low'
                        }
                        for _, row in product_stats.iterrows()
                    ]

                # Category Performance
                if category_col:
                    category_stats = df.groupby(category_col)['Revenue'].sum().sort_values(ascending=False)
                    category_performance = [
                        {
                            'category': str(cat),
                            'revenue': float(revenue),
                            'performance': 'High' if revenue > category_stats.quantile(0.7) else 
                                         'Medium' if revenue > category_stats.quantile(0.3) else 'Low'
                        }
                        for cat, revenue in category_stats.items()
                    ]

                # Expected Growth (from forecast if available)
                # Try to get forecast data from session or calculate simple trend
                if date_col:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.dropna(subset=[date_col])
                        df['YearMonth'] = df[date_col].dt.to_period('M')
                        monthly_revenue = df.groupby('YearMonth')['Revenue'].sum().sort_index()
                        
                        if len(monthly_revenue) >= 2:
                            recent = monthly_revenue.tail(2)
                            if len(recent) == 2:
                                prev = float(recent.iloc[0])
                                curr = float(recent.iloc[1])
                                if prev > 0:
                                    expected_growth = ((curr - prev) / prev) * 100
                    except Exception:
                        pass

    except Exception as e:
        messages.error(request, f"Error generating reports: {str(e)}")

    context = {
        'filename': filename,
        'has_data': has_data,
        'total_sales': round(total_sales, 2),
        'avg_monthly_sales': round(avg_monthly_sales, 2),
        'best_month': best_month,
        'best_month_revenue': round(best_month_revenue, 2),
        'expected_growth': round(expected_growth, 1),
        'product_performance': product_performance,
        'category_performance': category_performance,
        'monthly_sales_data': json.dumps(monthly_sales_data),
    }

    return render(request, 'reports.html', context)


# -------------------------------------------------
# 10. FORECASTS PAGE (/forecasts/) - DATA-DRIVEN
# -------------------------------------------------
@login_required
def forecasts(request):
    """
    Builds a time-series forecast using Linear Regression on aggregated data (M/Q/Y).
    Includes logic to fix 'Current Month' using latest dataset date.
    """
    file_path = request.session.get('data_path')
    filename = request.session.get('filename', 'No file uploaded')
    period = request.GET.get('period', 'M')  # M, Q, Y

    # Defaults
    forecasted_revenue = 0
    current_revenue = 0
    growth_rate = 0.0
    growth_class = 'success'
    growth_trend = 'Stable'
    peak_period_label = "N/A"
    peak_revenue = 0
    forecast_data = {"datasets": []}
    metrics = {}
    
    # Map period codes to readable names for UI
    period_names = {'M': 'Month', 'Q': 'Quarter', 'Y': 'Year'}
    period_name = period_names.get(period, 'Month')
    
    # Filter selections
    selected_product = request.GET.get('product')
    selected_region = request.GET.get('region')
    selected_category = request.GET.get('category')

    try:
        if not file_path or not os.path.exists(file_path):
            messages.info(request, "Upload a dataset first to see forecasts.")
        else:
            # 1. Load Data
            # 1. Load Data
            df = read_data_file(file_path)

            inferred = infer_core_columns(df)
            date_col = inferred.get("date")
            qty_col = inferred.get("quantity")
            price_col = inferred.get("price")
            
            # If price is missing but we have Derived_Price (from normalization), use that
            if not price_col and 'Derived_Price' in df.columns:
                price_col = 'Derived_Price'
            
            # Optional columns for filtering
            product_col = inferred.get('product')
            region_col = inferred.get('region')
            category_col = inferred.get('category')
            
            # Get unique values for filters
            products = sorted(df[product_col].dropna().unique().astype(str).tolist()) if product_col else []
            regions = sorted(df[region_col].dropna().unique().astype(str).tolist()) if region_col else []
            categories = sorted(df[category_col].dropna().unique().astype(str).tolist()) if category_col else []
            
            if not date_col or not qty_col or not price_col:
                messages.warning(request, "Forecasting requires Date, Quantity, and Price columns.")
            else:
                # 2. Preprocess & Filter
                
                # Apply Filters
                if selected_product and product_col:
                    df = df[df[product_col].astype(str) == selected_product]
                if selected_region and region_col:
                    df = df[df[region_col].astype(str) == selected_region]
                if selected_category and category_col:
                    df = df[df[category_col].astype(str) == selected_category]
                
                df['Revenue'] = df[qty_col].astype(float) * df[price_col].astype(float)
                
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])
                
                if df.empty:
                    messages.warning(request, "No data available for the selected filters.")
                else:
                    # 3. Aggregation (M/Q/Y)
                    df = df.sort_values(date_col)
                    
                    idx_col = date_col
                    val_col = 'Revenue'
                    
                    resample_rule = 'MS' if period == 'M' else 'QS' if period == 'Q' else 'YS'
                    
                    agg_df = df.set_index(idx_col).resample(resample_rule)[val_col].sum().reset_index()
                    agg_df = agg_df.rename(columns={idx_col: 'period_dt', val_col: 'y'})
                    
                    # Generate labels early 
                    if period == 'M':
                         agg_df['label'] = agg_df['period_dt'].dt.strftime('%b %Y')
                    elif period == 'Q':
                         agg_df['label'] = agg_df['period_dt'].apply(lambda x: f"Q{(x.month-1)//3 + 1} {x.year}")
                    elif period == 'Y':
                         agg_df['label'] = agg_df['period_dt'].dt.strftime('%Y')

                    # 4. Linear Regression Forecast
                    if len(agg_df) >= 2:
                        # Returns 3 values now: history, future, metrics
                        history_df, future_df, metrics = linear_regression_forecast(agg_df, periods_to_forecast=3, freq=period)
                        
                        # Handle Export Request
                        if request.GET.get('export') == 'csv':
                            import csv
                            from django.http import HttpResponse
                            
                            response = HttpResponse(content_type='text/csv')
                            response['Content-Disposition'] = f'attachment; filename="forecast_report_{period}.csv"'
                            
                            writer = csv.writer(response)
                            writer.writerow(['Date', 'Actual Revenue', 'Forecast Revenue', 'Lower CI', 'Upper CI', 'Trend'])
                            
                            # Write history
                            for _, row in history_df.iterrows():
                                writer.writerow([
                                    row['period_dt'].date(), 
                                    round(row['y'], 2), 
                                    '', # forecast
                                    round(row.get('lower_ci', 0), 2),
                                    round(row.get('upper_ci', 0), 2),
                                    round(row.get('trend', 0), 2)
                                ])
                                
                            # Write future
                            for _, row in future_df.iterrows():
                                writer.writerow([
                                    row['period_dt'].date(), 
                                    '', # actual
                                    round(row['yhat'], 2),
                                    round(row.get('lower_ci', 0), 2),
                                    round(row.get('upper_ci', 0), 2),
                                    '' # trend
                                ])
                                
                            return response

                        # 5. KPIs
                        if not history_df.empty:
                            last_actual = history_df.iloc[-1]
                            current_revenue = float(last_actual['y'])
                            
                        forecasted_revenue = float(future_df['yhat'].sum())
                        
                        if current_revenue > 0:
                            avg_forecast = future_df['yhat'].mean()
                            growth_rate = ((avg_forecast - current_revenue) / current_revenue) * 100
                            
                        growth_class = 'success' if growth_rate >= 0 else 'danger'
                        if growth_rate > 5:
                            growth_trend = "Growth"
                        elif growth_rate < -5:
                            growth_trend = "Decline"
                        else:
                            growth_trend = "Stable"

                        combined_df = pd.concat([
                            history_df[['period_dt', 'y', 'label']], 
                            future_df[['period_dt', 'yhat', 'label']].rename(columns={'yhat': 'y'})
                        ])
                        combined_df = combined_df.reset_index(drop=True)

                        if not combined_df.empty:
                            peak_row_idx = combined_df['y'].idxmax()
                            peak_row = combined_df.loc[peak_row_idx]
                            peak_period_label = peak_row['label']
                            peak_revenue = float(peak_row['y'])

                        # 6. Chart Data Preparation
                        labels = history_df['label'].tolist() + future_df['label'].tolist()
                        
                        actual_data = history_df['y'].tolist() + [None] * len(future_df)
                        
                        # Use last actual as bridge for forecast line
                        last_actual_val = history_df.iloc[-1]['y']
                        forecast_data_points = [None] * (len(history_df) - 1) + [last_actual_val] + future_df['yhat'].tolist()
                        
                        # Confidence Intervals (Bowtie)
                        # We need [None... None, LastActual] + [UpperCI...]
                        # But history also has CI now. We can show it for history too if we want "Trend Corridor"
                        # For simplicity, let's show CI for Future only, starting from last actual (width 0)
                        
                        upper_ci_data = [None] * (len(history_df) - 1) + [last_actual_val] + future_df['upper_ci'].tolist()
                        lower_ci_data = [None] * (len(history_df) - 1) + [last_actual_val] + future_df['lower_ci'].tolist()
                        
                        datasets = [
                            {
                                "label": f"Actual Revenue",
                                "data": actual_data,
                                "borderColor": "#0d6efd",
                                "backgroundColor": "rgba(13, 110, 253, 0.1)",
                                "borderWidth": 2,
                                "fill": False,
                                "tension": 0.1,
                                "order": 1
                            },
                            {
                                "label": "Forecast",
                                "data": forecast_data_points,
                                "borderColor": "#adb5bd",
                                "borderWidth": 2,
                                "borderDash": [5, 5],
                                "fill": False,
                                "tension": 0.1,
                                "order": 2
                            },
                             {
                                "label": "Upper Confidence",
                                "data": upper_ci_data,
                                "borderColor": "rgba(108, 117, 125, 0)", # Invisible border
                                "backgroundColor": "rgba(108, 117, 125, 0.2)",
                                "fill": "+1", # Fill to next dataset (Lower CI) 
                                "pointRadius": 0,
                                "order": 3
                            },
                            {
                                "label": "Lower Confidence",
                                "data": lower_ci_data,
                                "borderColor": "rgba(108, 117, 125, 0)",
                                "backgroundColor": "rgba(108, 117, 125, 0.2)",
                                "fill": False,
                                "pointRadius": 0,
                                "order": 4
                            }
                        ]
                        
                        forecast_data = {
                            "labels": labels,
                            "datasets": datasets
                        }
                    else:
                        messages.warning(request, "Not enough data points for forecasting (need at least 2).")

    except Exception as e:
        messages.error(request, f"Error generating forecast: {str(e)}")
        import traceback
        print(traceback.format_exc())

    context = {
        'filename': filename,
        'forecasted_revenue': round(forecasted_revenue, 0),
        'current_revenue': round(current_revenue, 0),
        'growth_rate': round(growth_rate, 1),
        'growth_class': growth_class,
        'growth_trend': growth_trend,
        'peak_month': peak_period_label,
        'peak_revenue': round(peak_revenue, 0),
        'forecast_data': json.dumps(forecast_data),
        'selected_period': period,
        'period_name': period_name,
        # Filter Context
        'products': vars().get('products', []),
        'regions': vars().get('regions', []),
        'categories': vars().get('categories', []),
        'selected_product': selected_product,
        'selected_region': selected_region,
        'selected_category': selected_category,
        # Metrics
        'metrics': metrics
    }

    return render(request, 'forecasts.html', context)

# -------------------------------------------------
# REGIONS PAGE (/regions/) - DATA-DRIVEN
# -------------------------------------------------
@login_required
def regions(request):
    """
    Region-level insights built from the uploaded dataset.
    """
    file_path = request.session.get('data_path')
    filename = request.session.get('filename', 'No file uploaded')

    total_regions = 0
    total_region_revenue = 0
    avg_region_value = 0
    top_region = "N/A"
    regions_list = []
    region_chart_data = {"labels": [], "datasets": []}

    try:
        if not file_path or not os.path.exists(file_path):
            messages.info(request, "Upload a dataset first to see region insights.")
        else:
            df = read_data_file(file_path)

            # Use shared robust column inference
            inferred = infer_core_columns(df)
            region_col = inferred.get('region')
            qty_col = inferred.get('quantity')
            price_col = inferred.get('price')
            
            # If price is missing but we have Derived_Price (from normalization), use that
            if not price_col and 'Derived_Price' in df.columns:
                price_col = 'Derived_Price'

            if not region_col or not qty_col or not price_col:
                messages.warning(
                    request,
                    "Region insights require Region, Quantity and Price columns in your file. "
                    f"(Found: {', '.join(df.columns)})"
                )
            else:
                df['Revenue'] = df[qty_col] * df[price_col]

                region_stats = (
                    df.groupby(region_col)
                    .agg(revenue=('Revenue', 'sum'), orders=(qty_col, 'sum'))
                    .reset_index()
                    .rename(columns={region_col: 'region'})
                )

                total_regions = int(region_stats['region'].nunique())
                total_region_revenue = float(region_stats['revenue'].sum())
                avg_region_value = float(region_stats['revenue'].mean()) if total_regions > 0 else 0

                if not region_stats.empty:
                    top_row = region_stats.sort_values('revenue', ascending=False).iloc[0]
                    top_region = str(top_row['region'])

                regions_list = [
                    {
                        "region": str(row['region']),
                        "revenue": float(row['revenue']),
                        "orders": int(row['orders']),
                    }
                    for _, row in region_stats.iterrows()
                ]

                region_chart_data = {
                    "labels": [r['region'] for r in regions_list],
                    "datasets": [{
                        "label": "Revenue by Region",
                        "data": [r['revenue'] for r in regions_list],
                        "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"][:len(regions_list)] or ["#36A2EB"],
                    }],
                }

    except Exception as e:
        messages.error(request, f"Error building region insights: {str(e)}")

    context = {
        'filename': filename,
        'total_regions': total_regions,
        'total_region_revenue': round(total_region_revenue, 0),
        'avg_region_value': round(avg_region_value, 0),
        'top_region': top_region,
        'regions_list': regions_list,
        'region_chart_data': json.dumps(region_chart_data),
    }

    return render(request, 'region.html', context)


@login_required
def view_dataset(request, dataset_id):
    """
    View the raw data + Health Check.
    - loads full df to check for duplicates/nulls
    - renders first 500 rows for the table
    """
    try:
        dataset = Dataset.objects.get(id=dataset_id, user=request.user)
        try:
            file_path = dataset.file.path
        except (ValueError, AttributeError):
            file_path = os.path.join(settings.MEDIA_ROOT, dataset.file.name)
        
        if not os.path.exists(file_path):
            messages.error(request, "File not found.")
            return redirect('analyzer:upload')

        # Load Data
        # Load Data
        df = read_data_file(file_path)

        # Check for cleaning request
        cleaning_applied = request.GET.get('clean') == 'true'
        rows_dropped = 0
        original_row_count = len(df)

        if cleaning_applied:
            # Clean data: drop duplicates and rows with missing values
            # Use advanced cleaning logic
            df_clean, report_log = clean_dataset(df)
            rows_dropped = original_row_count - len(df_clean)
            df = df_clean

            # Handle Save Action
            if request.method == 'POST' and request.POST.get('action') == 'save_cleaned':
                # Save the cleaned dataframe back to file
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                else:
                    df.to_excel(file_path, index=False)
                
                messages.success(request, "Dataset cleaned and saved successfully!")
                return redirect('analyzer:view_dataset', dataset_id=dataset_id)
        else:
            report_log = []

        # Filter Logic (Server-Side)
        columns = df.columns.tolist()
        filter_col = request.GET.get('filter_col')
        filter_val = request.GET.get('filter_val', '').strip()

        if filter_col and filter_val and filter_col in columns:
            # Apply case-insensitive string containment filter
            # coerced to string to handle numeric cols gracefully
            df = df[df[filter_col].astype(str).str.contains(filter_val, case=False, na=False)]

        # Health Checks (on the potentially filtered/cleaned data? 
        # Usually health checks are on the WHOLE dataset, but filter narrows the view.
        # Let's keep health checks on the loaded df (before filtering? or after?)
        # User wants to filter the data. The Table should show filtered data.
        # Health indicators usually show the state of the *file*. 
        # Let's keep health checks based on the `df` state *after cleaning* but *before filtering* 
        # so users know if the underlying data is clean, even if they are looking at a subset.
        # However, to avoid re-reading or complex copying, let's calculate health on the 'df' 
        # which is currently cleaned. 
        # actually, if I filter, 'df' changes. I should preserve a 'df_health' or calculate before filtering.
        
        # Let's calculate health metrics on the dataset *as it is being viewed* (cleaned or valid).
        # But if I filter for "City=NewYork", "Total Rows" should probably say how many match.
        
        total_rows = len(df)
        # Recalculate health for the filtered view? 
        # No, 'Missing Values' in a filtered view might be confusing. 
        # Let's stick to: Health = File Health (Clean/Dirty). Dimensions = Filtered count.
        
        # Refactoring slightly to ensure health is about the 'scope'
        # But for simplicity in this view, let's just let 'df' flow through.
        
        total_rows = len(df)
        total_cols = len(df.columns) # cols don't change
        missing_values = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        is_clean = (missing_values == 0) and (duplicate_rows == 0)

        # Prepare context data
        # Truncate for display performance logic
        df_display = df.head(500)
        
        html_table = df_display.to_html(
            classes='table table-striped table-hover table-bordered table-sm',
            index=False,
            na_rep='-'
        )
        
        row_count_display = len(df_display)
        total_rows_estimate = f"{total_rows}" if total_rows > 500 else total_rows

        context = {
            'dataset': dataset,
            'html_table': html_table,
            'row_count': row_count_display,
            'total_rows': total_rows,
            'total_cols': total_cols,
            'missing_values': missing_values,
            'duplicate_rows': duplicate_rows,
            'is_clean': is_clean,
            'cleaning_applied': cleaning_applied,
            'rows_dropped': rows_dropped,
            'columns': columns,
            'filter_col': filter_col,
            'filter_val': filter_val,
            'report_log': report_log,
        }
        return render(request, 'view_dataset.html', context)

    except Dataset.DoesNotExist:
        messages.error(request, "Dataset not found.")
        return redirect('analyzer:upload')
    except Exception as e:
        messages.error(request, f"Error viewing dataset: {str(e)}")
        return redirect('analyzer:upload')


# -------------------------------------------------
# 11. TEST VIEW
# -------------------------------------------------

# -------------------------------------------------
# 13. CREATE CUSTOM DASHBOARD
# -------------------------------------------------
@login_required
def create_dashboard(request):
    """
    View for creating custom charts from the dataset.
    Handles both page rendering and AJAX requests for chart data.
    """
    try:
        # Load Active Dataset (reuse helper)
        df, filename, file_path, dataset_id = get_or_load_active_dataset(request)
        
        # AJAX Request for Chart Data
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            if df is None:
                return JsonResponse({'error': 'No dataset loaded.'}, status=400)

            try:
                chart_type = request.GET.get('chart_type', 'bar')
                x_axis = request.GET.get('x_axis')
                y_axis = request.GET.get('y_axis')
                aggregation = request.GET.get('aggregation', 'sum')

                if not x_axis or not y_axis:
                    return JsonResponse({'error': 'Missing X or Y axis selection.'}, status=400)
                
                if x_axis not in df.columns or y_axis not in df.columns:
                     return JsonResponse({'error': 'Selected columns not found in dataset.'}, status=400)

                # Ensure Y-axis is numeric
                df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce').fillna(0)

                limit = request.GET.get('limit', '10')
                filter_op = request.GET.get('filter_op')
                try:
                    filter_value = float(request.GET.get('filter_value', 0))
                except (ValueError, TypeError):
                    filter_value = 0

                # Data Aggregation
                if aggregation == 'sum':
                    grouped = df.groupby(x_axis)[y_axis].sum()
                elif aggregation == 'avg':
                    grouped = df.groupby(x_axis)[y_axis].mean()
                elif aggregation == 'count':
                    grouped = df.groupby(x_axis)[y_axis].count()
                elif aggregation == 'min':
                    grouped = df.groupby(x_axis)[y_axis].min()
                elif aggregation == 'max':
                    grouped = df.groupby(x_axis)[y_axis].max()
                else:
                    return JsonResponse({'error': 'Invalid aggregation type.'}, status=400)
                
                # Apply Value Filters (Having clause equivalent)
                if filter_op == 'gt':
                    grouped = grouped[grouped > filter_value]
                elif filter_op == 'lt':
                    grouped = grouped[grouped < filter_value]
                elif filter_op == 'gte':
                    grouped = grouped[grouped >= filter_value]
                elif filter_op == 'lte':
                    grouped = grouped[grouped <= filter_value]

                # Sort values
                # Default to descending for bar/pie to show top performers
                if chart_type in ['bar', 'pie', 'doughnut']:
                    grouped = grouped.sort_values(ascending=False)
                elif chart_type == 'line':
                    # For line charts (often time series), sort by index (X-axis)
                    grouped = grouped.sort_index()
                
                # Apply Limit (Top N)
                if limit != 'all' and chart_type != 'line':
                    try:
                        limit_n = int(limit)
                        grouped = grouped.head(limit_n)
                    except ValueError:
                        pass # Ignore if invalid limit

                data = {
                    'labels': grouped.index.astype(str).tolist(),
                    'values': grouped.values.tolist(),
                    'chart_type': chart_type
                }
                return JsonResponse(data)

            except Exception as e:
                import traceback
                print(traceback.format_exc())
                return JsonResponse({'error': str(e)}, status=500)

        # Normal Page Load
        context = {
            'has_data': df is not None,
            'filename': filename,
            'columns': [],
            'numeric_columns': [],
            'date_col': ''
        }

        if df is not None:
            # Separate columns by type for dropdowns
            context['columns'] = sorted(df.columns.tolist())
            context['numeric_columns'] = sorted(df.select_dtypes(include=['number']).columns.tolist())
            
            # Try to identify potential default columns
            inferred = infer_core_columns(df)
            context['date_col'] = inferred.get('date', '')

            # Handle Edit Mode
            edit_id = request.GET.get('edit_id')
            if edit_id:
                try:
                    saved_chart = SavedChart.objects.get(id=edit_id, user=request.user)
                    context.update({
                        'saved_chart_id': saved_chart.id,
                        'saved_title': saved_chart.title,
                        'selected_chart_type': saved_chart.chart_type,
                        'selected_x_axis': saved_chart.x_axis,
                        'selected_y_axis': saved_chart.y_axis,
                        'selected_aggregation': saved_chart.aggregation,
                        'selected_limit': saved_chart.limit,
                        'selected_filter_op': saved_chart.filter_op,
                        'selected_filter_value': saved_chart.filter_value,
                        'is_editing': True
                    })
                except SavedChart.DoesNotExist:
                    messages.error(request, "Chart not found.")

        return render(request, 'create_dashboard.html', context)
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        messages.error(request, f"Error loading create dashboard: {str(e)}")
        return redirect('analyzer:dashboard')

def test_view(request):
    return render(request, "test.html")

# -------------------------------------------------
# 14. SAVE CHART & MY DASHBOARD
# -------------------------------------------------
from .models import SavedChart

@login_required
def save_chart(request):
    if request.method == 'POST':
        try:
            # Load Active Dataset (reuse helper)
            df, filename, file_path, dataset_id = get_or_load_active_dataset(request)
            
            if not dataset_id:
                return JsonResponse({'error': 'No active dataset found.'}, status=400)
                
            dataset = Dataset.objects.get(id=dataset_id)
            
            # Parse parameters
            chart_id = request.POST.get('chart_id')
            title = request.POST.get('title', 'Untitled Chart')
            chart_type = request.POST.get('chart_type')
            x_axis = request.POST.get('x_axis')
            y_axis = request.POST.get('y_axis')
            aggregation = request.POST.get('aggregation', 'sum')
            limit = request.POST.get('limit', '10')
            filter_op = request.POST.get('filter_op')
            filter_value = request.POST.get('filter_value')
            
            if filter_value:
                try:
                    filter_value = float(filter_value)
                except (ValueError, TypeError):
                    filter_value = None
            else:
                 filter_value = None

            if chart_id:
                # Update existing chart
                try:
                    chart = SavedChart.objects.get(id=chart_id, user=request.user)
                    chart.title = title
                    chart.chart_type = chart_type
                    chart.x_axis = x_axis
                    chart.y_axis = y_axis
                    chart.aggregation = aggregation
                    chart.limit = limit
                    chart.filter_op = filter_op
                    chart.filter_value = filter_value
                    if dataset_id:
                        chart.dataset = dataset
                    chart.save()
                    return JsonResponse({'message': 'Chart updated successfully!', 'id': chart.id})
                except SavedChart.DoesNotExist:
                     return JsonResponse({'error': 'Chart not found for update.'}, status=404)
            else:
                # Create SavedChart
                chart = SavedChart.objects.create(
                    user=request.user,
                    dataset=dataset,
                    title=title,
                    chart_type=chart_type,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    aggregation=aggregation,
                    limit=limit,
                    filter_op=filter_op,
                    filter_value=filter_value
                )
                return JsonResponse({'message': 'Chart saved successfully!', 'id': chart.id})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid method.'}, status=405)

@login_required
def my_dashboard(request):
    """
    Displays all saved charts for the user.
    """
    charts = SavedChart.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'my_dashboard.html', {'charts': charts})

@login_required
def delete_chart(request, chart_id):
    if request.method == 'POST':
        try:
            chart = SavedChart.objects.get(id=chart_id, user=request.user)
            chart.delete()
            messages.success(request, "Chart deleted successfully.")
            return JsonResponse({'message': 'Chart deleted.'})
        except SavedChart.DoesNotExist:
            return JsonResponse({'error': 'Chart not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid method.'}, status=405)
