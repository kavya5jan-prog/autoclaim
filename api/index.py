"""
Vercel serverless function wrapper for Flask app
This file exports the Flask app for Vercel's Python runtime
"""
import sys
import os

# Add parent directory to path to import app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Set Vercel environment variable
os.environ['VERCEL'] = '1'

# Import the Flask app
try:
    from app import app
except ImportError as e:
    # Fallback: try importing directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", os.path.join(parent_dir, "app.py"))
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    app = app_module.app

# Vercel expects the app to be exported
# The @vercel/python builder will automatically handle WSGI
__all__ = ['app']
