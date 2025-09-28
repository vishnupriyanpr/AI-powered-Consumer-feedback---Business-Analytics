"""API package for AI Customer Feedback Analyzer - AMIL Project"""

from .routes import api_bp
from .middleware import setup_middleware

__all__ = ['api_bp', 'setup_middleware']
