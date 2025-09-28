"""API Middleware for rate limiting and security - AMIL Project"""

from flask import request, jsonify, g
import time
import logging
from functools import wraps
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

class RateLimiter:
    """Thread-safe rate limiter"""

    def __init__(self):
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()

    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed within rate limit"""
        now = time.time()

        with self.lock:
            # Clean old requests
            while self.requests[key] and self.requests[key][0] < now - window:
                self.requests[key].popleft()

            # Check limit
            if len(self.requests[key]) >= limit:
                return False

            # Add current request
            self.requests[key].append(now)
            return True

# Global rate limiter instance
rate_limiter = RateLimiter()

def setup_middleware(app):
    """Setup middleware for the Flask app"""

    @app.before_request
    def before_request():
        """Execute before each request"""
        g.start_time = time.time()

        # Rate limiting
        if not rate_limiter.is_allowed(
                key=request.remote_addr,
                limit=60,  # 60 requests
                window=60  # per minute
        ):
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests. Please try again later.'
            }), 429

    @app.after_request
    def after_request(response):
        """Execute after each request"""
        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'

        # Add performance headers
        if hasattr(g, 'start_time'):
            duration = round((time.time() - g.start_time) * 1000, 2)
            response.headers['X-Response-Time'] = f'{duration}ms'

        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'

        return response

    logger.info("✅ Middleware configured successfully")

def require_api_key(f):
    """Decorator to require API key (optional for demo)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # For demo purposes, we'll skip API key validation
        # In production, you'd validate an API key here
        return f(*args, **kwargs)
    return decorated_function

def log_request(f):
    """Decorator to log API requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logger.info(f"API Request: {request.method} {request.path} from {request.remote_addr}")
        return f(*args, **kwargs)
    return decorated_function
