"""Main Flask application for AI Customer Feedback Analyzer - AMIL Project"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FeedbackAnalyzerApp:
    """Main application class for AI Customer Feedback Analyzer"""

    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'amil-project-2025-rtx-4060'

        # Enable CORS for frontend communication
        CORS(self.app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5000", "http://127.0.0.1:5000"])

        # Initialize components with error handling
        self.initialize_app()

    def initialize_app(self):
        """Initialize all application components with fallbacks"""
        logger.info("🚀 Initializing AI Customer Feedback Analyzer...")

        try:
            # Try to initialize components (with fallbacks if they fail)
            self.initialize_models()
            self.register_blueprints()
            self.register_error_handlers()

            logger.info("🎯 Application initialized successfully!")

        except Exception as e:
            logger.warning(f"⚠️  Some components failed to initialize: {str(e)}")
            logger.info("🔄 Running in fallback mode...")

            # Still register basic routes
            self.register_basic_routes()
            self.register_error_handlers()

    def initialize_models(self):
        """Initialize AI models with error handling"""
        logger.info("🧠 Loading AI models...")

        try:
            # Try to initialize GPU manager
            from utils.gpu_utils import GPUManager
            self.app.gpu_manager = GPUManager()
            logger.info("✅ GPU manager loaded")
        except Exception as e:
            logger.warning(f"⚠️  GPU manager failed: {str(e)}")
            self.app.gpu_manager = None

        try:
            # Try to initialize database
            from utils.database_manager import DatabaseManager
            self.app.db_manager = DatabaseManager()
            logger.info("✅ Database manager loaded")
        except Exception as e:
            logger.warning(f"⚠️  Database manager failed: {str(e)}")
            self.app.db_manager = None

        try:
            # Try to initialize sentiment analyzer
            from models.sentiment_analyzer import SentimentAnalyzer
            self.app.sentiment_analyzer = SentimentAnalyzer()
            logger.info("✅ Sentiment analyzer loaded")
        except Exception as e:
            logger.warning(f"⚠️  Sentiment analyzer failed: {str(e)}")
            self.app.sentiment_analyzer = None

        try:
            # Try to initialize topic modeler
            from models.topic_modeler import TopicModeler
            self.app.topic_modeler = TopicModeler()
            logger.info("✅ Topic modeler loaded")
        except Exception as e:
            logger.warning(f"⚠️  Topic modeler failed: {str(e)}")
            self.app.topic_modeler = None

        try:
            # Try to initialize response generator
            from models.response_generator import ResponseGenerator
            self.app.response_generator = ResponseGenerator()
            logger.info("✅ Response generator loaded")
        except Exception as e:
            logger.warning(f"⚠️  Response generator failed: {str(e)}")
            self.app.response_generator = None

        try:
            # Try to initialize urgency scorer
            from models.urgency_scorer import UrgencyScorer
            self.app.urgency_scorer = UrgencyScorer()
            logger.info("✅ Urgency scorer loaded")
        except Exception as e:
            logger.warning(f"⚠️  Urgency scorer failed: {str(e)}")
            self.app.urgency_scorer = None

    def register_blueprints(self):
        """Register API blueprints"""
        try:
            from api.routes import api_bp
            self.app.register_blueprint(api_bp, url_prefix='/api')
            logger.info("✅ API routes registered")
        except Exception as e:
            logger.error(f"❌ Failed to register API routes: {str(e)}")
            # Register minimal API routes
            self.register_basic_routes()

    def register_basic_routes(self):
        """Register basic routes as fallback"""

        @self.app.route('/api/system/health')
        def basic_health():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'mode': 'fallback'
            })

        @self.app.route('/api/analyze/sentiment', methods=['POST'])
        def basic_sentiment():
            data = request.get_json()
            return jsonify({
                'sentiment': {
                    'label': 'neutral',
                    'score': 0.5,
                    'note': 'Fallback mode - basic analysis'
                },
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            })

    def register_error_handlers(self):
        """Register error handlers"""

        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Endpoint not found',
                'message': 'The requested API endpoint does not exist'
            }), 404

        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {str(error)}")
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred'
            }), 500

        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({
                'error': 'Bad request',
                'message': 'Invalid request data'
            }), 400

    @property
    def application(self):
        """Get the Flask application instance"""
        return self.app

# Create application instance
app_instance = FeedbackAnalyzerApp()
app = app_instance.application

# Add health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'AMIL Project is running!'
    })

# Serve frontend files
@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    try:
        return send_from_directory('frontend', 'index.html')
    except Exception as e:
        return jsonify({
            'message': 'AMIL Project Backend is running!',
            'frontend': 'Place your index.html in the frontend/ directory',
            'api_health': '/health',
            'error': str(e)
        })

@app.route('/<path:path>')
def serve_static(path):
    """Serve static frontend files"""
    try:
        return send_from_directory('frontend', path)
    except:
        return serve_frontend()

if __name__ == '__main__':
    logger.info("🌟 Starting AMIL Project - AI Customer Feedback Analyzer")
    logger.info(f"🔧 Running on: localhost:5000")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
