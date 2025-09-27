"""Main Flask application for AI Customer Feedback Analyzer - AMIL Project"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import *
from utils.database_manager import DatabaseManager
from utils.gpu_utils import GPUManager
from models.sentiment_analyzer import SentimentAnalyzer
from models.topic_modeler import TopicModeler
from models.response_generator import ResponseGenerator
from models.urgency_scorer import UrgencyScorer
from api.routes import api_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'app.log'),
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
        CORS(self.app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

        # Initialize components
        self.gpu_manager = GPUManager()
        self.db_manager = DatabaseManager()
        self.sentiment_analyzer = None
        self.topic_modeler = None
        self.response_generator = None
        self.urgency_scorer = None

        self.initialize_app()

    def initialize_app(self):
        """Initialize all application components"""
        logger.info("🚀 Initializing AI Customer Feedback Analyzer...")
        logger.info(f"📊 GPU Status: {self.gpu_manager.get_device_info()}")

        try:
            # Initialize database
            self.db_manager.initialize_database()
            logger.info("✅ Database initialized successfully")

            # Initialize AI models
            self.initialize_models()

            # Register blueprints
            self.register_blueprints()

            # Register error handlers
            self.register_error_handlers()

            logger.info("🎯 Application initialized successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize application: {str(e)}")
            raise

    def initialize_models(self):
        """Initialize all AI models with GPU acceleration"""
        logger.info("🧠 Loading AI models...")

        # Initialize models with GPU support
        self.sentiment_analyzer = SentimentAnalyzer(
            device=DEVICE,
            model_name=SENTIMENT_MODEL
        )

        self.topic_modeler = TopicModeler(
            device=DEVICE,
            min_topic_size=3
        )

        self.response_generator = ResponseGenerator(
            device=DEVICE,
            model_name=SUMMARIZATION_MODEL
        )

        self.urgency_scorer = UrgencyScorer(
            high_threshold=URGENCY_THRESHOLD_HIGH,
            medium_threshold=URGENCY_THRESHOLD_MEDIUM
        )

        logger.info("✅ All models loaded successfully")

    def register_blueprints(self):
        """Register API blueprints"""
        self.app.register_blueprint(api_bp, url_prefix='/api')

        # Inject models into app context for API routes
        self.app.sentiment_analyzer = self.sentiment_analyzer
        self.app.topic_modeler = self.topic_modeler
        self.app.response_generator = self.response_generator
        self.app.urgency_scorer = self.urgency_scorer
        self.app.db_manager = self.db_manager
        self.app.gpu_manager = self.gpu_manager

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
        'gpu_available': app.gpu_manager.is_gpu_available(),
        'models_loaded': True
    })

# Serve frontend files
@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static frontend files"""
    try:
        return send_from_directory('../frontend', path)
    except:
        return send_from_directory('../frontend', 'index.html')

if __name__ == '__main__':
    logger.info("🌟 Starting AMIL Project - AI Customer Feedback Analyzer")
    logger.info(f"🔧 Running on: {API_HOST}:{API_PORT}")
    logger.info(f"💻 GPU Device: {DEVICE}")

    app.run(
        host=API_HOST,
        port=API_PORT,
        debug=DEBUG_MODE,
        threaded=True
    )
