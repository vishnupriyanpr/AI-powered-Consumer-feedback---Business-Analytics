"""Server runner for AI Customer Feedback Analyzer - AMIL Project"""

import sys
import os
import logging
from pathlib import Path
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

def main():
    logger = logging.getLogger(__name__)

    try:
        logger.info("🚀 Starting AMIL Project - AI Customer Feedback Analyzer")
        logger.info(f"📁 Project root: {project_root}")

        # Register serving routes
        @app.route('/')
        def serve_frontend():
            frontend_path = os.path.join(project_root, 'frontend', 'index.html')
            logger.info(f"Trying to serve: {frontend_path}")
            if os.path.exists(frontend_path):
                return send_from_directory(os.path.join(project_root, 'frontend'), 'index.html')
            else:
                return jsonify({
                    'message': 'AMIL Project Backend is running!',
                    'frontend': 'frontend/index.html not found - create it!',
                    'api_health': '/health'
                })

        @app.route('/<path:path>')
        def serve_static(path):
            static_path = os.path.join(project_root, 'frontend', path)
            if os.path.exists(static_path):
                return send_from_directory(os.path.join(project_root, 'frontend'), path)
            return serve_frontend()

        @app.route('/health')
        def health():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'message': 'AIML Project is running!'
            })

        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True,
            use_reloader=False
        )

    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
