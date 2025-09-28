"""Server runner for AI Customer Feedback Analyzer - AMIL Project"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TRANSFORMERS_CACHE'] = str(project_root / 'models' / 'transformers_cache')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main server runner"""
    logger = logging.getLogger(__name__)

    try:
        logger.info("🚀 Starting AMIL Project - AI Customer Feedback Analyzer")
        logger.info(f"📁 Project root: {project_root}")

        # Import and run the main application
        from backend.main import app

        # Run the Flask application
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production
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
