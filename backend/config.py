"""API routes for AI Customer Feedback Analyzer - AMIL Project"""

from flask import Blueprint, request, jsonify, current_app
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
import traceback

logger = logging.getLogger(__name__)

# Create API blueprint
api_bp = Blueprint('api', __name__)

def validate_request_data(required_fields: List[str]) -> Dict[str, Any]:
    """Validate incoming request data"""
    if not request.is_json:
        raise ValueError("Request must be JSON")

    data = request.get_json()
    if not data:
        raise ValueError("No data provided")

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    return data

def handle_api_error(func):
    """Decorator to handle API errors consistently"""
    def wrapper(*args, **kwargs):
        try:
            start_time = time.time()
            result = func(*args, **kwargs)

            # Add timing information
            if isinstance(result, tuple) and len(result) == 2:
                response_data, status_code = result
            else:
                response_data, status_code = result, 200

            if isinstance(response_data, dict):
                response_data['processing_time'] = round((time.time() - start_time) * 1000, 2)

            return jsonify(response_data), status_code

        except ValueError as e:
            logger.warning(f"Validation error in {func.__name__}: {str(e)}")
            return jsonify({
                'error': 'Validation Error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }), 400

        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred',
                'timestamp': datetime.now().isoformat()
            }), 500

    wrapper.__name__ = func.__name__
    return wrapper

@api_bp.route('/analyze/sentiment', methods=['POST'])
@handle_api_error
def analyze_sentiment():
    """Analyze sentiment of feedback text"""
    data = validate_request_data(['text'])

    text = data['text'].strip()
    if len(text) < 10:
        raise ValueError("Text must be at least 10 characters long")

    # Get sentiment analysis
    sentiment_result = current_app.sentiment_analyzer.analyze(text)

    # Store in database
    current_app.db_manager.store_analysis(
        text=text,
        analysis_type='sentiment',
        result=sentiment_result,
        language=data.get('language', 'auto')
    )

    return {
        'sentiment': sentiment_result,
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }

@api_bp.route('/analyze/themes', methods=['POST'])
@handle_api_error
def analyze_themes():
    """Extract themes and topics from feedback"""
    data = validate_request_data(['text'])

    text = data['text'].strip()
    if len(text) < 10:
        raise ValueError("Text must be at least 10 characters long")

    # Get theme analysis
    themes_result = current_app.topic_modeler.extract_themes(text)

    # Store in database
    current_app.db_manager.store_analysis(
        text=text,
        analysis_type='themes',
        result=themes_result,
        language=data.get('language', 'auto')
    )

    return {
        'themes': themes_result,
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }

@api_bp.route('/analyze/urgency', methods=['POST'])
@handle_api_error
def analyze_urgency():
    """Calculate urgency score for feedback"""
    data = validate_request_data(['text'])

    text = data['text'].strip()

    # Get urgency analysis
    urgency_result = current_app.urgency_scorer.calculate_urgency(text)

    # Store in database
    current_app.db_manager.store_analysis(
        text=text,
        analysis_type='urgency',
        result=urgency_result,
        language=data.get('language', 'auto')
    )

    return {
        'urgency': urgency_result,
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }

@api_bp.route('/generate/response', methods=['POST'])
@handle_api_error
def generate_response():
    """Generate automated response to feedback"""
    data = validate_request_data(['text'])

    text = data['text'].strip()
    tone = data.get('tone', 'professional')  # professional, casual

    # Get sentiment first for context
    sentiment = current_app.sentiment_analyzer.analyze(text)

    # Generate response
    response_result = current_app.response_generator.generate_response(
        text,
        sentiment=sentiment['label'],
        tone=tone
    )

    # Store in database
    current_app.db_manager.store_analysis(
        text=text,
        analysis_type='response',
        result=response_result,
        language=data.get('language', 'auto')
    )

    return {
        'response': response_result,
        'context': {'sentiment': sentiment['label'], 'tone': tone},
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }

@api_bp.route('/generate/actions', methods=['POST'])
@handle_api_error
def generate_gdg_actions():
    """Generate GDG-specific action items based on analysis"""
    data = validate_request_data(['sentiment', 'themes', 'urgency'])

    sentiment_data = data['sentiment']
    themes_data = data['themes']
    urgency_data = data['urgency']

    # Generate actions based on analysis
    actions = []

    # High urgency items get immediate attention
    if urgency_data.get('score', 0) > 0.8:
        actions.append({
            'title': 'Immediate Follow-up Required',
            'description': 'High urgency feedback detected - requires immediate response and action',
            'priority': 'High',
            'timeline': 'Within 24 hours',
            'category': 'urgent_response'
        })

    # Negative sentiment with specific themes
    if sentiment_data.get('label') == 'negative':
        themes = themes_data.get('topics', [])

        for theme in themes:
            theme_lower = theme.lower()

            if any(keyword in theme_lower for keyword in ['technical', 'bug', 'error', 'broken']):
                actions.append({
                    'title': 'Technical Issue Resolution',
                    'description': f'Create troubleshooting guide for {theme} issues',
                    'priority': 'High',
                    'timeline': 'Within 1 week',
                    'category': 'technical'
                })

            elif any(keyword in theme_lower for keyword in ['content', 'presentation', 'speaker']):
                actions.append({
                    'title': 'Content Quality Improvement',
                    'description': f'Review and improve {theme} quality standards',
                    'priority': 'Medium',
                    'timeline': 'Next event cycle',
                    'category': 'content'
                })

            elif any(keyword in theme_lower for keyword in ['venue', 'logistics', 'wifi']):
                actions.append({
                    'title': 'Logistics Enhancement',
                    'description': f'Address {theme} concerns for future events',
                    'priority': 'High',
                    'timeline': 'Before next event',
                    'category': 'logistics'
                })

    # Positive feedback actions
    elif sentiment_data.get('label') == 'positive':
        actions.append({
            'title': 'Success Pattern Documentation',
            'description': 'Document what worked well to replicate in future events',
            'priority': 'Low',
            'timeline': 'End of month',
            'category': 'improvement'
        })

    # Default action if no specific actions generated
    if not actions:
        actions.append({
            'title': 'General Feedback Review',
            'description': 'Review feedback for general insights and improvements',
            'priority': 'Low',
            'timeline': 'Monthly review',
            'category': 'general'
        })

    # Store actions in database
    current_app.db_manager.store_analysis(
        text=f"Generated actions for feedback analysis",
        analysis_type='actions',
        result={'actions': actions},
        language='en'
    )

    return {
        'actions': actions,
        'generated_count': len(actions),
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }

@api_bp.route('/analytics/history', methods=['GET'])
@handle_api_error
def get_analytics_history():
    """Get historical analytics data"""
    limit = request.args.get('limit', 100, type=int)
    analysis_type = request.args.get('type', None)

    history = current_app.db_manager.get_analysis_history(
        limit=limit,
        analysis_type=analysis_type
    )

    return {
        'history': history,
        'count': len(history),
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }

@api_bp.route('/analytics/summary', methods=['GET'])
@handle_api_error
def get_analytics_summary():
    """Get summary analytics"""
    days = request.args.get('days', 30, type=int)

    summary = current_app.db_manager.get_analytics_summary(days=days)

    return {
        'summary': summary,
        'period_days': days,
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }

@api_bp.route('/system/gpu-status', methods=['GET'])
@handle_api_error
def get_gpu_status():
    """Get GPU status information"""
    gpu_info = current_app.gpu_manager.get_device_info()

    return {
        'gpu_available': current_app.gpu_manager.is_gpu_available(),
        'gpu_name': gpu_info.get('name', 'N/A'),
        'gpu_memory': gpu_info.get('memory', 'N/A'),
        'device': str(current_app.gpu_manager.device),
        'cuda_version': gpu_info.get('cuda_version', 'N/A'),
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }

@api_bp.route('/system/health', methods=['GET'])
@handle_api_error
def get_system_health():
    """Get system health status"""
    health_data = {
        'database': 'healthy',
        'models': 'loaded',
        'gpu': 'available' if current_app.gpu_manager.is_gpu_available() else 'unavailable',
        'memory_usage': current_app.gpu_manager.get_memory_usage(),
        'uptime': 'active',
        'timestamp': datetime.now().isoformat()
    }

    return {
        'health': health_data,
        'status': 'success'
    }

@api_bp.route('/batch/analyze', methods=['POST'])
@handle_api_error
def batch_analyze():
    """Analyze multiple feedback items in batch"""
    data = validate_request_data(['feedback_items'])

    feedback_items = data['feedback_items']
    if not isinstance(feedback_items, list):
        raise ValueError("feedback_items must be a list")

    if len(feedback_items) > 50:  # Limit batch size
        raise ValueError("Maximum 50 items per batch")

    results = []

    for i, item in enumerate(feedback_items):
        if not isinstance(item, dict) or 'text' not in item:
            continue

        text = item['text'].strip()
        if len(text) < 10:
            continue

        try:
            # Perform all analyses
            sentiment = current_app.sentiment_analyzer.analyze(text)
            themes = current_app.topic_modeler.extract_themes(text)
            urgency = current_app.urgency_scorer.calculate_urgency(text)

            result = {
                'id': item.get('id', i),
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': sentiment,
                'themes': themes,
                'urgency': urgency,
                'processed': True
            }

            # Store in database
            current_app.db_manager.store_analysis(
                text=text,
                analysis_type='batch',
                result=result,
                language=item.get('language', 'auto')
            )

        except Exception as e:
            result = {
                'id': item.get('id', i),
                'error': str(e),
                'processed': False
            }

        results.append(result)

    processed_count = sum(1 for r in results if r.get('processed', False))

    return {
        'results': results,
        'total_items': len(feedback_items),
        'processed_count': processed_count,
        'failed_count': len(feedback_items) - processed_count,
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    }

# Error handlers for the blueprint
@api_bp.errorhandler(400)
def handle_bad_request(e):
    return jsonify({
        'error': 'Bad Request',
        'message': 'The request could not be processed',
        'timestamp': datetime.now().isoformat()
    }), 400

@api_bp.errorhandler(429)
def handle_rate_limit(e):
    return jsonify({
        'error': 'Rate Limit Exceeded',
        'message': 'Too many requests. Please try again later.',
        'timestamp': datetime.now().isoformat()
    }), 429
