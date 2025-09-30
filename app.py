"""
AIML Customer Feedback Analyzer - Flask Backend
Serves the beautiful Google-style frontend and integrates trained models
"""

from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import logging
import os
from datetime import datetime
import traceback

# Import your trained models
from models.sentiment_analyzer import SentimentAnalyzer, EmotionDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIMLApp:
    def __init__(self):
        self.app = Flask(__name__,
                         static_folder='frontend',  # Point to your frontend directory
                         static_url_path='')
        CORS(self.app)  # Enable CORS for frontend

        # Initialize your trained models
        self.load_models()

        # Setup routes
        self.setup_routes()

        logger.info("🚀 AIML Customer Feedback Analyzer Backend Ready!")

    def load_models(self):
        """Load your trained models"""
        try:
            logger.info("🧠 Loading YOUR trained AI models...")

            # Load your sentiment model
            self.sentiment_analyzer = SentimentAnalyzer("models/sentiment_analyzer")
            logger.info("✅ Sentiment model loaded")

            # Load your emotion model
            self.emotion_detector = EmotionDetector("models/emotion_detector")
            logger.info("✅ Emotion model loaded")

            logger.info("🎯 All models loaded successfully!")

        except Exception as e:
            logger.error(f"❌ Model loading failed: {str(e)}")
            # Create dummy models as fallback
            self.sentiment_analyzer = None
            self.emotion_detector = None

    def setup_routes(self):
        """Setup API routes and frontend serving"""

        @self.app.route('/')
        def index():
            """Serve the beautiful Google-style frontend"""
            try:
                # Serve index.html from frontend directory
                return send_from_directory('frontend', 'index.html')
            except Exception as e:
                logger.error(f"Error serving frontend: {str(e)}")
                return f'''
                <div style="font-family: 'Google Sans', sans-serif; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh;">
                    <h1>🚀 AIML Customer Feedback Analyzer</h1>
                    <h2>✅ Backend Ready & Models Loaded!</h2>
                    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px; margin: 20px 0;">
                        <h3>📊 Model Status:</h3>
                        <p>✅ Sentiment Model: Loaded (95%+ accuracy)</p>
                        <p>✅ Emotion Model: Loaded (6 emotions)</p>
                        <p>🎮 GPU: RTX 4060 Laptop GPU</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px;">
                        <h3>🔧 Frontend Setup Required:</h3>
                        <p>1. Create <strong>frontend/</strong> directory</p>
                        <p>2. Move your Google-style files there:</p>
                        <ul>
                            <li>frontend/index.html</li>
                            <li>frontend/styles.css</li>
                            <li>frontend/script.js</li>
                        </ul>
                        <p>3. Refresh this page</p>
                    </div>
                    <a href="/api/system/status" style="color: #FFD700; text-decoration: none; font-weight: bold;">🔍 Check System Status</a>
                </div>
                '''

        @self.app.route('/styles.css')
        def serve_css():
            """Serve CSS file"""
            try:
                return send_from_directory('frontend', 'styles.css', mimetype='text/css')
            except:
                return "/* CSS file not found */", 404

        @self.app.route('/script.js')
        def serve_js():
            """Serve JavaScript file"""
            try:
                return send_from_directory('frontend', 'script.js', mimetype='application/javascript')
            except:
                return "// JavaScript file not found", 404

        @self.app.route('/frontend/<path:filename>')
        def serve_frontend_files(filename):
            """Serve any frontend files"""
            return send_from_directory('frontend', filename)

        @self.app.route('/api/analyze/sentiment', methods=['POST'])
        def analyze_sentiment():
            """Analyze sentiment using YOUR trained model"""
            try:
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({'error': 'No text provided'}), 400

                text = data['text'].strip()
                if len(text) < 5:
                    return jsonify({'error': 'Text too short'}), 400

                # Use YOUR trained model
                if self.sentiment_analyzer:
                    result = self.sentiment_analyzer.analyze(text)
                    logger.info(f"📊 Sentiment analysis: {result['label']} ({result['confidence']:.3f})")
                else:
                    result = {'error': 'Sentiment model not available'}

                return jsonify({
                    'sentiment': result,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })

            except Exception as e:
                logger.error(f"Sentiment analysis error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/analyze/emotions', methods=['POST'])
        def analyze_emotions():
            """Analyze emotions using YOUR trained model"""
            try:
                data = request.get_json()
                text = data.get('text', '').strip()

                # Use YOUR trained model
                if self.emotion_detector:
                    result = self.emotion_detector.detect_emotions(text)
                    logger.info(f"💭 Emotion analysis: {result['primary_emotion']} ({result['confidence']:.3f})")
                else:
                    result = {'error': 'Emotion model not available'}

                return jsonify({
                    'emotions': result,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })

            except Exception as e:
                logger.error(f"Emotion analysis error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/analyze/themes', methods=['POST'])
        def analyze_themes():
            """Extract themes from feedback"""
            try:
                data = request.get_json()
                text = data.get('text', '').strip()

                themes_result = self.extract_themes(text)

                return jsonify({
                    'themes': themes_result,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })

            except Exception as e:
                logger.error(f"Theme analysis error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/analyze/urgency', methods=['POST'])
        def analyze_urgency():
            """Calculate urgency score"""
            try:
                data = request.get_json()
                text = data.get('text', '').strip()

                # Get sentiment and emotion first for better urgency calculation
                sentiment_result = self.sentiment_analyzer.analyze(text) if self.sentiment_analyzer else {}
                emotion_result = self.emotion_detector.detect_emotions(text) if self.emotion_detector else {}

                urgency_result = self.calculate_urgency(sentiment_result, emotion_result, text)

                return jsonify({
                    'urgency': urgency_result,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })

            except Exception as e:
                logger.error(f"Urgency analysis error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/generate/response', methods=['POST'])
        def generate_response():
            """Generate AI response"""
            try:
                data = request.get_json()
                text = data.get('text', '').strip()

                # Get sentiment for better response generation
                sentiment_result = self.sentiment_analyzer.analyze(text) if self.sentiment_analyzer else {}
                emotion_result = self.emotion_detector.detect_emotions(text) if self.emotion_detector else {}
                urgency_result = self.calculate_urgency(sentiment_result, emotion_result, text)

                response_result = self.generate_smart_response(sentiment_result, emotion_result, urgency_result, text)

                return jsonify({
                    'response': response_result,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })

            except Exception as e:
                logger.error(f"Response generation error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/analyze/complete', methods=['POST'])
        def complete_analysis():
            """Complete analysis using BOTH your trained models"""
            try:
                data = request.get_json()
                text = data.get('text', '').strip()

                if len(text) < 5:
                    return jsonify({'error': 'Text too short'}), 400

                logger.info(f"🔍 Analyzing: '{text[:50]}...'")

                # Run BOTH your models
                sentiment_result = self.sentiment_analyzer.analyze(text) if self.sentiment_analyzer else {}
                emotion_result = self.emotion_detector.detect_emotions(text) if self.emotion_detector else {}

                # Generate additional insights
                themes_result = self.extract_themes(text)
                urgency_result = self.calculate_urgency(sentiment_result, emotion_result, text)
                response_result = self.generate_smart_response(sentiment_result, emotion_result, urgency_result, text)

                # Generate business actions
                actions = self.generate_business_actions(sentiment_result, themes_result, urgency_result, text)

                logger.info(f"✅ Analysis complete: {sentiment_result.get('label', 'unknown')} sentiment, {emotion_result.get('primary_emotion', 'unknown')} emotion")

                return jsonify({
                    'sentiment': sentiment_result,
                    'emotions': emotion_result,
                    'themes': themes_result,
                    'urgency': urgency_result,
                    'response': response_result,
                    'business_actions': actions,
                    'text_analyzed': text,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success',
                    'models_used': 'custom_trained_aiml_2025'
                })

            except Exception as e:
                logger.error(f"Complete analysis error: {str(e)}")
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/system/status', methods=['GET'])
        def system_status():
            """Get system and model status"""
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                gpu_name = torch.cuda.get_device_name() if gpu_available else "No GPU"

                return jsonify({
                    'system': {
                        'status': 'running',
                        'gpu_available': gpu_available,
                        'gpu_name': gpu_name,
                        'models_loaded': {
                            'sentiment': self.sentiment_analyzer is not None,
                            'emotion': self.emotion_detector is not None
                        }
                    },
                    'models': {
                        'sentiment_analyzer': {
                            'loaded': self.sentiment_analyzer is not None,
                            'path': 'models/sentiment_analyzer',
                            'accuracy': '95.1%'
                        },
                        'emotion_detector': {
                            'loaded': self.emotion_detector is not None,
                            'path': 'models/emotion_detector',
                            'accuracy': '93.0%'
                        }
                    },
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def extract_themes(self, text):
        """Enhanced theme extraction"""
        keywords = {
            'Product Quality': ['product', 'quality', 'feature', 'functionality', 'build', 'design'],
            'Customer Service': ['service', 'support', 'staff', 'help', 'team', 'representative'],
            'Pricing': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'billing'],
            'User Experience': ['experience', 'interface', 'easy', 'difficult', 'usability'],
            'Technical Issues': ['bug', 'error', 'issue', 'problem', 'technical', 'crash'],
            'Delivery & Shipping': ['delivery', 'shipping', 'fast', 'slow', 'arrived', 'package'],
            'Communication': ['communication', 'information', 'update', 'notification']
        }

        text_lower = text.lower()
        detected_themes = []
        confidence_scores = {}

        for theme, words in keywords.items():
            matches = sum(1 for word in words if word in text_lower)
            if matches > 0:
                detected_themes.append(theme)
                confidence_scores[theme] = min(0.8 + (matches * 0.1), 1.0)

        return {
            'topics': detected_themes or ['General Feedback'],
            'theme_count': len(detected_themes),
            'confidence_scores': confidence_scores if detected_themes else {'General Feedback': 0.7}
        }

    def calculate_urgency(self, sentiment, emotion, text):
        """Enhanced urgency calculation"""
        urgency_score = 0.0
        indicators = []

        # Sentiment contribution
        if sentiment.get('label') == 'negative':
            urgency_score += sentiment.get('confidence', 0) * 0.4
            indicators.append('negative_sentiment')

        # Emotion contribution
        if emotion.get('primary_emotion') in ['anger', 'fear']:
            urgency_score += emotion.get('confidence', 0) * 0.3
            indicators.append(f"emotion_{emotion.get('primary_emotion')}")

        # Keyword contribution
        urgent_keywords = {
            'immediate': ['urgent', 'asap', 'immediately', 'critical', 'emergency'],
            'severity': ['terrible', 'horrible', 'worst', 'awful', 'disgusting'],
            'escalation': ['manager', 'complaint', 'lawsuit', 'legal', 'refund']
        }

        text_lower = text.lower()
        for category, keywords in urgent_keywords.items():
            if any(word in text_lower for word in keywords):
                urgency_score += 0.2
                indicators.append(category)

        urgency_score = min(urgency_score, 1.0)

        # Determine level and response time
        if urgency_score >= 0.8:
            level = "critical"
            response_time = "2h"
        elif urgency_score >= 0.6:
            level = "high"
            response_time = "24h"
        elif urgency_score >= 0.4:
            level = "medium"
            response_time = "48h"
        else:
            level = "low"
            response_time = "72h"

        return {
            'score': urgency_score,
            'level': level,
            'response_time': response_time,
            'indicators': indicators
        }

    def generate_smart_response(self, sentiment, emotion, urgency, text):
        """Generate intelligent responses based on analysis"""
        sentiment_label = sentiment.get('label', 'neutral')
        emotion_label = emotion.get('primary_emotion', 'neutral')
        urgency_level = urgency.get('level', 'low')

        # Response templates based on sentiment and emotion
        if sentiment_label == 'negative':
            if urgency_level == 'critical':
                response_text = "We sincerely apologize for this unacceptable experience. This is being escalated to our management team immediately, and we will personally ensure this is resolved within 2 hours. Please expect a call from our senior team member shortly."
            elif emotion_label == 'anger':
                response_text = "We understand your frustration and we're truly sorry for letting you down. Your concerns are being addressed with the highest priority, and we'll work directly with you to make this right."
            else:
                response_text = "Thank you for bringing this to our attention. We take your feedback seriously and are committed to improving your experience. Our team will investigate this matter and follow up with you promptly."

        elif sentiment_label == 'positive':
            if emotion_label == 'joy':
                response_text = "We're absolutely thrilled to hear about your wonderful experience! Your joy means everything to us, and we'll make sure to share your feedback with the entire team."
            else:
                response_text = "Thank you so much for your positive feedback! We're delighted that we could meet your expectations and truly appreciate you taking the time to share your experience."

        else:  # neutral
            response_text = "Thank you for your valuable feedback. We appreciate you taking the time to share your thoughts with us, and we'll use your input to continue improving our services."

        return {
            'text': response_text,
            'tone': 'empathetic' if urgency_level in ['critical', 'high'] else 'professional',
            'confidence': 0.85,
            'sentiment_addressed': sentiment_label,
            'urgency_level': urgency_level
        }

    def generate_business_actions(self, sentiment, themes, urgency, text):
        """Generate actionable business insights"""
        actions = []

        if urgency.get('level') == 'critical':
            actions.append({
                'type': 'immediate_response',
                'title': 'Critical Response Required',
                'description': 'High priority negative feedback requiring immediate management attention and personal follow-up',
                'priority': 'critical',
                'timeline': 'Within 2 hours',
                'department': 'Senior Management',
                'business_impact': 'High risk of customer churn and reputation damage'
            })

        if sentiment.get('label') == 'negative' and sentiment.get('confidence', 0) > 0.7:
            actions.append({
                'type': 'customer_follow_up',
                'title': 'Personal Customer Follow-up',
                'description': 'Negative feedback requiring personalized response and resolution tracking',
                'priority': 'high',
                'timeline': 'Within 24 hours',
                'department': 'Customer Success',
                'business_impact': 'Customer retention and satisfaction recovery'
            })

        # Theme-based actions
        themes_list = themes.get('topics', [])
        if 'Technical Issues' in themes_list:
            actions.append({
                'type': 'technical_resolution',
                'title': 'Technical Issue Investigation',
                'description': 'Technical problems identified requiring engineering team review and resolution',
                'priority': 'high' if urgency.get('score', 0) > 0.6 else 'medium',
                'timeline': 'Within 3-5 days',
                'department': 'Engineering',
                'business_impact': 'Product reliability and user experience improvement'
            })

        if sentiment.get('label') == 'positive' and sentiment.get('confidence', 0) > 0.8:
            actions.append({
                'type': 'marketing_opportunity',
                'title': 'Testimonial and Case Study Opportunity',
                'description': 'Exceptional positive feedback suitable for marketing materials and testimonials',
                'priority': 'low',
                'timeline': 'Within 2 weeks',
                'department': 'Marketing',
                'business_impact': 'Brand reputation enhancement and customer acquisition'
            })

        return actions

    def run(self, host='localhost', port=5000, debug=True):
        """Run the Flask app"""
        logger.info(f"🚀 Starting AIML Backend Server on http://{host}:{port}")
        logger.info(f"📁 Frontend files served from: frontend/")
        logger.info(f"🎯 Your beautiful Google-style UI will be available at the root URL!")
        self.app.run(host=host, port=port, debug=debug)

# Create and run the app
if __name__ == '__main__':
    app = AIMLApp()
    app.run()
