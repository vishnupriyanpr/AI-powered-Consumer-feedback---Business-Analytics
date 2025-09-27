"""Sentiment Analysis Model - AMIL Project"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import logging
import time
from typing import Dict, List, Union
import numpy as np

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Advanced sentiment analyzer with emotion detection optimized for RTX 4060"""

    def __init__(self, device: str = "cuda", model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        # Emotion keywords for advanced analysis
        self.emotion_keywords = {
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated'],
            'joy': ['happy', 'excited', 'thrilled', 'delighted', 'pleased', 'satisfied'],
            'sadness': ['sad', 'disappointed', 'upset', 'unhappy', 'depressed'],
            'fear': ['scared', 'worried', 'anxious', 'nervous', 'concerned'],
            'surprise': ['surprised', 'amazed', 'shocked', 'astonished'],
            'disgust': ['disgusted', 'revolted', 'appalled', 'horrible', 'terrible']
        }

        self.initialize_model()

    def initialize_model(self):
        """Initialize the sentiment analysis model with GPU optimization"""
        logger.info(f"🧠 Loading sentiment model: {self.model_name}")
        logger.info(f"💻 Using device: {self.device}")

        try:
            start_time = time.time()

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Initialize model with GPU optimization
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
                logger.info(f"✅ Model moved to GPU: {torch.cuda.get_device_name()}")

            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )

            load_time = time.time() - start_time
            logger.info(f"✅ Sentiment model loaded successfully in {load_time:.2f} seconds")

        except Exception as e:
            logger.error(f"❌ Failed to load sentiment model: {str(e)}")
            raise

    def analyze(self, text: str, include_emotions: bool = True) -> Dict:
        """
        Analyze sentiment of input text with optional emotion detection

        Args:
            text: Input text to analyze
            include_emotions: Whether to include emotion analysis

        Returns:
            Dict containing sentiment analysis results
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        text = text.strip()
        start_time = time.time()

        try:
            # Get sentiment prediction
            results = self.pipeline(text)

            # Process results
            sentiment_scores = {result['label']: result['score'] for result in results[0]}

            # Determine primary sentiment
            primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[primary_sentiment]

            # Map labels to readable format
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral',
                'LABEL_2': 'positive'
            }

            sentiment_label = label_mapping.get(primary_sentiment, primary_sentiment.lower())

            # Calculate intensity
            intensity = self._calculate_intensity(confidence)

            result = {
                'label': sentiment_label,
                'score': confidence,
                'intensity': intensity,
                'all_scores': {
                    'positive': sentiment_scores.get('LABEL_2', 0),
                    'neutral': sentiment_scores.get('LABEL_1', 0),
                    'negative': sentiment_scores.get('LABEL_0', 0)
                },
                'confidence': confidence,
                'inference_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            # Add emotion analysis if requested
            if include_emotions:
                emotions = self._analyze_emotions(text)
                result['emotions'] = emotions

            return result

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {
                'label': 'neutral',
                'score': 0.0,
                'intensity': 'low',
                'all_scores': {'positive': 0, 'neutral': 1, 'negative': 0},
                'confidence': 0.0,
                'error': str(e),
                'inference_time_ms': round((time.time() - start_time) * 1000, 2)
            }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for multiple texts in batch for better GPU utilization

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment analysis results
        """
        if not texts:
            return []

        start_time = time.time()
        results = []

        try:
            # Process in batches for memory efficiency
            batch_size = 16 if self.device == "cuda" else 4

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_results = self.pipeline(batch_texts)

                for text, result in zip(batch_texts, batch_results):
                    processed_result = self._process_pipeline_result(text, result)
                    results.append(processed_result)

            total_time = time.time() - start_time
            logger.info(f"Batch analysis completed: {len(texts)} texts in {total_time:.2f}s")

        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {str(e)}")
            # Return error results for each text
            for text in texts:
                results.append({
                    'label': 'neutral',
                    'score': 0.0,
                    'error': str(e),
                    'text_preview': text[:50] + '...' if len(text) > 50 else text
                })

        return results

    def _process_pipeline_result(self, text: str, pipeline_result: List[Dict]) -> Dict:
        """Process individual pipeline result"""
        sentiment_scores = {result['label']: result['score'] for result in pipeline_result}
        primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        confidence = sentiment_scores[primary_sentiment]

        label_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral',
            'LABEL_2': 'positive'
        }

        sentiment_label = label_mapping.get(primary_sentiment, primary_sentiment.lower())

        return {
            'label': sentiment_label,
            'score': confidence,
            'intensity': self._calculate_intensity(confidence),
            'all_scores': {
                'positive': sentiment_scores.get('LABEL_2', 0),
                'neutral': sentiment_scores.get('LABEL_1', 0),
                'negative': sentiment_scores.get('LABEL_0', 0)
            },
            'confidence': confidence,
            'text_preview': text[:50] + '...' if len(text) > 50 else text
        }

    def _calculate_intensity(self, confidence: float) -> str:
        """Calculate sentiment intensity based on confidence score"""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        else:
            return 'low'

    def _analyze_emotions(self, text: str) -> Dict:
        """
        Analyze emotions in text using keyword matching
        More sophisticated emotion models can be added later
        """
        text_lower = text.lower()
        emotions_found = {}

        for emotion, keywords in self.emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                emotions_found[emotion] = {
                    'intensity': min(matches / len(keywords), 1.0),
                    'keywords_found': matches
                }

        return emotions_found

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'tokenizer_vocab_size': self.tokenizer.vocab_size if self.tokenizer else 0,
            'max_length': self.tokenizer.model_max_length if self.tokenizer else 0,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }

    def benchmark_performance(self, test_texts: List[str] = None) -> Dict:
        """Benchmark model performance"""
        if test_texts is None:
            test_texts = [
                "I absolutely love this product! It's amazing!",
                "This is okay, nothing special but decent.",
                "I hate this so much, it's completely broken and useless!",
                "The experience was wonderful and exceeded my expectations.",
                "It's just average, could be better but not terrible."
            ]

        start_time = time.time()
        results = []

        for text in test_texts:
            result = self.analyze(text, include_emotions=False)
            results.append(result)

        total_time = time.time() - start_time
        avg_time_per_text = total_time / len(test_texts)

        return {
            'total_texts': len(test_texts),
            'total_time_seconds': round(total_time, 3),
            'average_time_per_text_ms': round(avg_time_per_text * 1000, 2),
            'throughput_texts_per_second': round(len(test_texts) / total_time, 2),
            'device': self.device,
            'sample_results': results[:2]  # Show first 2 results as samples
        }

# Factory function for creating sentiment analyzer
def create_sentiment_analyzer(device: str = "cuda", model_name: str = None) -> SentimentAnalyzer:
    """Factory function to create sentiment analyzer with error handling"""
    try:
        if model_name is None:
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

        analyzer = SentimentAnalyzer(device=device, model_name=model_name)
        return analyzer

    except Exception as e:
        logger.error(f"Failed to create sentiment analyzer: {str(e)}")
        # Fallback to CPU if GPU fails
        if device == "cuda":
            logger.warning("Falling back to CPU for sentiment analysis")
            return SentimentAnalyzer(device="cpu", model_name=model_name)
        raise
