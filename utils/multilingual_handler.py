"""Multilingual Text Processing Handler - AMIL Project"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import re
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
from googletrans import Translator
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

class MultilingualHandler:
    """Advanced multilingual text processing with language detection and translation"""

    def __init__(self):
        self.translator = Translator()
        self._translation_cache = {}
        self._detection_cache = {}
        self._lock = threading.Lock()

        # Supported languages with their full names
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'hi': 'Hindi',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'ru': 'Russian',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'da': 'Danish',
            'fi': 'Finnish',
            'pl': 'Polish',
            'tr': 'Turkish',
            'th': 'Thai'
        }

        # Language-specific text preprocessing patterns
        self.language_patterns = {
            'en': {
                'contractions': {
                    "don't": "do not", "won't": "will not", "can't": "cannot",
                    "shouldn't": "should not", "wouldn't": "would not",
                    "couldn't": "could not", "isn't": "is not", "aren't": "are not",
                    "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
                    "haven't": "have not", "hadn't": "had not", "doesn't": "does not",
                    "didn't": "did not", "you're": "you are", "we're": "we are"
                },
                'informal_words': {
                    'gonna': 'going to', 'wanna': 'want to', 'gotta': 'got to',
                    'dunno': 'do not know', 'kinda': 'kind of', 'sorta': 'sort of'
                }
            },
            'es': {
                'accents': True,  # Preserve Spanish accents
                'contractions': {}  # Spanish doesn't have contractions like English
            },
            'fr': {
                'accents': True,  # Preserve French accents
                'contractions': {}
            },
            'de': {
                'compound_words': True,  # German compound word handling
                'case_sensitive': True
            },
            'hi': {
                'devanagari': True,  # Hindi script support
                'transliteration': True
            },
            'zh': {
                'simplified': True,  # Support for simplified Chinese
                'traditional': True  # Support for traditional Chinese
            }
        }

        # Common phrases in different languages for better context understanding
        self.common_phrases = {
            'positive': {
                'en': ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love it'],
                'es': ['excelente', 'increíble', 'fantástico', 'maravilloso', 'me encanta'],
                'fr': ['excellent', 'incroyable', 'fantastique', 'merveilleux', 'j\'adore'],
                'de': ['ausgezeichnet', 'unglaublich', 'fantastisch', 'wunderbar', 'liebe es'],
                'hi': ['बहुत अच्छा', 'शानदार', 'अद्भुत', 'बेहतरीन', 'पसंद है']
            },
            'negative': {
                'en': ['terrible', 'awful', 'horrible', 'bad', 'hate', 'worst'],
                'es': ['terrible', 'horrible', 'malo', 'odio', 'peor'],
                'fr': ['terrible', 'horrible', 'mauvais', 'déteste', 'pire'],
                'de': ['schrecklich', 'furchtbar', 'schlecht', 'hasse', 'schlimmste'],
                'hi': ['भयानक', 'बुरा', 'घृणा', 'सबसे खराब', 'नफरत']
            }
        }

    @lru_cache(maxsize=1000)
    def detect_language(self, text: str, return_confidence: bool = False) -> Dict:
        """
        Detect language of input text with confidence scores

        Args:
            text: Input text to analyze
            return_confidence: Whether to return confidence scores for all detected languages

        Returns:
            Dict containing detected language and optional confidence scores
        """
        if not text or not text.strip():
            return {'language': 'unknown', 'confidence': 0.0, 'error': 'Empty text'}

        # Check cache first
        text_hash = hash(text.strip().lower())
        if text_hash in self._detection_cache:
            return self._detection_cache[text_hash]

        try:
            start_time = time.time()

            # Primary detection
            detected_lang = detect(text)

            # Get confidence scores if requested
            if return_confidence:
                lang_probs = detect_langs(text)
                confidence_scores = {lang.lang: lang.prob for lang in lang_probs}
                primary_confidence = confidence_scores.get(detected_lang, 0.0)
            else:
                confidence_scores = {}
                primary_confidence = 0.9  # Default high confidence for simple detection

            # Validate against supported languages
            if detected_lang not in self.supported_languages:
                detected_lang = 'en'  # Default fallback
                primary_confidence = 0.5

            result = {
                'language': detected_lang,
                'language_name': self.supported_languages.get(detected_lang, 'Unknown'),
                'confidence': round(primary_confidence, 3),
                'detection_time_ms': round((time.time() - start_time) * 1000, 2),
                'supported': detected_lang in self.supported_languages
            }

            if return_confidence:
                result['all_detections'] = confidence_scores

            # Cache result
            with self._lock:
                self._detection_cache[text_hash] = result

            return result

        except LangDetectException as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return {
                'language': 'en',  # Default fallback
                'language_name': 'English',
                'confidence': 0.1,
                'error': str(e),
                'supported': True
            }
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {str(e)}")
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'supported': False
            }

    def translate_text(self, text: str, target_language: str = 'en', source_language: str = 'auto') -> Dict:
        """
        Translate text to target language

        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code ('auto' for auto-detection)

        Returns:
            Dict containing translation result and metadata
        """
        if not text or not text.strip():
            return {'translated_text': '', 'error': 'Empty text'}

        # Check cache
        cache_key = f"{hash(text)}_{source_language}_{target_language}"
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]

        try:
            start_time = time.time()

            # Skip translation if already in target language
            if source_language != 'auto':
                detected_lang = source_language
            else:
                detection_result = self.detect_language(text)
                detected_lang = detection_result['language']

            if detected_lang == target_language:
                result = {
                    'translated_text': text,
                    'source_language': detected_lang,
                    'target_language': target_language,
                    'translation_needed': False,
                    'confidence': 1.0,
                    'translation_time_ms': 0
                }
            else:
                # Perform translation
                translation = self.translator.translate(
                    text,
                    src=source_language,
                    dest=target_language
                )

                result = {
                    'translated_text': translation.text,
                    'source_language': translation.src,
                    'target_language': target_language,
                    'translation_needed': True,
                    'confidence': 0.9,  # Default confidence for successful translation
                    'translation_time_ms': round((time.time() - start_time) * 1000, 2)
                }

            # Cache result
            with self._lock:
                self._translation_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return {
                'translated_text': text,  # Return original text as fallback
                'source_language': 'unknown',
                'target_language': target_language,
                'translation_needed': True,
                'error': str(e),
                'translation_time_ms': round((time.time() - start_time) * 1000, 2)
            }

    def preprocess_multilingual_text(self, text: str, language: str = 'auto') -> Dict:
        """
        Preprocess text based on language-specific rules

        Args:
            text: Input text to preprocess
            language: Language code ('auto' for auto-detection)

        Returns:
            Dict containing preprocessed text and metadata
        """
        if not text or not text.strip():
            return {'processed_text': '', 'error': 'Empty text'}

        start_time = time.time()
        original_text = text

        try:
            # Detect language if auto
            if language == 'auto':
                detection_result = self.detect_language(text)
                detected_language = detection_result['language']
            else:
                detected_language = language

            # Apply language-specific preprocessing
            processed_text = self._apply_language_preprocessing(text, detected_language)

            # Common preprocessing for all languages
            processed_text = self._apply_common_preprocessing(processed_text)

            result = {
                'processed_text': processed_text,
                'original_text': original_text,
                'detected_language': detected_language,
                'language_name': self.supported_languages.get(detected_language, 'Unknown'),
                'preprocessing_applied': True,
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            return result

        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}")
            return {
                'processed_text': text,  # Return original text as fallback
                'original_text': original_text,
                'detected_language': 'unknown',
                'error': str(e),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

    def _apply_language_preprocessing(self, text: str, language: str) -> str:
        """Apply language-specific preprocessing rules"""
        if language not in self.language_patterns:
            return text

        patterns = self.language_patterns[language]
        processed_text = text

        # Handle English contractions and informal words
        if language == 'en':
            # Expand contractions
            if 'contractions' in patterns:
                for contraction, expansion in patterns['contractions'].items():
                    processed_text = re.sub(
                        r'\b' + re.escape(contraction) + r'\b',
                        expansion,
                        processed_text,
                        flags=re.IGNORECASE
                    )

            # Handle informal words
            if 'informal_words' in patterns:
                for informal, formal in patterns['informal_words'].items():
                    processed_text = re.sub(
                        r'\b' + re.escape(informal) + r'\b',
                        formal,
                        processed_text,
                        flags=re.IGNORECASE
                    )

        # Handle other language-specific rules
        elif language in ['es', 'fr'] and patterns.get('accents'):
            # Preserve accented characters for Spanish and French
            pass  # No modification needed, just preserve

        elif language == 'de' and patterns.get('compound_words'):
            # German compound word handling would go here
            pass

        return processed_text

    def _apply_common_preprocessing(self, text: str) -> str:
        """Apply common preprocessing rules for all languages"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Handle common punctuation issues
        text = re.sub(r'([.!?])\1+', r'\1', text)  # Remove repeated punctuation
        text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove space before punctuation

        # Fix common encoding issues
        text = text.replace(''', "'").replace('"', '"').replace('"', '"')
        
        return text
    
    def analyze_multilingual_sentiment_keywords(self, text: str, language: str = 'auto') -> Dict:
        """
        Analyze sentiment keywords specific to detected language
        
        Args:
            text: Input text to analyze
            language: Language code ('auto' for auto-detection)
            
        Returns:
            Dict containing language-specific sentiment analysis
        """
        if language == 'auto':
            detection_result = self.detect_language(text)
            detected_language = detection_result['language']
        else:
            detected_language = language
        
        text_lower = text.lower()
        
        # Initialize results
        sentiment_indicators = {
            'positive_keywords': [],
            'negative_keywords': [],
            'positive_count': 0,
            'negative_count': 0,
            'language': detected_language,
            'language_name': self.supported_languages.get(detected_language, 'Unknown')
        }
        
        # Check for language-specific sentiment keywords
        if detected_language in self.common_phrases['positive']:
            positive_phrases = self.common_phrases['positive'][detected_language]
            for phrase in positive_phrases:
                if phrase.lower() in text_lower:
                    sentiment_indicators['positive_keywords'].append(phrase)
        
        if detected_language in self.common_phrases['negative']:
            negative_phrases = self.common_phrases['negative'][detected_language]
            for phrase in negative_phrases:
                if phrase.lower() in text_lower:
                    sentiment_indicators['negative_keywords'].append(phrase)
        
        sentiment_indicators['positive_count'] = len(sentiment_indicators['positive_keywords'])
        sentiment_indicators['negative_count'] = len(sentiment_indicators['negative_keywords'])
        
        # Calculate sentiment score
        total_indicators = sentiment_indicators['positive_count'] + sentiment_indicators['negative_count']
        if total_indicators > 0:
            sentiment_score = (sentiment_indicators['positive_count'] - sentiment_indicators['negative_count']) / total_indicators
        else:
            sentiment_score = 0.0
        
        sentiment_indicators['sentiment_score'] = sentiment_score
        sentiment_indicators['sentiment_label'] = 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
        
        return sentiment_indicators
    
    def batch_detect_languages(self, texts: List[str]) -> List[Dict]:
        """Detect languages for multiple texts efficiently"""
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.detect_language(text)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'language': 'unknown',
                    'confidence': 0.0,
                    'batch_index': i,
                    'error': str(e)
                })
        
        return results
    
    def batch_translate(self, texts: List[str], target_language: str = 'en') -> List[Dict]:
        """Translate multiple texts efficiently"""
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.translate_text(text, target_language)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'translated_text': text,
                    'batch_index': i,
                    'error': str(e)
                })
        
        return results
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.supported_languages.copy()
    
    def clear_cache(self):
        """Clear translation and detection caches"""
        with self._lock:
            self._translation_cache.clear()
            self._detection_cache.clear()
        logger.info("Multilingual handler caches cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'translation_cache_size': len(self._translation_cache),
            'detection_cache_size': len(self._detection_cache),
            'supported_languages_count': len(self.supported_languages)
        }
    
    def benchmark_performance(self, test_texts: List[str] = None) -> Dict:
        """Benchmark multilingual processing performance"""
        if test_texts is None:
            test_texts = [
                "This is a great event!",  # English
                "¡Este evento es fantástico!",  # Spanish
                "Cet événement est merveilleux!",  # French
                "Diese Veranstaltung ist großartig!",  # German
                "यह कार्यक्रम बहुत अच्छा है!",  # Hindi
            ]
        
        start_time = time.time()
        results = {
            'detection_results': [],
            'translation_results': [],
            'preprocessing_results': []
        }
        
        # Test language detection
        for text in test_texts:
            detection_result = self.detect_language(text, return_confidence=True)
            results['detection_results'].append(detection_result)
        
        # Test translation to English
        for text in test_texts:
            translation_result = self.translate_text(text, 'en')
            results['translation_results'].append(translation_result)
        
        # Test preprocessing
        for text in test_texts:
            preprocessing_result = self.preprocess_multilingual_text(text)
            results['preprocessing_results'].append(preprocessing_result)
        
        total_time = time.time() - start_time
        
        return {
            'total_texts': len(test_texts),
            'total_time_seconds': round(total_time, 3),
            'average_time_per_text_ms': round((total_time / len(test_texts)) * 1000, 2),
            'operations_per_second': round((len(test_texts) * 3) / total_time, 2),  # 3 operations per text
            'results_sample': {
                'detection': results['detection_results'][0] if results['detection_results'] else {},
                'translation': results['translation_results'][0] if results['translation_results'] else {},
                'preprocessing': results['preprocessing_results'][0] if results['preprocessing_results'] else {}
            }
        }
