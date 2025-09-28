"""Topic Modeling without BERTopic - AMIL Project"""

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import time
import re
from typing import Dict, List, Tuple, Union
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

class TopicModeler:
    """Simplified topic modeling without BERTopic dependency"""

    def __init__(self, device: str = "cuda", min_topic_size: int = 3):
        self.device = device
        self.min_topic_size = min_topic_size
        self.sentence_model = None
        self.vectorizer = None

        # GDG-specific keywords (same as before)
        self.gdg_keywords = {
            'technical_content': [
                'code', 'programming', 'api', 'framework', 'library', 'sdk',
                'development', 'developer', 'coding', 'algorithm', 'database',
                'cloud', 'ml', 'ai', 'machine learning', 'tensorflow', 'android'
            ],
            'presentation_quality': [
                'speaker', 'presentation', 'slides', 'demo', 'explanation',
                'clear', 'unclear', 'confusing', 'boring', 'engaging',
                'pace', 'speed', 'slow', 'fast', 'understandable'
            ],
            'event_logistics': [
                'venue', 'location', 'parking', 'food', 'refreshments',
                'wifi', 'internet', 'microphone', 'audio', 'video',
                'seating', 'space', 'room', 'temperature', 'timing'
            ],
            'networking': [
                'networking', 'community', 'meetup', 'connect', 'social',
                'interaction', 'discussion', 'collaboration', 'team',
                'group', 'people', 'participants', 'attendees'
            ],
            'workshop_content': [
                'workshop', 'hands-on', 'tutorial', 'exercise', 'practice',
                'setup', 'installation', 'configuration', 'prerequisites',
                'difficulty', 'level', 'beginner', 'advanced', 'intermediate'
            ]
        }

        self.custom_stopwords = set(stopwords.words('english')).union({
            'event', 'gdg', 'google', 'session', 'today', 'yesterday',
            'good', 'bad', 'nice', 'great', 'ok', 'okay', 'well',
            'really', 'very', 'quite', 'pretty', 'much', 'many'
        })

        self.initialize_models()

    def initialize_models(self):
        """Initialize simplified topic modeling components"""
        logger.info("🧠 Initializing simplified topic modeling...")

        try:
            start_time = time.time()

            # Initialize sentence transformer
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self.sentence_model = SentenceTransformer(model_name, device=self.device)

            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )

            load_time = time.time() - start_time
            logger.info(f"✅ Topic modeling components loaded in {load_time:.2f} seconds")

        except Exception as e:
            logger.error(f"❌ Failed to initialize topic models: {str(e)}")
            raise

    def extract_themes(self, text: str, max_themes: int = 5) -> Dict:
        """Extract themes using simplified approach"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        text = text.strip()
        start_time = time.time()

        try:
            # Use keyword-based theme extraction
            keyword_themes = self._extract_keyword_themes(text)
            tfidf_themes = self._extract_tfidf_themes(text)

            # Combine themes
            all_themes = self._combine_themes(keyword_themes, tfidf_themes)
            top_themes = all_themes[:max_themes] if all_themes else []

            # Generate word cloud data
            wordcloud_data = self._generate_wordcloud_data(text)

            result = {
                'topics': top_themes,
                'theme_count': len(top_themes),
                'confidence_scores': {theme: 0.8 for theme in top_themes},
                'categories': self._categorize_themes(top_themes),
                'wordcloud_data': wordcloud_data,
                'analysis_methods': ['keyword_matching', 'tfidf'],
                'inference_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            return result

        except Exception as e:
            logger.error(f"Theme extraction failed: {str(e)}")
            return {
                'topics': [],
                'theme_count': 0,
                'confidence_scores': {},
                'categories': {},
                'wordcloud_data': {},
                'error': str(e),
                'inference_time_ms': round((time.time() - start_time) * 1000, 2)
            }

    # Include all the other methods from the original file
    # (they don't depend on BERTopic)

    def _extract_keyword_themes(self, text: str) -> List[Tuple[str, float]]:
        """Extract themes using keyword matching"""
        text_lower = text.lower()
        theme_scores = {}

        for category, keywords in self.gdg_keywords.items():
            matches = []
            for keyword in keywords:
                if keyword in text_lower:
                    matches.append(keyword)

            if matches:
                score = len(matches) / len(keywords)
                theme_scores[category.replace('_', ' ').title()] = score

        return sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)

    def _extract_tfidf_themes(self, text: str) -> List[Tuple[str, float]]:
        """Extract themes using TF-IDF analysis"""
        try:
            cleaned_text = self._clean_text(text)
            tokens = word_tokenize(cleaned_text.lower())

            filtered_tokens = [token for token in tokens if token not in self.custom_stopwords and len(token) > 2]

            if len(filtered_tokens) < 3:
                return []

            doc = ' '.join(filtered_tokens)
            tfidf_matrix = self.vectorizer.fit_transform([doc])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            term_scores = list(zip(feature_names, scores))
            term_scores.sort(key=lambda x: x[1], reverse=True)

            themes = []
            for term, score in term_scores[:10]:
                if score > 0.1:
                    themes.append((term.replace('_', ' ').title(), score))

            return themes

        except Exception as e:
            logger.warning(f"TF-IDF theme extraction failed: {str(e)}")
            return []

    def _combine_themes(self, *theme_lists) -> List[str]:
        """Combine themes from different methods"""
        all_themes = {}

        for theme_list in theme_lists:
            for theme, score in theme_list:
                if theme in all_themes:
                    all_themes[theme] += score
                else:
                    all_themes[theme] = score

        sorted_themes = sorted(all_themes.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, score in sorted_themes]

    def _categorize_themes(self, themes: List[str]) -> Dict[str, List[str]]:
        """Categorize themes"""
        categories = {
            'Technical': [],
            'Event Management': [],
            'Content Quality': [],
            'Community': [],
            'General': []
        }

        for theme in themes:
            theme_lower = theme.lower()

            if any(keyword in theme_lower for keyword in ['technical', 'code', 'programming']):
                categories['Technical'].append(theme)
            elif any(keyword in theme_lower for keyword in ['venue', 'logistics', 'wifi']):
                categories['Event Management'].append(theme)
            elif any(keyword in theme_lower for keyword in ['presentation', 'speaker', 'content']):
                categories['Content Quality'].append(theme)
            elif any(keyword in theme_lower for keyword in ['networking', 'community', 'social']):
                categories['Community'].append(theme)
            else:
                categories['General'].append(theme)

        return {k: v for k, v in categories.items() if v}

    def _generate_wordcloud_data(self, text: str) -> Dict:
        """Generate word frequency data"""
        try:
            cleaned_text = self._clean_text(text)
            tokens = word_tokenize(cleaned_text.lower())

            filtered_tokens = [
                token for token in tokens
                if token not in self.custom_stopwords
                   and len(token) > 2
                   and token.isalpha()
            ]

            word_freq = Counter(filtered_tokens)
            top_words = dict(word_freq.most_common(20))

            return {
                'words': top_words,
                'total_words': len(filtered_tokens),
                'unique_words': len(word_freq)
            }

        except Exception as e:
            logger.warning(f"Wordcloud data generation failed: {str(e)}")
            return {'words': {}, 'total_words': 0, 'unique_words': 0}

    def _clean_text(self, text: str) -> str:
        """Clean text for processing"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
