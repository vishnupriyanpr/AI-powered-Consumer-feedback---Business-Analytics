"""Topic Modeling and Theme Extraction - AMIL Project"""

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from bertopic import BERTopic
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
    """Advanced topic modeling with BERTopic optimized for RTX 4060 CUDA 12.4"""

    def __init__(self, device: str = "cuda", min_topic_size: int = 3):
        self.device = device
        self.min_topic_size = min_topic_size
        self.sentence_model = None
        self.topic_model = None
        self.vectorizer = None

        # GDG-specific keywords for theme detection
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

        # Stopwords for better theme extraction
        self.custom_stopwords = set(stopwords.words('english')).union({
            'event', 'gdg', 'google', 'session', 'today', 'yesterday',
            'good', 'bad', 'nice', 'great', 'ok', 'okay', 'well',
            'really', 'very', 'quite', 'pretty', 'much', 'many'
        })

        self.initialize_models()

    def initialize_models(self):
        """Initialize topic modeling components with CUDA 12.4 optimization"""
        logger.info("🧠 Initializing topic modeling components...")
        logger.info(f"💻 Using device: {self.device} with CUDA 12.4")

        try:
            start_time = time.time()

            # Initialize sentence transformer for embeddings
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self.sentence_model = SentenceTransformer(model_name, device=self.device)

            # Optimize for CUDA 12.4
            if self.device == "cuda" and torch.cuda.is_available():
                # Enable optimizations for RTX 4060
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                logger.info(f"✅ Sentence model loaded on GPU: {torch.cuda.get_device_name()}")
                logger.info(f"🚀 CUDA 12.4 optimizations enabled")

            # Initialize TF-IDF vectorizer for keyword extraction
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )

            # Initialize BERTopic model
            self.topic_model = BERTopic(
                embedding_model=self.sentence_model,
                min_topic_size=self.min_topic_size,
                nr_topics="auto",
                calculate_probabilities=True,
                verbose=False
            )

            load_time = time.time() - start_time
            logger.info(f"✅ Topic modeling components loaded in {load_time:.2f} seconds")

        except Exception as e:
            logger.error(f"❌ Failed to initialize topic models: {str(e)}")
            raise

    def extract_themes(self, text: str, max_themes: int = 5) -> Dict:
        """
        Extract themes and topics from feedback text

        Args:
            text: Input feedback text
            max_themes: Maximum number of themes to return

        Returns:
            Dict containing extracted themes and analysis
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        text = text.strip()
        start_time = time.time()

        try:
            # Multi-approach theme extraction
            keyword_themes = self._extract_keyword_themes(text)
            tfidf_themes = self._extract_tfidf_themes(text)
            embedding_themes = self._extract_embedding_themes(text)

            # Combine and rank themes
            all_themes = self._combine_themes(keyword_themes, tfidf_themes, embedding_themes)

            # Select top themes
            top_themes = all_themes[:max_themes] if all_themes else []

            # Generate word cloud data for visualization
            wordcloud_data = self._generate_wordcloud_data(text)

            result = {
                'topics': top_themes,
                'theme_count': len(top_themes),
                'confidence_scores': {theme: 0.8 for theme in top_themes},  # Placeholder confidence
                'categories': self._categorize_themes(top_themes),
                'wordcloud_data': wordcloud_data,
                'analysis_methods': ['keyword_matching', 'tfidf', 'embeddings'],
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

    def _extract_keyword_themes(self, text: str) -> List[Tuple[str, float]]:
        """Extract themes using GDG-specific keyword matching"""
        text_lower = text.lower()
        theme_scores = {}

        for category, keywords in self.gdg_keywords.items():
            matches = []
            for keyword in keywords:
                if keyword in text_lower:
                    matches.append(keyword)

            if matches:
                # Score based on number of matches and keyword importance
                score = len(matches) / len(keywords)
                theme_scores[category.replace('_', ' ').title()] = score

        # Sort by score
        return sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)

    def _extract_tfidf_themes(self, text: str) -> List[Tuple[str, float]]:
        """Extract themes using TF-IDF analysis"""
        try:
            # Clean and tokenize text
            cleaned_text = self._clean_text(text)
            tokens = word_tokenize(cleaned_text.lower())

            # Remove stopwords
            filtered_tokens = [token for token in tokens if token not in self.custom_stopwords and len(token) > 2]

            if len(filtered_tokens) < 3:
                return []

            # Create document for TF-IDF
            doc = ' '.join(filtered_tokens)

            # Get TF-IDF scores
            tfidf_matrix = self.vectorizer.fit_transform([doc])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            # Get top scoring terms
            term_scores = list(zip(feature_names, scores))
            term_scores.sort(key=lambda x: x[1], reverse=True)

            # Convert terms to themes
            themes = []
            for term, score in term_scores[:10]:
                if score > 0.1:  # Minimum threshold
                    themes.append((term.replace('_', ' ').title(), score))

            return themes

        except Exception as e:
            logger.warning(f"TF-IDF theme extraction failed: {str(e)}")
            return []

    def _extract_embedding_themes(self, text: str) -> List[Tuple[str, float]]:
        """Extract themes using sentence embeddings and clustering"""
        try:
            # Split text into sentences
            sentences = self._split_sentences(text)

            if len(sentences) < 2:
                return []

            # Generate embeddings
            embeddings = self.sentence_model.encode(sentences)

            # Perform clustering if we have enough sentences
            if len(sentences) >= 3:
                n_clusters = min(3, len(sentences))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings)

                # Extract representative themes from clusters
                themes = []
                for i in range(n_clusters):
                    cluster_sentences = [sentences[j] for j in range(len(sentences)) if clusters[j] == i]
                    if cluster_sentences:
                        # Use the longest sentence as representative theme
                        representative = max(cluster_sentences, key=len)
                        theme = self._extract_theme_from_sentence(representative)
                        if theme:
                            themes.append((theme, 0.7))  # Default confidence

                return themes

            return []

        except Exception as e:
            logger.warning(f"Embedding theme extraction failed: {str(e)}")
            return []

    def _combine_themes(self, *theme_lists) -> List[str]:
        """Combine themes from different extraction methods"""
        all_themes = {}

        for theme_list in theme_lists:
            for theme, score in theme_list:
                if theme in all_themes:
                    all_themes[theme] += score
                else:
                    all_themes[theme] = score

        # Sort by combined score and return theme names
        sorted_themes = sorted(all_themes.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, score in sorted_themes]

    def _categorize_themes(self, themes: List[str]) -> Dict[str, List[str]]:
        """Categorize themes into GDG-relevant categories"""
        categories = {
            'Technical': [],
            'Event Management': [],
            'Content Quality': [],
            'Community': [],
            'General': []
        }

        for theme in themes:
            theme_lower = theme.lower()

            if any(keyword in theme_lower for keyword in ['technical', 'code', 'programming', 'api', 'development']):
                categories['Technical'].append(theme)
            elif any(keyword in theme_lower for keyword in ['venue', 'logistics', 'wifi', 'food', 'timing']):
                categories['Event Management'].append(theme)
            elif any(keyword in theme_lower for keyword in ['presentation', 'speaker', 'content', 'workshop']):
                categories['Content Quality'].append(theme)
            elif any(keyword in theme_lower for keyword in ['networking', 'community', 'social', 'interaction']):
                categories['Community'].append(theme)
            else:
                categories['General'].append(theme)

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def _generate_wordcloud_data(self, text: str) -> Dict:
        """Generate word frequency data for word cloud visualization"""
        try:
            cleaned_text = self._clean_text(text)
            tokens = word_tokenize(cleaned_text.lower())

            # Filter tokens
            filtered_tokens = [
                token for token in tokens
                if token not in self.custom_stopwords
                   and len(token) > 2
                   and token.isalpha()
            ]

            # Count frequency
            word_freq = Counter(filtered_tokens)

            # Get top 20 words
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
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences

    def _extract_theme_from_sentence(self, sentence: str) -> str:
        """Extract a theme description from a sentence"""
        # Simple extraction - take key phrases
        words = sentence.split()
        if len(words) > 5:
            # Take middle portion of sentence as theme
            start = len(words) // 4
            end = 3 * len(words) // 4
            theme_words = words[start:end]
            return ' '.join(theme_words).strip()
        return sentence.strip()

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze themes for multiple texts efficiently"""
        if not texts:
            return []

        start_time = time.time()
        results = []

        try:
            # Process each text
            for i, text in enumerate(texts):
                if text and text.strip():
                    result = self.extract_themes(text, max_themes=3)
                    result['batch_index'] = i
                    result['text_preview'] = text[:100] + '...' if len(text) > 100 else text
                else:
                    result = {
                        'topics': [],
                        'theme_count': 0,
                        'batch_index': i,
                        'error': 'Empty or invalid text'
                    }

                results.append(result)

            total_time = time.time() - start_time
            logger.info(f"Batch theme analysis completed: {len(texts)} texts in {total_time:.2f}s")

        except Exception as e:
            logger.error(f"Batch theme analysis failed: {str(e)}")
            # Return error results
            for i, text in enumerate(texts):
                results.append({
                    'topics': [],
                    'theme_count': 0,
                    'batch_index': i,
                    'error': str(e)
                })

        return results

    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'sentence_model': self.sentence_model.get_model_card_data() if self.sentence_model else None,
            'device': self.device,
            'min_topic_size': self.min_topic_size,
            'gdg_categories': list(self.gdg_keywords.keys()),
            'custom_stopwords_count': len(self.custom_stopwords)
        }

    def benchmark_performance(self, test_texts: List[str] = None) -> Dict:
        """Benchmark topic modeling performance"""
        if test_texts is None:
            test_texts = [
                "The Android development workshop was great! The speaker explained the concepts clearly and the hands-on coding session was very helpful. However, the wifi connection was unstable which made it difficult to download the required SDKs.",
                "I loved the machine learning presentation today. The demo was impressive and I learned a lot about TensorFlow. The venue was perfect and the networking session afterwards was valuable for connecting with other developers.",
                "The event logistics could be improved. The parking situation was terrible and the food was cold. Also, the microphone wasn't working properly during the first presentation. Content was good though.",
                "Great community event! Met some amazing developers and had interesting discussions about cloud technologies. The speaker was engaging and the workshop format worked well. Looking forward to the next meetup.",
                "The beginner-friendly tutorial was exactly what I needed. Clear explanations, good pacing, and practical examples. The setup instructions were easy to follow and everything worked smoothly."
            ]

        start_time = time.time()
        results = []

        for text in test_texts:
            result = self.extract_themes(text, max_themes=3)
            results.append(result)

        total_time = time.time() - start_time
        avg_time = total_time / len(test_texts)

        return {
            'total_texts': len(test_texts),
            'total_time_seconds': round(total_time, 3),
            'average_time_per_text_ms': round(avg_time * 1000, 2),
            'throughput_texts_per_second': round(len(test_texts) / total_time, 2),
            'device': self.device,
            'sample_results': results[:2]
        }
