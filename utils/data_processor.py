"""Data Processing Utilities for AI Customer Feedback Analyzer - AMIL Project"""

import pandas as pd
import numpy as np
import logging
import time
import json
import csv
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
from collections import Counter, defaultdict
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class DataProcessor:
    """Advanced data processing utilities with batch operations and caching"""

    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._processing_cache = {}
        self._lock = threading.Lock()

        # Data validation patterns
        self.validation_patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone': re.compile(r'(\+?1-?)?(\(?([0-9]{3})\)?[-.\s]?)?([0-9]{3})[-.\s]?([0-9]{4})'),
            'social_handle': re.compile(r'@[A-Za-z0-9_]+'),
            'hashtag': re.compile(r'#[A-Za-z0-9_]+')
        }

        # Text cleaning patterns
        self.cleaning_patterns = {
            'html_tags': re.compile(r'<[^>]+>'),
            'extra_whitespace': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s.,!?;:\'"()-]'),
            'repeated_chars': re.compile(r'(.)\1{3,}'),  # 4+ repeated characters
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'email_addresses': re.compile(r'\S+@\S+\.\S+')
        }

        # Statistical thresholds for data quality
        self.quality_thresholds = {
            'min_text_length': 5,
            'max_text_length': 10000,
            'min_word_count': 2,
            'max_duplicate_percentage': 20,  # Max % of duplicate records
            'min_language_confidence': 0.7,
            'max_special_char_ratio': 0.3
        }

    def clean_text(self, text: str, aggressive: bool = False) -> Dict:
        """
        Clean and normalize text data

        Args:
            text: Input text to clean
            aggressive: Whether to apply aggressive cleaning

        Returns:
            Dict containing cleaned text and metadata
        """
        if not text or not isinstance(text, str):
            return {
                'cleaned_text': '',
                'original_text': text,
                'cleaning_applied': [],
                'quality_score': 0.0,
                'error': 'Invalid input text'
            }

        start_time = time.time()
        original_text = text
        cleaned_text = text
        cleaning_applied = []

        try:
            # Remove HTML tags
            if self.cleaning_patterns['html_tags'].search(cleaned_text):
                cleaned_text = self.cleaning_patterns['html_tags'].sub('', cleaned_text)
                cleaning_applied.append('html_removal')

            # Handle URLs (preserve or remove based on aggressiveness)
            if self.cleaning_patterns['urls'].search(cleaned_text):
                if aggressive:
                    cleaned_text = self.cleaning_patterns['urls'].sub('[URL]', cleaned_text)
                    cleaning_applied.append('url_replacement')
                else:
                    # Keep URLs but ensure they don't break text flow
                    urls = self.cleaning_patterns['urls'].findall(cleaned_text)
                    for url in urls:
                        if len(url) > 50:  # Replace very long URLs
                            cleaned_text = cleaned_text.replace(url, '[URL]')
                    cleaning_applied.append('long_url_replacement')

            # Handle email addresses
            if self.cleaning_patterns['email_addresses'].search(cleaned_text):
                if aggressive:
                    cleaned_text = self.cleaning_patterns['email_addresses'].sub('[EMAIL]', cleaned_text)
                    cleaning_applied.append('email_replacement')

            # Fix repeated characters (e.g., "soooo good" -> "so good")
            if self.cleaning_patterns['repeated_chars'].search(cleaned_text):
                cleaned_text = self.cleaning_patterns['repeated_chars'].sub(r'\1\1', cleaned_text)
                cleaning_applied.append('repeated_char_reduction')

            # Remove excessive special characters (if aggressive)
            if aggressive:
                # Count special character ratio
                special_char_count = len(self.cleaning_patterns['special_chars'].findall(cleaned_text))
                total_chars = len(cleaned_text)
                if total_chars > 0 and (special_char_count / total_chars) > self.quality_thresholds['max_special_char_ratio']:
                    cleaned_text = self.cleaning_patterns['special_chars'].sub('', cleaned_text)
                    cleaning_applied.append('special_char_removal')

            # Normalize whitespace
            cleaned_text = self.cleaning_patterns['extra_whitespace'].sub(' ', cleaned_text)
            cleaned_text = cleaned_text.strip()
            cleaning_applied.append('whitespace_normalization')

            # Calculate quality score
            quality_score = self._calculate_text_quality(cleaned_text, original_text)

            result = {
                'cleaned_text': cleaned_text,
                'original_text': original_text,
                'cleaning_applied': cleaning_applied,
                'quality_score': quality_score,
                'length_change': len(cleaned_text) - len(original_text),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            return result

        except Exception as e:
            logger.error(f"Text cleaning failed: {str(e)}")
            return {
                'cleaned_text': original_text,
                'original_text': original_text,
                'cleaning_applied': [],
                'quality_score': 0.5,
                'error': str(e),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }

    def validate_data(self, data: Dict) -> Dict:
        """
        Validate data quality and structure

        Args:
            data: Data dictionary to validate

        Returns:
            Dict containing validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'suggestions': [],
            'quality_score': 1.0
        }

        try:
            # Check required fields
            required_fields = ['text']
            for field in required_fields:
                if field not in data or not data[field]:
                    validation_result['issues'].append(f"Missing required field: {field}")
                    validation_result['is_valid'] = False

            if 'text' in data:
                text = data['text']

                # Text length validation
                if len(text) < self.quality_thresholds['min_text_length']:
                    validation_result['issues'].append(f"Text too short ({len(text)} chars)")
                    validation_result['is_valid'] = False
                elif len(text) > self.quality_thresholds['max_text_length']:
                    validation_result['warnings'].append(f"Text very long ({len(text)} chars)")

                # Word count validation
                word_count = len(text.split())
                if word_count < self.quality_thresholds['min_word_count']:
                    validation_result['issues'].append(f"Too few words ({word_count})")
                    validation_result['is_valid'] = False

                # Character encoding validation
                try:
                    text.encode('utf-8')
                except UnicodeEncodeError:
                    validation_result['issues'].append("Text contains invalid characters")
                    validation_result['is_valid'] = False

                # Detect potential spam patterns
                spam_indicators = self._detect_spam_patterns(text)
                if spam_indicators:
                    validation_result['warnings'].extend(spam_indicators)

            # Validate optional fields
            if 'sentiment' in data:
                valid_sentiments = ['positive', 'neutral', 'negative']
                if data['sentiment'] not in valid_sentiments:
                    validation_result['issues'].append(f"Invalid sentiment: {data['sentiment']}")

            if 'urgency_score' in data:
                score = data['urgency_score']
                if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                    validation_result['issues'].append(f"Invalid urgency score: {score}")

            # Calculate overall quality score
            validation_result['quality_score'] = self._calculate_data_quality_score(validation_result)

        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            validation_result['issues'].append(f"Validation error: {str(e)}")
            validation_result['is_valid'] = False

        return validation_result

    def batch_process_data(self, data_items: List[Dict],
                           operations: List[str] = None,
                           max_workers: int = 4) -> Dict:
        """
        Process multiple data items in batch with parallel processing

        Args:
            data_items: List of data dictionaries to process
            operations: List of operations to perform ['clean', 'validate', 'extract_features']
            max_workers: Maximum number of worker threads

        Returns:
            Dict containing batch processing results
        """
        if not data_items:
            return {'processed_items': [], 'summary': {'total': 0, 'successful': 0, 'failed': 0}}

        if operations is None:
            operations = ['clean', 'validate']

        start_time = time.time()
        processed_items = []
        failed_count = 0

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all processing tasks
                future_to_item = {
                    executor.submit(self._process_single_item, item, operations): i
                    for i, item in enumerate(data_items)
                }

                # Collect results as they complete
                for future in as_completed(future_to_item):
                    item_index = future_to_item[future]
                    try:
                        result = future.result()
                        result['original_index'] = item_index
                        processed_items.append(result)
                    except Exception as e:
                        logger.error(f"Failed to process item {item_index}: {str(e)}")
                        failed_count += 1
                        processed_items.append({
                            'original_index': item_index,
                            'error': str(e),
                            'processing_successful': False
                        })

            # Sort by original index
            processed_items.sort(key=lambda x: x.get('original_index', 0))

            processing_time = time.time() - start_time

            summary = {
                'total': len(data_items),
                'successful': len(data_items) - failed_count,
                'failed': failed_count,
                'processing_time_seconds': round(processing_time, 2),
                'items_per_second': round(len(data_items) / processing_time, 2),
                'operations_performed': operations
            }

            return {
                'processed_items': processed_items,
                'summary': summary
            }

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return {
                'processed_items': [],
                'summary': {
                    'total': len(data_items),
                    'successful': 0,
                    'failed': len(data_items),
                    'error': str(e)
                }
            }

    def _process_single_item(self, item: Dict, operations: List[str]) -> Dict:
        """Process a single data item with specified operations"""
        result = {
            'original_data': item,
            'processing_successful': True,
            'operations_performed': []
        }

        current_data = item.copy()

        try:
            # Text cleaning
            if 'clean' in operations and 'text' in current_data:
                cleaning_result = self.clean_text(current_data['text'])
                current_data['text'] = cleaning_result['cleaned_text']
                result['cleaning_result'] = cleaning_result
                result['operations_performed'].append('clean')

            # Data validation
            if 'validate' in operations:
                validation_result = self.validate_data(current_data)
                result['validation_result'] = validation_result
                result['operations_performed'].append('validate')

            # Feature extraction
            if 'extract_features' in operations and 'text' in current_data:
                features = self.extract_text_features(current_data['text'])
                result['text_features'] = features
                result['operations_performed'].append('extract_features')

            result['processed_data'] = current_data

        except Exception as e:
            result['processing_successful'] = False
            result['error'] = str(e)

        return result

    def extract_text_features(self, text: str) -> Dict:
        """
        Extract various features from text for analysis

        Args:
            text: Input text to analyze

        Returns:
            Dict containing extracted features
        """
        if not text:
            return {}

        try:
            # Basic statistics
            word_count = len(text.split())
            char_count = len(text)
            char_count_no_spaces = len(text.replace(' ', ''))
            sentence_count = len(re.split(r'[.!?]+', text))

            # Punctuation analysis
            punctuation_chars = '.,!?;:\'"-()[]{}/*&^%$#@'
            punctuation_count = sum(1 for char in text if char in punctuation_chars)

            # Capitalization analysis
            uppercase_count = sum(1 for char in text if char.isupper())
            lowercase_count = sum(1 for char in text if char.islower())

            # Special pattern detection
            email_count = len(self.validation_patterns['email'].findall(text))
            url_count = len(self.validation_patterns['url'].findall(text))
            phone_count = len(self.validation_patterns['phone'].findall(text))
            social_handle_count = len(self.validation_patterns['social_handle'].findall(text))
            hashtag_count = len(self.validation_patterns['hashtag'].findall(text))

            # Readability metrics
            avg_words_per_sentence = word_count / max(sentence_count, 1)
            avg_chars_per_word = char_count_no_spaces / max(word_count, 1)

            # Language complexity indicators
            unique_words = len(set(text.lower().split()))
            lexical_diversity = unique_words / max(word_count, 1)

            # Emotional indicators
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_word_count = sum(1 for word in text.split() if word.isupper() and len(word) > 1)

            features = {
                'word_count': word_count,
                'char_count': char_count,
                'char_count_no_spaces': char_count_no_spaces,
                'sentence_count': sentence_count,
                'punctuation_count': punctuation_count,
                'uppercase_count': uppercase_count,
                'lowercase_count': lowercase_count,
                'email_count': email_count,
                'url_count': url_count,
                'phone_count': phone_count,
                'social_handle_count': social_handle_count,
                'hashtag_count': hashtag_count,
                'avg_words_per_sentence': round(avg_words_per_sentence, 2),
                'avg_chars_per_word': round(avg_chars_per_word, 2),
                'unique_words': unique_words,
                'lexical_diversity': round(lexical_diversity, 3),
                'exclamation_count': exclamation_count,
                'question_count': question_count,
                'caps_word_count': caps_word_count,
                'punctuation_ratio': round(punctuation_count / max(char_count, 1), 3),
                'uppercase_ratio': round(uppercase_count / max(char_count, 1), 3)
            }

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return {'error': str(e)}

    def detect_duplicates(self, data_items: List[Dict],
                          similarity_threshold: float = 0.9) -> Dict:
        """
        Detect duplicate or near-duplicate entries

        Args:
            data_items: List of data items to check
            similarity_threshold: Threshold for considering items similar

        Returns:
            Dict containing duplicate detection results
        """
        if len(data_items) < 2:
            return {'duplicates': [], 'unique_count': len(data_items), 'duplicate_count': 0}

        try:
            duplicates = []
            processed_texts = []

            for i, item in enumerate(data_items):
                text = item.get('text', '').strip().lower()
                if not text:
                    continue

                # Simple hash-based exact duplicate detection
                text_hash = hashlib.md5(text.encode()).hexdigest()

                # Check for exact matches
                exact_matches = [
                    j for j, prev_item in enumerate(data_items[:i])
                    if hashlib.md5(prev_item.get('text', '').strip().lower().encode()).hexdigest() == text_hash
                ]

                if exact_matches:
                    duplicates.append({
                        'type': 'exact',
                        'indices': [exact_matches[0], i],
                        'similarity': 1.0,
                        'text_preview': text[:100] + '...' if len(text) > 100 else text
                    })

                # For similarity-based detection, we'd need more sophisticated algorithms
                # This is a simplified version for exact matches

                processed_texts.append(text_hash)

            unique_count = len(data_items) - len(duplicates)
            duplicate_count = len(duplicates)

            return {
                'duplicates': duplicates,
                'unique_count': unique_count,
                'duplicate_count': duplicate_count,
                'duplicate_percentage': round((duplicate_count / len(data_items)) * 100, 2),
                'detection_method': 'hash_based_exact_match'
            }

        except Exception as e:
            logger.error(f"Duplicate detection failed: {str(e)}")
            return {'error': str(e), 'duplicates': [], 'unique_count': 0, 'duplicate_count': 0}

    def export_processed_data(self, processed_data: List[Dict],
                              output_path: str,
                              format_type: str = 'csv') -> Dict:
        """
        Export processed data to various formats

        Args:
            processed_data: List of processed data items
            output_path: Output file path
            format_type: Export format ('csv', 'json', 'excel')

        Returns:
            Dict containing export results
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format_type.lower() == 'csv':
                return self._export_to_csv(processed_data, output_path)
            elif format_type.lower() == 'json':
                return self._export_to_json(processed_data, output_path)
            elif format_type.lower() == 'excel':
                return self._export_to_excel(processed_data, output_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            logger.error(f"Data export failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _export_to_csv(self, data: List[Dict], output_path: Path) -> Dict:
        """Export data to CSV format"""
        try:
            # Flatten nested dictionaries for CSV export
            flattened_data = []
            for item in data:
                flat_item = self._flatten_dict(item)
                flattened_data.append(flat_item)

            if flattened_data:
                df = pd.DataFrame(flattened_data)
                df.to_csv(output_path, index=False, encoding='utf-8')

                return {
                    'success': True,
                    'file_path': str(output_path),
                    'records_exported': len(flattened_data),
                    'columns': list(df.columns),
                    'file_size_mb': round(output_path.stat().st_size / (1024 * 1024), 2)
                }
            else:
                return {'success': False, 'error': 'No data to export'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _export_to_json(self, data: List[Dict], output_path: Path) -> Dict:
        """Export data to JSON format"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            return {
                'success': True,
                'file_path': str(output_path),
                'records_exported': len(data),
                'file_size_mb': round(output_path.stat().st_size / (1024 * 1024), 2)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _export_to_excel(self, data: List[Dict], output_path: Path) -> Dict:
        """Export data to Excel format"""
        try:
            flattened_data = [self._flatten_dict(item) for item in data]

            if flattened_data:
                df = pd.DataFrame(flattened_data)
                df.to_excel(output_path, index=False, engine='openpyxl')

                return {
                    'success': True,
                    'file_path': str(output_path),
                    'records_exported': len(flattened_data),
                    'columns': list(df.columns),
                    'file_size_mb': round(output_path.stat().st_size / (1024 * 1024), 2)
                }
            else:
                return {'success': False, 'error': 'No data to export'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for CSV/Excel export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _calculate_text_quality(self, cleaned_text: str, original_text: str) -> float:
        """Calculate text quality score"""
        try:
            if not cleaned_text:
                return 0.0

            # Length score (penalty for too short or too long)
            length_score = 1.0
            if len(cleaned_text) < 10:
                length_score = len(cleaned_text) / 10
            elif len(cleaned_text) > 1000:
                length_score = 1000 / len(cleaned_text)

            # Word diversity score
            words = cleaned_text.lower().split()
            unique_words = len(set(words))
            diversity_score = min(unique_words / max(len(words), 1), 1.0)

            # Character diversity score
            unique_chars = len(set(cleaned_text.lower()))
            char_diversity_score = min(unique_chars / 26, 1.0)  # Normalized to alphabet

            # Cleaning impact score (penalize excessive cleaning)
            original_len = len(original_text)
            cleaned_len = len(cleaned_text)
            if original_len > 0:
                preservation_score = cleaned_len / original_len
                preservation_score = min(preservation_score, 1.0)
            else:
                preservation_score = 0.0

            # Combine scores
            quality_score = (
                    length_score * 0.3 +
                    diversity_score * 0.3 +
                    char_diversity_score * 0.2 +
                    preservation_score * 0.2
            )

            return round(quality_score, 3)

        except Exception:
            return 0.5  # Default medium quality

    def _calculate_data_quality_score(self, validation_result: Dict) -> float:
        """Calculate overall data quality score from validation results"""
        base_score = 1.0

        # Deduct for issues
        issue_count = len(validation_result.get('issues', []))
        base_score -= issue_count * 0.2

        # Small deduction for warnings
        warning_count = len(validation_result.get('warnings', []))
        base_score -= warning_count * 0.05

        return max(base_score, 0.0)

    def _detect_spam_patterns(self, text: str) -> List[str]:
        """Detect potential spam patterns in text"""
        indicators = []

        # Excessive capitalization
        if len(text) > 10:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.5:
                indicators.append("Excessive capitalization detected")

        # Excessive punctuation
        punct_count = sum(1 for c in text if c in '!?.')
        if punct_count > len(text.split()) * 0.5:
            indicators.append("Excessive punctuation detected")

        # Repeated words
        words = text.lower().split()
        word_counts = Counter(words)
        for word, count in word_counts.items():
            if len(word) > 3 and count > len(words) * 0.3:
                indicators.append(f"Repeated word pattern: '{word}'")

        # Suspicious patterns
        spam_patterns = [
            r'click here',
            r'free money',
            r'win now',
            r'call now',
            r'act fast',
            r'limited time'
        ]

        text_lower = text.lower()
        for pattern in spam_patterns:
            if re.search(pattern, text_lower):
                indicators.append(f"Spam pattern detected: {pattern}")

        return indicators

    def get_processing_stats(self) -> Dict:
        """Get processing statistics and cache information"""
        return {
            'cache_size': len(self._processing_cache),
            'cache_directory': str(self.cache_dir),
            'validation_patterns_count': len(self.validation_patterns),
            'cleaning_patterns_count': len(self.cleaning_patterns),
            'quality_thresholds': self.quality_thresholds
        }

    def clear_cache(self):
        """Clear processing cache"""
        with self._lock:
            self._processing_cache.clear()
        logger.info("Data processor cache cleared")
