"""Urgency Scoring Model for Customer Feedback - AMIL Project"""

import re
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import math

logger = logging.getLogger(__name__)

class UrgencyScorer:
    """Advanced urgency scoring system for customer feedback analysis"""

    def __init__(self, high_threshold: float = 0.8, medium_threshold: float = 0.5):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

        # Critical keywords that indicate high urgency
        self.critical_keywords = {
            'system_failure': {
                'keywords': ['crash', 'broken', 'not working', 'failed', 'error', 'bug', 'freeze', 'hang'],
                'weight': 0.9,
                'description': 'System failures requiring immediate attention'
            },
            'security_issues': {
                'keywords': ['security', 'hack', 'breach', 'unsafe', 'vulnerability', 'compromised'],
                'weight': 1.0,
                'description': 'Security-related concerns'
            },
            'data_loss': {
                'keywords': ['lost data', 'deleted', 'missing', 'corrupted', 'disappeared'],
                'weight': 0.95,
                'description': 'Data loss or corruption issues'
            },
            'blocking_issues': {
                'keywords': ['cant access', 'blocked', 'stuck', 'unable to', 'impossible'],
                'weight': 0.8,
                'description': 'Issues that completely block user progress'
            },
            'urgent_requests': {
                'keywords': ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'deadline'],
                'weight': 0.85,
                'description': 'Explicitly urgent requests'
            },
            'customer_impact': {
                'keywords': ['angry', 'frustrated', 'disappointed', 'terrible', 'horrible', 'worst'],
                'weight': 0.7,
                'description': 'High negative customer impact'
            }
        }

        # Moderate urgency indicators
        self.moderate_keywords = {
            'performance_issues': {
                'keywords': ['slow', 'laggy', 'delayed', 'timeout', 'loading', 'performance'],
                'weight': 0.6,
                'description': 'Performance-related issues'
            },
            'usability_problems': {
                'keywords': ['confusing', 'difficult', 'hard to use', 'unclear', 'complicated'],
                'weight': 0.5,
                'description': 'Usability and user experience issues'
            },
            'feature_requests': {
                'keywords': ['need', 'want', 'request', 'suggestion', 'improvement', 'enhance'],
                'weight': 0.4,
                'description': 'Feature requests and improvements'
            },
            'minor_bugs': {
                'keywords': ['minor', 'small issue', 'cosmetic', 'typo', 'alignment'],
                'weight': 0.3,
                'description': 'Minor issues and cosmetic problems'
            }
        }

        # Urgency modifiers based on context
        self.context_modifiers = {
            'event_timing': {
                'before_event': ['before', 'upcoming', 'next week', 'soon'],
                'during_event': ['now', 'currently', 'happening', 'live'],
                'after_event': ['happened', 'was', 'yesterday', 'last week'],
                'weights': {'before_event': 1.2, 'during_event': 1.5, 'after_event': 0.8}
            },
            'audience_size': {
                'large_audience': ['everyone', 'all participants', 'entire group', 'whole team'],
                'small_audience': ['just me', 'only I', 'few people', 'some users'],
                'weights': {'large_audience': 1.3, 'small_audience': 0.9}
            },
            'frequency': {
                'recurring': ['always', 'every time', 'constantly', 'repeatedly'],
                'occasional': ['sometimes', 'occasionally', 'rarely', 'once'],
                'weights': {'recurring': 1.2, 'occasional': 0.8}
            }
        }

        # Sentiment impact on urgency
        self.sentiment_multipliers = {
            'positive': 0.7,    # Positive feedback is generally less urgent
            'neutral': 1.0,     # Neutral baseline
            'negative': 1.4     # Negative feedback increases urgency
        }

    def calculate_urgency(self,
                          text: str,
                          sentiment: Optional[str] = None,
                          metadata: Optional[Dict] = None) -> Dict:
        """
        Calculate urgency score for feedback text

        Args:
            text: Feedback text to analyze
            sentiment: Optional sentiment analysis result
            metadata: Optional metadata (event timing, user info, etc.)

        Returns:
            Dict containing urgency analysis results
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        start_time = time.time()
        text_lower = text.lower()

        try:
            # Base urgency calculation
            base_score = self._calculate_base_urgency(text_lower)

            # Apply context modifiers
            context_multiplier = self._calculate_context_multiplier(text_lower, metadata)

            # Apply sentiment multiplier
            sentiment_multiplier = self.sentiment_multipliers.get(sentiment, 1.0)

            # Calculate final urgency score
            final_score = min(base_score * context_multiplier * sentiment_multiplier, 1.0)

            # Determine urgency level
            urgency_level = self._determine_urgency_level(final_score)

            # Extract key indicators
            indicators = self._extract_urgency_indicators(text_lower)

            # Calculate confidence
            confidence = self._calculate_confidence(indicators, final_score)

            # Generate recommendations
            recommendations = self._generate_recommendations(urgency_level, indicators)

            result = {
                'score': round(final_score, 3),
                'level': urgency_level,
                'confidence': round(confidence, 3),
                'indicators': indicators,
                'recommendations': recommendations,
                'score_breakdown': {
                    'base_score': round(base_score, 3),
                    'context_multiplier': round(context_multiplier, 3),
                    'sentiment_multiplier': round(sentiment_multiplier, 3),
                    'final_score': round(final_score, 3)
                },
                'thresholds': {
                    'high': self.high_threshold,
                    'medium': self.medium_threshold
                },
                'inference_time_ms': round((time.time() - start_time) * 1000, 2)
            }

            return result

        except Exception as e:
            logger.error(f"Urgency calculation failed: {str(e)}")
            return {
                'score': 0.0,
                'level': 'low',
                'confidence': 0.0,
                'indicators': [],
                'recommendations': [],
                'error': str(e),
                'inference_time_ms': round((time.time() - start_time) * 1000, 2)
            }

    def _calculate_base_urgency(self, text_lower: str) -> float:
        """Calculate base urgency score from keyword analysis"""
        total_score = 0.0
        max_possible_score = 0.0

        # Check critical keywords
        for category, config in self.critical_keywords.items():
            keywords = config['keywords']
            weight = config['weight']
            matches = sum(1 for keyword in keywords if keyword in text_lower)

            if matches > 0:
                # Score based on number of matches and category weight
                category_score = min(matches / len(keywords), 1.0) * weight
                total_score += category_score

            max_possible_score += weight

        # Check moderate keywords
        for category, config in self.moderate_keywords.items():
            keywords = config['keywords']
            weight = config['weight']
            matches = sum(1 for keyword in keywords if keyword in text_lower)

            if matches > 0:
                category_score = min(matches / len(keywords), 1.0) * weight
                total_score += category_score

            max_possible_score += weight

        # Normalize score
        if max_possible_score > 0:
            normalized_score = total_score / max_possible_score
        else:
            normalized_score = 0.0

        return min(normalized_score, 1.0)

    def _calculate_context_multiplier(self, text_lower: str, metadata: Optional[Dict]) -> float:
        """Calculate context-based multiplier"""
        multiplier = 1.0

        # Event timing context
        timing_weights = self.context_modifiers['event_timing']['weights']
        for timing, keywords in self.context_modifiers['event_timing'].items():
            if timing == 'weights':
                continue

            if any(keyword in text_lower for keyword in keywords):
                multiplier *= timing_weights[timing]
                break

        # Audience size context
        audience_weights = self.context_modifiers['audience_size']['weights']
        for audience, keywords in self.context_modifiers['audience_size'].items():
            if audience == 'weights':
                continue

            if any(keyword in text_lower for keyword in keywords):
                multiplier *= audience_weights[audience]
                break

        # Frequency context
        frequency_weights = self.context_modifiers['frequency']['weights']
        for frequency, keywords in self.context_modifiers['frequency'].items():
            if frequency == 'weights':
                continue

            if any(keyword in text_lower for keyword in keywords):
                multiplier *= frequency_weights[frequency]
                break

        # Metadata-based adjustments
        if metadata:
            # Event proximity
            if metadata.get('days_until_event', 0) <= 1:
                multiplier *= 1.5
            elif metadata.get('days_until_event', 0) <= 7:
                multiplier *= 1.2

            # User role (organizers get higher priority)
            if metadata.get('user_role') == 'organizer':
                multiplier *= 1.3
            elif metadata.get('user_role') == 'speaker':
                multiplier *= 1.2

        return multiplier

    def _determine_urgency_level(self, score: float) -> str:
        """Determine urgency level based on score"""
        if score >= self.high_threshold:
            return 'high'
        elif score >= self.medium_threshold:
            return 'medium'
        else:
            return 'low'

    def _extract_urgency_indicators(self, text_lower: str) -> List[Dict]:
        """Extract specific urgency indicators found in text"""
        indicators = []

        # Check all keyword categories
        all_categories = {**self.critical_keywords, **self.moderate_keywords}

        for category, config in all_categories.items():
            keywords = config['keywords']
            weight = config['weight']
            description = config['description']

            found_keywords = [kw for kw in keywords if kw in text_lower]

            if found_keywords:
                indicators.append({
                    'category': category,
                    'keywords_found': found_keywords,
                    'weight': weight,
                    'description': description,
                    'match_count': len(found_keywords)
                })

        # Sort by weight (highest first)
        indicators.sort(key=lambda x: x['weight'], reverse=True)

        return indicators

    def _calculate_confidence(self, indicators: List[Dict], final_score: float) -> float:
        """Calculate confidence in urgency assessment"""
        if not indicators:
            return 0.1  # Low confidence with no indicators

        # Base confidence from number of indicators
        indicator_confidence = min(len(indicators) / 3, 1.0)

        # Confidence from indicator weights
        weight_confidence = sum(ind['weight'] for ind in indicators[:3]) / 3

        # Confidence from score consistency
        if final_score > 0.8:
            score_confidence = 0.9
        elif final_score > 0.5:
            score_confidence = 0.7
        else:
            score_confidence = 0.5

        # Combined confidence
        combined_confidence = (indicator_confidence + weight_confidence + score_confidence) / 3

        return min(combined_confidence, 1.0)

    def _generate_recommendations(self, urgency_level: str, indicators: List[Dict]) -> List[str]:
        """Generate action recommendations based on urgency analysis"""
        recommendations = []

        if urgency_level == 'high':
            recommendations.extend([
                "🚨 Immediate attention required - respond within 1 hour",
                "📞 Consider direct phone/chat contact for faster resolution",
                "📋 Escalate to senior team members or management",
                "📊 Track resolution time and follow up within 24 hours"
            ])
        elif urgency_level == 'medium':
            recommendations.extend([
                "⏰ Respond within 4-6 hours during business hours",
                "📝 Acknowledge receipt and provide timeline for resolution",
                "🔍 Investigate root cause and document findings",
                "📈 Monitor for similar issues from other users"
            ])
        else:  # low urgency
            recommendations.extend([
                "📅 Respond within 24-48 hours",
                "💭 Consider batch processing with similar feedback",
                "📚 Document for future improvements or FAQ updates",
                "🔄 Include in regular review cycles"
            ])

        # Add specific recommendations based on indicators
        for indicator in indicators[:2]:  # Top 2 indicators
            category = indicator['category']

            if category == 'system_failure':
                recommendations.append("🔧 Immediately check system status and logs")
            elif category == 'security_issues':
                recommendations.append("🔒 Involve security team and assess risk level")
            elif category == 'data_loss':
                recommendations.append("💾 Check backup systems and data recovery options")
            elif category == 'performance_issues':
                recommendations.append("⚡ Run performance diagnostics and monitoring")
            elif category == 'usability_problems':
                recommendations.append("👥 Consider user testing and UX review")

        return list(set(recommendations))  # Remove duplicates

    def analyze_batch(self, feedback_items: List[Dict]) -> List[Dict]:
        """Analyze urgency for multiple feedback items"""
        if not feedback_items:
            return []

        start_time = time.time()
        results = []

        try:
            for i, item in enumerate(feedback_items):
                try:
                    text = item.get('text', '')
                    sentiment = item.get('sentiment')
                    metadata = item.get('metadata', {})

                    if text and text.strip():
                        result = self.calculate_urgency(text, sentiment, metadata)
                        result['batch_index'] = i
                        result['original_id'] = item.get('id', i)
                    else:
                        result = {
                            'score': 0.0,
                            'level': 'low',
                            'batch_index': i,
                            'original_id': item.get('id', i),
                            'error': 'Empty or invalid text'
                        }

                    results.append(result)

                except Exception as e:
                    results.append({
                        'score': 0.0,
                        'level': 'low',
                        'batch_index': i,
                        'original_id': item.get('id', i),
                        'error': str(e)
                    })

            # Sort by urgency score (highest first)
            results.sort(key=lambda x: x.get('score', 0), reverse=True)

            total_time = time.time() - start_time
            logger.info(f"Batch urgency analysis: {len(feedback_items)} items in {total_time:.2f}s")

        except Exception as e:
            logger.error(f"Batch urgency analysis failed: {str(e)}")

        return results

    def get_urgency_statistics(self, results: List[Dict]) -> Dict:
        """Get statistics from urgency analysis results"""
        if not results:
            return {'total': 0, 'high': 0, 'medium': 0, 'low': 0}

        stats = {
            'total': len(results),
            'high': sum(1 for r in results if r.get('level') == 'high'),
            'medium': sum(1 for r in results if r.get('level') == 'medium'),
            'low': sum(1 for r in results if r.get('level') == 'low'),
            'average_score': np.mean([r.get('score', 0) for r in results]),
            'score_distribution': {
                '0.0-0.2': sum(1 for r in results if 0.0 <= r.get('score', 0) < 0.2),
                '0.2-0.4': sum(1 for r in results if 0.2 <= r.get('score', 0) < 0.4),
                '0.4-0.6': sum(1 for r in results if 0.4 <= r.get('score', 0) < 0.6),
                '0.6-0.8': sum(1 for r in results if 0.6 <= r.get('score', 0) < 0.8),
                '0.8-1.0': sum(1 for r in results if 0.8 <= r.get('score', 0) <= 1.0)
            }
        }

        # Calculate percentages
        total = stats['total']
        stats['percentages'] = {
            'high': round((stats['high'] / total) * 100, 1) if total > 0 else 0,
            'medium': round((stats['medium'] / total) * 100, 1) if total > 0 else 0,
            'low': round((stats['low'] / total) * 100, 1) if total > 0 else 0
        }

        return stats

    def benchmark_performance(self, test_texts: List[str] = None) -> Dict:
        """Benchmark urgency scoring performance"""
        if test_texts is None:
            test_texts = [
                "The system crashed during the presentation! This is urgent!",
                "Everything is broken and not working at all. We need help immediately!",
                "The workshop was okay but could be improved.",
                "Great event! Really enjoyed the content and speakers.",
                "Minor bug in the UI - button alignment is slightly off",
                "Can't access my account and lost all my work. This is terrible!",
                "The slides were hard to read and the speaker was unclear",
                "Amazing experience! Well organized and informative.",
                "Need to fix the wifi connection before tomorrow's event",
                "Performance is very slow and timing out frequently"
            ]

        start_time = time.time()
        results = []

        for text in test_texts:
            result = self.calculate_urgency(text)
            results.append(result)

        total_time = time.time() - start_time
        avg_time = total_time / len(test_texts)

        # Calculate statistics
        stats = self.get_urgency_statistics(results)

        return {
            'total_texts': len(test_texts),
            'total_time_seconds': round(total_time, 3),
            'average_time_per_text_ms': round(avg_time * 1000, 2),
            'throughput_texts_per_second': round(len(test_texts) / total_time, 2),
            'urgency_distribution': stats['percentages'],
            'average_urgency_score': round(stats['average_score'], 3),
            'sample_results': results[:3]
        }
