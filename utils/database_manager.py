"""Database Manager for AI Customer Feedback Analyzer - AMIL Project"""

import sqlite3
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Advanced database manager with optimized queries and caching"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or "database/feedback_analytics.db"
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()

        # Cache for frequently accessed data
        self._cache = {}
        self._cache_expiry = {}
        self._cache_ttl = 300  # 5 minutes

        self.initialize_database()

    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row

            # Optimize SQLite settings
            cursor = self._local.connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.execute("PRAGMA cache_size = 10000")
            cursor.execute("PRAGMA temp_store = MEMORY")
            cursor.close()

        return self._local.connection

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            cursor.close()

    def initialize_database(self):
        """Initialize database schema"""
        logger.info("🗄️  Initializing database schema...")

        try:
            with self.get_cursor() as cursor:
                # Feedback analysis table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS feedback_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        text_hash TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        result TEXT NOT NULL,
                        language TEXT DEFAULT 'auto',
                        sentiment_label TEXT,
                        sentiment_score REAL,
                        urgency_score REAL,
                        urgency_level TEXT,
                        themes TEXT,
                        confidence REAL,
                        inference_time_ms REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Analytics summary table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        total_feedback INTEGER DEFAULT 0,
                        positive_count INTEGER DEFAULT 0,
                        neutral_count INTEGER DEFAULT 0,
                        negative_count INTEGER DEFAULT 0,
                        high_urgency_count INTEGER DEFAULT 0,
                        medium_urgency_count INTEGER DEFAULT 0,
                        low_urgency_count INTEGER DEFAULT 0,
                        average_sentiment_score REAL DEFAULT 0.0,
                        average_urgency_score REAL DEFAULT 0.0,
                        top_themes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date)
                    )
                """)

                # GDG actions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS gdg_actions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        feedback_id INTEGER,
                        action_title TEXT NOT NULL,
                        action_description TEXT,
                        priority TEXT NOT NULL,
                        category TEXT,
                        status TEXT DEFAULT 'pending',
                        assigned_to TEXT,
                        due_date TIMESTAMP,
                        completed_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (feedback_id) REFERENCES feedback_analysis (id)
                    )
                """)

                # System performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation_type TEXT NOT NULL,
                        execution_time_ms REAL NOT NULL,
                        gpu_used BOOLEAN DEFAULT FALSE,
                        memory_usage_mb REAL,
                        error_occurred BOOLEAN DEFAULT FALSE,
                        error_message TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback_analysis(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_sentiment ON feedback_analysis(sentiment_label)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_urgency ON feedback_analysis(urgency_level)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_hash ON feedback_analysis(text_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_priority ON gdg_actions(priority)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_status ON gdg_actions(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON system_performance(timestamp)")

            logger.info("✅ Database schema initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {str(e)}")
            raise

    def store_analysis(self,
                       text: str,
                       analysis_type: str,
                       result: Dict,
                       language: str = 'auto') -> int:
        """Store analysis result in database"""
        try:
            # Create hash for duplicate detection
            text_hash = str(hash(text.strip().lower()))

            # Extract key fields from result
            sentiment_label = None
            sentiment_score = None
            urgency_score = None
            urgency_level = None
            themes = None
            confidence = result.get('confidence', 0.0)
            inference_time = result.get('inference_time_ms', 0.0)

            if analysis_type == 'sentiment':
                sentiment_label = result.get('label')
                sentiment_score = result.get('score', 0.0)
            elif analysis_type == 'urgency':
                urgency_score = result.get('score', 0.0)
                urgency_level = result.get('level')
            elif analysis_type == 'themes':
                themes = json.dumps(result.get('topics', []))

            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO feedback_analysis (
                        text, text_hash, analysis_type, result, language,
                        sentiment_label, sentiment_score, urgency_score, 
                        urgency_level, themes, confidence, inference_time_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    text, text_hash, analysis_type, json.dumps(result), language,
                    sentiment_label, sentiment_score, urgency_score,
                    urgency_level, themes, confidence, inference_time
                ))

                return cursor.lastrowid

        except Exception as e:
            logger.error(f"Failed to store analysis: {str(e)}")
            return 0

    def get_analysis_history(self,
                             limit: int = 100,
                             analysis_type: Optional[str] = None,
                             days: int = 30) -> List[Dict]:
        """Get analysis history with optional filtering"""
        cache_key = f"history_{limit}_{analysis_type}_{days}"

        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        try:
            with self.get_cursor() as cursor:
                query = """
                    SELECT * FROM feedback_analysis 
                    WHERE created_at >= datetime('now', '-{} days')
                """.format(days)

                params = []

                if analysis_type:
                    query += " AND analysis_type = ?"
                    params.append(analysis_type)

                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    result = dict(row)
                    # Parse JSON fields
                    if result['result']:
                        result['result'] = json.loads(result['result'])
                    if result['themes']:
                        result['themes'] = json.loads(result['themes'])
                    results.append(result)

                # Cache the result
                self._cache_result(cache_key, results)

                return results

        except Exception as e:
            logger.error(f"Failed to get analysis history: {str(e)}")
            return []

    def get_analytics_summary(self, days: int = 30) -> Dict:
        """Get comprehensive analytics summary"""
        cache_key = f"summary_{days}"

        # Check cache
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        try:
            with self.get_cursor() as cursor:
                # Get basic counts
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_feedback,
                        COUNT(CASE WHEN sentiment_label = 'positive' THEN 1 END) as positive_count,
                        COUNT(CASE WHEN sentiment_label = 'neutral' THEN 1 END) as neutral_count,
                        COUNT(CASE WHEN sentiment_label = 'negative' THEN 1 END) as negative_count,
                        COUNT(CASE WHEN urgency_level = 'high' THEN 1 END) as high_urgency,
                        COUNT(CASE WHEN urgency_level = 'medium' THEN 1 END) as medium_urgency,
                        COUNT(CASE WHEN urgency_level = 'low' THEN 1 END) as low_urgency,
                        AVG(sentiment_score) as avg_sentiment,
                        AVG(urgency_score) as avg_urgency,
                        AVG(inference_time_ms) as avg_inference_time
                    FROM feedback_analysis 
                    WHERE created_at >= datetime('now', '-{} days')
                """.format(days))

                summary_row = cursor.fetchone()

                # Get daily trends
                cursor.execute("""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as count,
                        AVG(sentiment_score) as avg_sentiment,
                        AVG(urgency_score) as avg_urgency
                    FROM feedback_analysis 
                    WHERE created_at >= datetime('now', '-{} days')
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                """.format(days))

                daily_trends = [dict(row) for row in cursor.fetchall()]

                # Get top themes
                cursor.execute("""
                    SELECT themes, COUNT(*) as frequency
                    FROM feedback_analysis 
                    WHERE themes IS NOT NULL 
                    AND created_at >= datetime('now', '-{} days')
                    GROUP BY themes
                    ORDER BY frequency DESC
                    LIMIT 10
                """.format(days))

                themes_data = cursor.fetchall()
                top_themes = []
                for row in themes_data:
                    try:
                        themes = json.loads(row['themes'])
                        for theme in themes:
                            top_themes.append({'theme': theme, 'frequency': row['frequency']})
                    except:
                        continue

                # Build summary
                summary = dict(summary_row) if summary_row else {}
                summary.update({
                    'daily_trends': daily_trends,
                    'top_themes': top_themes[:10],
                    'period_days': days,
                    'last_updated': datetime.now().isoformat()
                })

                # Calculate percentages
                total = summary.get('total_feedback', 0)
                if total > 0:
                    summary['sentiment_percentages'] = {
                        'positive': round((summary.get('positive_count', 0) / total) * 100, 1),
                        'neutral': round((summary.get('neutral_count', 0) / total) * 100, 1),
                        'negative': round((summary.get('negative_count', 0) / total) * 100, 1)
                    }
                    summary['urgency_percentages'] = {
                        'high': round((summary.get('high_urgency', 0) / total) * 100, 1),
                        'medium': round((summary.get('medium_urgency', 0) / total) * 100, 1),
                        'low': round((summary.get('low_urgency', 0) / total) * 100, 1)
                    }

                # Cache the result
                self._cache_result(cache_key, summary)

                return summary

        except Exception as e:
            logger.error(f"Failed to get analytics summary: {str(e)}")
            return {}

    def store_gdg_action(self,
                         feedback_id: int,
                         action_title: str,
                         action_description: str,
                         priority: str,
                         category: str = None) -> int:
        """Store GDG action item"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO gdg_actions (
                        feedback_id, action_title, action_description, 
                        priority, category
                    ) VALUES (?, ?, ?, ?, ?)
                """, (feedback_id, action_title, action_description, priority, category))

                return cursor.lastrowid

        except Exception as e:
            logger.error(f"Failed to store GDG action: {str(e)}")
            return 0

    def get_pending_actions(self, priority: Optional[str] = None) -> List[Dict]:
        """Get pending GDG actions"""
        try:
            with self.get_cursor() as cursor:
                query = "SELECT * FROM gdg_actions WHERE status = 'pending'"
                params = []

                if priority:
                    query += " AND priority = ?"
                    params.append(priority)

                query += " ORDER BY priority DESC, created_at ASC"

                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get pending actions: {str(e)}")
            return []

    def log_performance(self,
                        operation_type: str,
                        execution_time_ms: float,
                        gpu_used: bool = False,
                        memory_usage_mb: float = None,
                        error_occurred: bool = False,
                        error_message: str = None):
        """Log system performance metrics"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO system_performance (
                        operation_type, execution_time_ms, gpu_used, 
                        memory_usage_mb, error_occurred, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    operation_type, execution_time_ms, gpu_used,
                    memory_usage_mb, error_occurred, error_message
                ))

        except Exception as e:
            logger.error(f"Failed to log performance: {str(e)}")

    def get_performance_stats(self, hours: int = 24) -> Dict:
        """Get system performance statistics"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        operation_type,
                        COUNT(*) as operation_count,
                        AVG(execution_time_ms) as avg_time,
                        MIN(execution_time_ms) as min_time,
                        MAX(execution_time_ms) as max_time,
                        SUM(CASE WHEN gpu_used THEN 1 ELSE 0 END) as gpu_operations,
                        SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) as error_count
                    FROM system_performance 
                    WHERE timestamp >= datetime('now', '-{} hours')
                    GROUP BY operation_type
                    ORDER BY operation_count DESC
                """.format(hours))

                stats = [dict(row) for row in cursor.fetchall()]

                return {
                    'period_hours': hours,
                    'operation_stats': stats,
                    'total_operations': sum(s['operation_count'] for s in stats),
                    'total_errors': sum(s['error_count'] for s in stats),
                    'gpu_utilization': sum(s['gpu_operations'] for s in stats) / max(sum(s['operation_count'] for s in stats), 1)
                }

        except Exception as e:
            logger.error(f"Failed to get performance stats: {str(e)}")
            return {}

    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """Clean up old data to maintain database performance"""
        try:
            with self.get_cursor() as cursor:
                # Clean old feedback analysis
                cursor.execute("""
                    DELETE FROM feedback_analysis 
                    WHERE created_at < datetime('now', '-{} days')
                """.format(days_to_keep))

                deleted_feedback = cursor.rowcount

                # Clean old performance logs
                cursor.execute("""
                    DELETE FROM system_performance 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep))

                deleted_performance = cursor.rowcount

                # Vacuum database to reclaim space
                cursor.execute("VACUUM")

                logger.info(f"Cleaned up {deleted_feedback} feedback records and {deleted_performance} performance logs")

                return deleted_feedback + deleted_performance

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
            return 0

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if not expired"""
        if cache_key in self._cache:
            if time.time() < self._cache_expiry.get(cache_key, 0):
                return self._cache[cache_key]
            else:
                # Remove expired cache
                del self._cache[cache_key]
                if cache_key in self._cache_expiry:
                    del self._cache_expiry[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: Any):
        """Cache result with expiry"""
        self._cache[cache_key] = result
        self._cache_expiry[cache_key] = time.time() + self._cache_ttl

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with self.get_cursor() as cursor:
                # Table sizes
                tables = ['feedback_analysis', 'analytics_summary', 'gdg_actions', 'system_performance']
                table_stats = {}

                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    table_stats[table] = cursor.fetchone()['count']

                # Database file size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

                return {
                    'database_path': str(self.db_path),
                    'database_size_mb': round(db_size / (1024 * 1024), 2),
                    'table_counts': table_stats,
                    'cache_size': len(self._cache),
                    'last_cleanup': 'Manual cleanup required'
                }

        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {}

    def close_connections(self):
        """Close all database connections"""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

        # Clear cache
        self._cache.clear()
        self._cache_expiry.clear()
