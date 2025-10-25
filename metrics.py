import sqlite3
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

class MetricsTracker:
    """Tracks and reports on evaluation criteria for code review analysis."""

    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    repo_path TEXT,
                    pr_url TEXT,
                    analysis_result TEXT,  -- JSON string
                    analysis_time REAL,    -- seconds
                    total_lines_changed INTEGER,
                    total_files INTEGER
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    finding_title TEXT,
                    actual_outcome TEXT,  -- 'true_positive', 'false_positive', 'missed'
                    user_feedback TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_logs(id)
                )
            ''')
            conn.commit()

    def log_analysis(self, repo_path: str, pr_url: str, analysis_result: Dict[str, Any],
                    analysis_time: float, total_lines_changed: int, total_files: int) -> int:
        """Log an analysis run to the database. Returns the log ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO analysis_logs (timestamp, repo_path, pr_url, analysis_result,
                                         analysis_time, total_lines_changed, total_files)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                repo_path,
                pr_url,
                json.dumps(analysis_result),
                analysis_time,
                total_lines_changed,
                total_files
            ))
            conn.commit()
            return cursor.lastrowid

    def log_feedback(self, analysis_id: int, finding_title: str, actual_outcome: str,
                    user_feedback: str = "") -> None:
        """Log manual feedback on a finding for false positive calculation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO feedback_logs (analysis_id, finding_title, actual_outcome,
                                         user_feedback, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                analysis_id,
                finding_title,
                actual_outcome,
                user_feedback,
                datetime.now().isoformat()
            ))
            conn.commit()

    def calculate_false_positive_rate(self) -> Dict[str, Any]:
        """Calculate false positive rates from feedback logs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT actual_outcome, COUNT(*) as count
                FROM feedback_logs
                GROUP BY actual_outcome
            ''')
            outcomes = dict(cursor.fetchall())

        total_feedbacks = sum(outcomes.values())
        if total_feedbacks == 0:
            return {"false_positive_rate": 0.0, "total_feedbacks": 0, "details": {}}

        false_positives = outcomes.get('false_positive', 0)
        rate = false_positives / total_feedbacks if total_feedbacks > 0 else 0.0

        return {
            "false_positive_rate": rate,
            "total_feedbacks": total_feedbacks,
            "details": outcomes
        }

    def estimate_review_time_reduction(self, pr_size_threshold: int = 100) -> Dict[str, Any]:
        """Estimate review time reduction based on PR size vs analysis time.

        Assumes manual review time is proportional to PR size, and analysis time is the automated part.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT total_lines_changed, analysis_time
                FROM analysis_logs
                WHERE total_lines_changed > 0
            ''')
            data = cursor.fetchall()

        if not data:
            return {"average_reduction_percent": 0.0, "total_analyses": 0}

        # Simple heuristic: assume manual review time = 0.1 * lines_changed (minutes)
        # Analysis time in seconds, convert to minutes
        reductions = []
        for lines, analysis_sec in data:
            manual_time_min = lines * 0.1  # heuristic
            analysis_time_min = analysis_sec / 60
            if manual_time_min > 0:
                reduction_percent = (analysis_time_min / manual_time_min) * 100
                reductions.append(reduction_percent)

        avg_reduction = sum(reductions) / len(reductions) if reductions else 0.0

        return {
            "average_reduction_percent": avg_reduction,
            "total_analyses": len(data),
            "details": reductions[:10]  # last 10 for sample
        }

    def get_aggregated_reports(self) -> Dict[str, Any]:
        """Generate aggregated metrics reports."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT COUNT(*), AVG(analysis_time), AVG(total_lines_changed), AVG(total_files)
                FROM analysis_logs
            ''')
            total_analyses, avg_time, avg_lines, avg_files = cursor.fetchone()

        false_pos = self.calculate_false_positive_rate()
        time_red = self.estimate_review_time_reduction()

        return {
            "total_analyses": total_analyses or 0,
            "average_analysis_time_seconds": avg_time or 0.0,
            "average_lines_changed": avg_lines or 0.0,
            "average_files": avg_files or 0.0,
            "false_positive_metrics": false_pos,
            "review_time_reduction": time_red,
            "generated_at": datetime.now().isoformat()
        }

    def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis logs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT id, timestamp, repo_path, pr_url, analysis_time, total_lines_changed, total_files
                FROM analysis_logs
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "repo_path": row[2],
                "pr_url": row[3],
                "analysis_time": row[4],
                "total_lines_changed": row[5],
                "total_files": row[6]
            }
            for row in rows
        ]

# Global instance
metrics_tracker = MetricsTracker()
