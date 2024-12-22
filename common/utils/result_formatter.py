# app/common/utils/result_formatter.py
"""
Module for formatting and analyzing search results and other structured data.
Provides utilities for processing, analyzing, and presenting results in various formats.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from statistics import mean, median, stdev
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ResultType(Enum):
    """Enumeration for different types of results that can be processed"""
    SEARCH = "search"
    ANALYSIS = "analysis"
    GENERAL = "general"

@dataclass
class StatisticalSummary:
    """Data class for holding statistical summary information"""
    count: int
    mean: float
    median: float
    min_value: float
    max_value: float
    std_dev: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistical summary to dictionary format"""
        result = {
            'count': self.count,
            'mean': round(self.mean, 4),
            'median': round(self.median, 4),
            'min_value': round(self.min_value, 4),
            'max_value': round(self.max_value, 4)
        }
        if self.std_dev is not None:
            result['std_dev'] = round(self.std_dev, 4)
        return result

class ResultAnalyzer:
    """Class to analyze results and provide statistical insights"""

    @staticmethod
    def calculate_statistics(results: List[Dict[str, Any]],
                           score_field: str = 'similarity_score') -> StatisticalSummary:
        """Calculate statistical measures for numerical values in results

        Args:
            results: List of result dictionaries
            score_field: Field name containing the numerical values to analyze

        Returns:
            StatisticalSummary object containing calculated statistics

        Raises:
            ValueError: If results list is empty or score_field is invalid
        """
        try:
            if not results:
                raise ValueError("Empty results list provided")

            # Extract numerical values
            values = []
            for result in results:
                if 'results' in result:
                    values.extend([r[score_field] for r in result['results']
                                 if score_field in r])
                elif score_field in result:
                    values.append(result[score_field])

            if not values:
                raise ValueError(f"No values found for field: {score_field}")

            # Calculate basic statistics
            stats = StatisticalSummary(
                count=len(values),
                mean=mean(values),
                median=median(values),
                min_value=min(values),
                max_value=max(values),
                std_dev=stdev(values) if len(values) > 1 else None
            )

            logger.debug(f"Calculated statistics for {len(values)} values")
            return stats

        except Exception as e:
            error_msg = f"Failed to calculate statistics: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    @staticmethod
    def group_by_range(results: List[Dict[str, Any]],
                      ranges: List[Tuple[float, float]],
                      value_field: str = 'similarity_score') -> Dict[str, List[Dict[str, Any]]]:
        """Group results by numerical ranges

        Args:
            results: List of results to group
            ranges: List of tuples defining value ranges (start, end)
            value_field: Field name containing the value to check

        Returns:
            Dictionary with range keys and corresponding results

        Raises:
            ValueError: If ranges are invalid or results are empty
        """
        try:
            if not results or not ranges:
                raise ValueError("Empty results or ranges provided")

            # Validate ranges
            for start, end in ranges:
                if not (isinstance(start, (int, float)) and isinstance(end, (int, float)) and start < end):
                    raise ValueError(f"Invalid range values: {start}-{end}")

            # Initialize groups
            grouped_results = {f"{start}-{end}": [] for start, end in ranges}

            # Group results
            for result in results:
                if 'results' in result:
                    items = result['results']
                else:
                    items = [result]

                for item in items:
                    if value_field not in item:
                        continue
                    value = item[value_field]
                    for start, end in ranges:
                        if start <= value < end:
                            grouped_results[f"{start}-{end}"].append(item)
                            break

            logger.debug(f"Grouped results into {len(ranges)} ranges")
            return grouped_results

        except Exception as e:
            error_msg = f"Failed to group results: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

class ResultFormatter:
    """Class to format results in various representations"""

    def __init__(self, result_type: ResultType = ResultType.GENERAL):
        """Initialize formatter with specific result type"""
        self.result_type = result_type

    def to_detailed_dict(self,
                        results: List[Dict[str, Any]],
                        include_metadata: bool = True) -> List[Dict[str, Any]]:
        """Convert results to a detailed dictionary format

        Args:
            results: List of results to format
            include_metadata: Whether to include metadata in output

        Returns:
            List of formatted dictionaries with detailed information

        Raises:
            ValueError: If results are invalid
        """
        try:
            formatted_results = []
            for result in results:
                formatted_result = {
                    'query': result.get('question', 'No query provided'),
                    'match_count': len(result.get('results', [])),
                    'matches': []
                }

                for match in result.get('results', []):
                    formatted_match = {
                        'id': match.get('data_point_id', 'No ID'),
                        'score': round(match.get('similarity_score', 0.0), 4),
                    }

                    if include_metadata and 'metadata' in match:
                        formatted_match['metadata'] = self._format_metadata(match['metadata'])

                    formatted_result['matches'].append(formatted_match)

                formatted_results.append(formatted_result)

            logger.debug(f"Formatted {len(formatted_results)} results to detailed dictionary")
            return formatted_results

        except Exception as e:
            error_msg = f"Failed to format results: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def to_summary_text(self,
                       results: List[Dict[str, Any]],
                       max_matches: int = 5) -> str:
        """Convert results to a human-readable summary text

        Args:
            results: List of results to summarize
            max_matches: Maximum number of matches to include in summary

        Returns:
            Formatted summary text

        Raises:
            ValueError: If results are invalid
        """
        try:
            summary_parts = []

            for result in results:
                query = result.get('question', 'No query provided')
                matches = result.get('results', [])[:max_matches]

                summary_parts.append(f"\nQuery: {query}")
                summary_parts.append(f"Top {len(matches)} matches:")

                for i, match in enumerate(matches, 1):
                    score = match.get('similarity_score', 0.0)
                    summary_parts.append(f"\n{i}. Score: {score:.4f}")

                    if 'metadata' in match:
                        metadata = match['metadata']
                        filename = metadata.get('filename', 'N/A')
                        content = metadata.get('content', '')
                        content_preview = f"{content[:100]}..." if len(content) > 100 else content

                        summary_parts.append(f"   File: {filename}")
                        summary_parts.append(f"   Preview: {content_preview}")

                summary_parts.append("\n" + "-"*50)

            logger.debug("Created summary text")
            return "\n".join(summary_parts)

        except Exception as e:
            error_msg = f"Failed to create summary text: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    @staticmethod
    def _format_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format metadata for consistent presentation

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Formatted metadata dictionary

        Raises:
            ValueError: If metadata is invalid
        """
        try:
            # Handle created_at field
            created_at = metadata.get('created_at')
            if isinstance(created_at, (int, float)):
                created_at = datetime.fromtimestamp(created_at)
            elif isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except ValueError:
                    created_at = None

            # Format content preview
            content = metadata.get('content', '')
            content_preview = f"{content[:200]}..." if len(content) > 200 else content

            formatted = {
                'filename': metadata.get('filename', 'N/A'),
                'content_preview': content_preview,
                'created_at': created_at.isoformat() if created_at else 'N/A'
            }

            # Include additional metadata if available
            if 'additional_metadata' in metadata:
                formatted['additional_metadata'] = metadata['additional_metadata']

            return formatted

        except Exception as e:
            error_msg = f"Failed to format metadata: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def filter_results(self,
                      results: List[Dict[str, Any]],
                      min_score: float = 0.0,
                      max_results_per_query: Optional[int] = None) -> List[Dict[str, Any]]:
        """Filter results based on criteria

        Args:
            results: List of results to filter
            min_score: Minimum score threshold
            max_results_per_query: Maximum number of results per query

        Returns:
            Filtered results list

        Raises:
            ValueError: If filtering parameters are invalid
        """
        try:
            if not 0 <= min_score <= 1:
                raise ValueError("min_score must be between 0 and 1")

            filtered_results = []
            for result in results:
                filtered_matches = [
                    match for match in result.get('results', [])
                    if match.get('similarity_score', 0.0) >= min_score
                ]

                if max_results_per_query is not None:
                    filtered_matches = filtered_matches[:max_results_per_query]

                filtered_results.append({
                    'question': result.get('question', 'No query provided'),
                    'results': filtered_matches
                })

            logger.debug(
                f"Filtered results with min_score={min_score}, "
                f"max_results_per_query={max_results_per_query}"
            )
            return filtered_results

        except Exception as e:
            error_msg = f"Failed to filter results: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
