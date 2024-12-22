# app/rag/utils/result_formatter.py
"""
Module for formatting and analyzing search results.
Provides utilities for processing and presenting semantic search results.
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
from statistics import mean, median, stdev
from datetime import datetime

logger = logging.getLogger(__name__)

class SearchResultAnalyzer:
    """Class to analyze search results and provide statistical insights"""

    @staticmethod
    def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistical measures for search results

        Args:
            results: List of search results with similarity scores

        Returns:
            Dictionary containing statistical measures

        Raises:
            ValueError: If results list is empty or invalid
        """
        try:
            if not results:
                raise ValueError("Empty results list provided")

            # Extract similarity scores
            scores = []
            for result in results:
                if 'results' in result:
                    scores.extend([r['similarity_score'] for r in result['results']])

            if not scores:
                raise ValueError("No similarity scores found in results")

            # Calculate statistics
            stats = {
                'count': len(scores),
                'mean_similarity': mean(scores),
                'median_similarity': median(scores),
                'min_similarity': min(scores),
                'max_similarity': max(scores)
            }

            # Calculate standard deviation if more than one score exists
            if len(scores) > 1:
                stats['std_dev'] = stdev(scores)

            logger.info("Successfully calculated result statistics")
            return stats

        except Exception as e:
            error_msg = f"Failed to calculate statistics: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    @staticmethod
    def group_by_similarity_range(results: List[Dict[str, Any]],
                                ranges: List[Tuple[float, float]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by similarity score ranges

        Args:
            results: List of search results
            ranges: List of tuples defining score ranges (start, end)

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
                if not (0 <= start <= 1 and 0 <= end <= 1 and start < end):
                    raise ValueError("Invalid range values. Must be between 0 and 1")

            # Initialize groups
            grouped_results = {f"{start:.1f}-{end:.1f}": [] for start, end in ranges}

            # Group results
            for result in results:
                for r in result.get('results', []):
                    score = r['similarity_score']
                    for start, end in ranges:
                        if start <= score < end:
                            grouped_results[f"{start:.1f}-{end:.1f}"].append(r)
                            break

            logger.info("Successfully grouped results by similarity ranges")
            return grouped_results

        except Exception as e:
            error_msg = f"Failed to group results: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

class ResultFormatter:
    """Class to format search results in various representations"""

    @staticmethod
    def to_detailed_dict(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert search results to a detailed dictionary format

        Args:
            results: List of search results

        Returns:
            List of formatted dictionaries with detailed information

        Raises:
            ValueError: If results are invalid
        """
        try:
            formatted_results = []
            for result in results:
                question_results = {
                    'question': result['question'],
                    'match_count': len(result['results']),
                    'matches': []
                }

                for match in result['results']:
                    formatted_match = {
                        'data_point_id': match['data_point_id'],
                        'similarity_score': round(match['similarity_score'], 4),
                        'metadata': {
                            'filename': match['metadata'].get('filename', ''),
                            'content_preview': match['metadata'].get('content', '')[:200] + '...'
                            if len(match['metadata'].get('content', '')) > 200 else match['metadata'].get('content', ''),
                            'created_at': match['metadata'].get('created_at', ''),
                        }
                    }
                    question_results['matches'].append(formatted_match)

                formatted_results.append(question_results)

            logger.info("Successfully formatted results to detailed dictionary")
            return formatted_results

        except Exception as e:
            error_msg = f"Failed to format results: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    @staticmethod
    def to_summary_text(results: List[Dict[str, Any]],
                        max_matches: int = 5) -> str:
        """Convert search results to a human-readable summary text

        Args:
            results: List of search results
            max_matches: Maximum number of matches to include in summary

        Returns:
            Formatted summary text

        Raises:
            ValueError: If results are invalid
        """
        try:
            summary_parts = []

            for result in results:
                question = result['question']
                matches = result['results'][:max_matches]

                summary_parts.append(f"\nQuestion: {question}")
                summary_parts.append(f"Top {len(matches)} matches:")

                for i, match in enumerate(matches, 1):
                    summary_parts.append(f"\n{i}. Score: {match['similarity_score']:.4f}")
                    summary_parts.append(f"   File: {match['metadata'].get('filename', 'N/A')}")
                    content_preview = match['metadata'].get('content', '')[:100] + '...'
                    summary_parts.append(f"   Preview: {content_preview}")

                summary_parts.append("\n" + "-"*50)

            logger.info("Successfully created summary text")
            return "\n".join(summary_parts)

        except Exception as e:
            error_msg = f"Failed to create summary text: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    @staticmethod
    def filter_results(results: List[Dict[str, Any]],
                        min_score: float = 0.0,
                        max_results_per_query: Optional[int] = None) -> List[Dict[str, Any]]:
        """Filter search results based on criteria

        Args:
            results: List of search results
            min_score: Minimum similarity score to include
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
                    match for match in result['results']
                    if match['similarity_score'] >= min_score
                ]

                if max_results_per_query is not None:
                    filtered_matches = filtered_matches[:max_results_per_query]

                filtered_results.append({
                    'question': result['question'],
                    'results': filtered_matches
                })

            logger.info(
                f"Filtered results with min_score={min_score}, "
                f"max_results_per_query={max_results_per_query}"
            )
            return filtered_results

        except Exception as e:
            error_msg = f"Failed to filter results: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

def format_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Format metadata for consistent presentation

    Args:
        metadata: Raw metadata dictionary

    Returns:
        Formatted metadata dictionary

    Raises:
        ValueError: If metadata is invalid
    """
    try:
        created_at = metadata.get('created_at')
        if isinstance(created_at, (int, float)):
            # Convert Unix timestamp to datetime
            created_at = datetime.fromtimestamp(created_at)
        elif isinstance(created_at, str):
            try:
                # Try to parse string as datetime
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except ValueError:
                created_at = 'N/A'

        formatted = {
            'filename': metadata.get('filename', 'N/A'),
            'content_preview': metadata.get('content', '')[:200] + '...'
            if len(metadata.get('content', '')) > 200 else metadata.get('content', ''),
            'created_at': created_at.isoformat() if isinstance(created_at, datetime) else created_at
        }

        # Add additional metadata if available
        if 'additional_metadata' in metadata:
            formatted['additional_metadata'] = metadata['additional_metadata']

        return formatted

    except Exception as e:
        error_msg = f"Failed to format metadata: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
