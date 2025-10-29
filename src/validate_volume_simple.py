"""
Simple volume validation: check if topic distribution matches reference.
"""

from collections import Counter
from typing import Any

import typer

from utils import load_json_data, save_json_data, setup_logging, validate_data_integrity

app = typer.Typer()
logger = setup_logging()


class SimpleVolumeValidator:
    """Simple volume validator: check topic distribution against reference."""

    def __init__(self):
        # Define topics directly in code (no yaml needed)
        self.topics = [
            'Unexpected Charges & Pricing',
            'App Stability & Performance',
            'Booking Process Issues',
            'Customer Service',
            'Payment Problems',
            'Search & Filtering',
            'Cancellation & Refunds',
            'Interface & Navigation',
            'Data & Privacy'
        ]

        # Simple thresholds
        self.max_percentage_diff = 15.0  # Max % point difference per topic
        self.min_topic_samples = 5       # Min samples per topic

    def parse_reference_distribution(self, reference_file: str) -> dict[str, float]:
        """Parse reference topic distribution from the reference file."""

        logger.info(f"Loading reference distribution from {reference_file}")
        reference_data = load_json_data(reference_file)

        # Extract topic percentages from reference data
        topic_percentages = {}

        # The reference data structure: gemini-2.5-flash -> array of objects with subtopic and percent
        if 'gemini-2.5-flash' in reference_data and reference_data['gemini-2.5-flash']:
            for item in reference_data['gemini-2.5-flash'][0]:  # First array contains the data
                subtopic = item.get('subtopic', '')
                percent = item.get('percent', 0)

                # Map reference subtopics to our topics
                mapped_topic = self.map_reference_to_topic(subtopic)
                if mapped_topic:
                    topic_percentages[mapped_topic] = percent

        logger.info(f"Reference distribution: {topic_percentages}")
        return topic_percentages

    def map_reference_to_topic(self, subtopic: str) -> str:
        """Map reference subtopic names to our topic names."""

        mapping = {
            'Unexpected Charges & Pricing': 'Unexpected Charges & Pricing',
            'App Stability & Performance': 'App Stability & Performance',
            'Property & Booking Discrepancies': 'Booking Process Issues',
            'Customer Support Issues': 'Customer Service',
            'Notifications & Advertising': 'Interface & Navigation',  # Map to closest topic
            'Ease of Booking & Confirmation': 'Booking Process Issues',
            'Effective Customer Support': 'Customer Service',
            'Clear Cancellation Policy': 'Cancellation & Refunds',
            'Interface & Navigation': 'Interface & Navigation'
        }

        return mapping.get(subtopic)

    def calculate_system_distribution(self, tickets: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate topic distribution from classified tickets."""

        topic_counts = Counter(ticket.get('topic__pred') for ticket in tickets)
        total_tickets = len(tickets)

        if total_tickets == 0:
            return {}

        distribution = {}
        for topic in self.topics:
            count = topic_counts.get(topic, 0)
            percentage = (count / total_tickets) * 100
            distribution[topic] = percentage

        return distribution

    def validate_distribution(self, system_dist: dict[str, float], reference_dist: dict[str, float]) -> dict[str, Any]:
        """Validate if system distribution matches reference distribution."""

        results = {}
        overall_pass = True

        for topic in self.topics:
            system_pct = system_dist.get(topic, 0)
            reference_pct = reference_dist.get(topic, 0)

            # Calculate percentage point difference
            diff = abs(system_pct - reference_pct)

            # Check if difference is within threshold
            topic_pass = diff <= self.max_percentage_diff

            if not topic_pass:
                overall_pass = False

            results[topic] = {
                'system_percentage': system_pct,
                'reference_percentage': reference_pct,
                'difference': diff,
                'pass': topic_pass,
                'threshold': self.max_percentage_diff
            }

        return {
            'overall_pass': overall_pass,
            'max_difference_allowed': self.max_percentage_diff,
            'topic_results': results,
            'summary': {
                'total_topics': len(self.topics),
                'passed_topics': sum(1 for r in results.values() if r['pass']),
                'failed_topics': sum(1 for r in results.values() if not r['pass'])
            }
        }

    def validate_volume(self, tickets: list[dict[str, Any]], reference_file: str) -> dict[str, Any]:
        """Main validation function."""

        logger.info("Starting volume validation...")

        # Calculate system distribution
        system_dist = self.calculate_system_distribution(tickets)
        logger.info(f"System distribution: {system_dist}")

        # Load reference distribution
        reference_dist = self.parse_reference_distribution(reference_file)

        # Validate distribution
        validation_results = self.validate_distribution(system_dist, reference_dist)

        # Add metadata
        validation_results['metadata'] = {
            'total_tickets': len(tickets),
            'topics_analyzed': len(self.topics),
            'validation_timestamp': 'unknown'
        }

        return validation_results


@app.command()
def main(
    input_file: str = typer.Option(..., "--input", "-i", help="Input JSON file with classified tickets"),
    reference_file: str = typer.Option(..., "--reference", "-r", help="Reference JSON file with topic distribution"),
    output_file: str = typer.Option(..., "--output", "-o", help="Output JSON file with validation results")
):
    """Validate topic volume distribution against reference."""

    logger.info("Starting simple volume validation...")

    # Load data
    data = load_json_data(input_file)
    tickets = data['tickets']

    # Validate data integrity
    validate_data_integrity(tickets)

    # Initialize validator
    validator = SimpleVolumeValidator()

    # Run validation
    results = validator.validate_volume(tickets, reference_file)

    # Save results
    save_json_data(results, output_file)

    # Print summary
    logger.info("Volume validation results:")
    logger.info(f"  Overall pass: {results['overall_pass']}")
    logger.info(f"  Passed topics: {results['summary']['passed_topics']}/{results['summary']['total_topics']}")

    if results['overall_pass']:
        logger.info("✅ Volume validation PASSED!")
    else:
        logger.info("❌ Volume validation FAILED!")
        failed_topics = [topic for topic, result in results['topic_results'].items() if not result['pass']]
        logger.info(f"  Failed topics: {', '.join(failed_topics)}")

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    app()
