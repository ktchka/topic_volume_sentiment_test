"""
Simplified classification and validation: topics + sentiment validation in ONE API call.
"""

import json
import os
from collections import Counter
from typing import Any

import typer
from openai import OpenAI

from utils import (
    create_text_hash,
    load_cache,
    load_json_data,
    save_cache,
    save_json_data,
    setup_logging,
    validate_data_integrity,
)

app = typer.Typer()
logger = setup_logging()


class SimpleClassifier:
    """Simple classifier: topics + sentiment validation in ONE API call."""

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

        self.sentiment_labels = ['Positive', 'Negative', 'Neutral']

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.openai_available = True
        else:
            self.client = None
            self.openai_available = False
            logger.warning("OPENAI_API_KEY not set, will use rule-based classification only")

        self.model = "gpt-4o-mini"
        self.temperature = 0.0

        # Cache for API responses
        self.cache_file = "data/artifacts/classification_cache.json"
        self.cache = load_cache(self.cache_file)

    def create_prompt(self, text: str, existing_sentiment: str) -> str:
        """Create prompt for topic classification + sentiment validation."""

        topics_list = "\n".join([f"- {topic}" for topic in self.topics])

        prompt = f"""Analyze this Booking.com review and:
1. Classify the topic
2. Validate if the existing sentiment label is correct

Review text: "{text}"
Existing sentiment label: "{existing_sentiment}"

TOPICS (choose the most relevant one):
{topics_list}

SENTIMENT VALIDATION:
- Check if the existing sentiment label ({existing_sentiment}) makes sense for this review
- Consider the overall tone, specific complaints/praises, and context
- Provide confidence score (0-1) for both topic and sentiment validation

Instructions:
1. Choose the SINGLE most relevant topic from the list above
2. Validate if the existing sentiment label is correct for this review
3. Provide confidence scores for both classifications

Respond with ONLY valid JSON in this exact format:
{{
    "topic": "Selected Topic Name",
    "topic_confidence": 0.85,
    "sentiment_validation": {{
        "is_correct": true,
        "confidence": 0.90
    }}
}}"""

        return prompt

    def classify_with_openai(self, text: str, existing_sentiment: str) -> dict[str, Any]:
        """Classify topic and validate sentiment using OpenAI API."""

        text_hash = create_text_hash(f"{text}-{existing_sentiment}")
        if text_hash in self.cache:
            logger.debug(f"Using cached result for text hash: {text_hash[:8]}...")
            return self.cache[text_hash]

        prompt = self.create_prompt(text, existing_sentiment)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a classification expert. Respond ONLY with valid JSON in the exact format specified."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
                max_tokens=300
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            # Validate required fields
            if 'topic' not in result or 'sentiment_validation' not in result:
                raise ValueError("Missing required fields in response")

            if result['topic'] not in self.topics:
                raise ValueError(f"Invalid topic: {result['topic']}")

            if 'is_correct' not in result['sentiment_validation']:
                raise ValueError("Missing is_correct field in sentiment_validation")

            result['classification_method'] = 'openai'
            self.cache[text_hash] = result
            save_cache(self.cache, self.cache_file)
            return result

        except Exception as e:
            logger.error(f"OpenAI classification failed: {e}")
            return self.classify_with_rules(text, existing_sentiment)

    def classify_with_rules(self, text: str, existing_sentiment: str) -> dict[str, Any]:
        """Fallback rule-based classification."""

        text_lower = text.lower()

        # Simple topic classification based on keywords
        topic_keywords = {
            'Unexpected Charges & Pricing': ['price', 'cost', 'fee', 'charge', 'expensive', 'cheap', 'money', 'payment'],
            'App Stability & Performance': ['app', 'crash', 'slow', 'freeze', 'bug', 'glitch', 'performance', 'lag'],
            'Booking Process Issues': ['book', 'reservation', 'process', 'step', 'form', 'complete', 'finish'],
            'Customer Service': ['service', 'support', 'help', 'staff', 'employee', 'representative', 'assistance'],
            'Payment Problems': ['payment', 'card', 'credit', 'debit', 'transaction', 'billing', 'invoice'],
            'Search & Filtering': ['search', 'filter', 'find', 'look', 'browse', 'results', 'options'],
            'Cancellation & Refunds': ['cancel', 'refund', 'return', 'money back', 'reimburse'],
            'Interface & Navigation': ['interface', 'navigation', 'menu', 'button', 'click', 'design', 'layout'],
            'Data & Privacy': ['data', 'privacy', 'personal', 'information', 'security', 'private']
        }

        # Find best topic match
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(text_lower.count(kw) for kw in keywords)
            topic_scores[topic] = score

        best_topic = max(topic_scores, key=topic_scores.get) if topic_scores else 'Interface & Navigation'
        topic_confidence = min(0.8, 0.3 + (topic_scores[best_topic] * 0.1))

        # Simple sentiment validation - assume existing sentiment is correct for rules
        sentiment_confidence = 0.6  # Default moderate confidence for rules

        return {
            'topic': best_topic,
            'topic_confidence': topic_confidence,
            'sentiment_validation': {
                'is_correct': True,  # Assume correct for rules fallback
                'confidence': sentiment_confidence
            },
            'classification_method': 'rules'
        }

    def classify_ticket(self, ticket: dict[str, Any], mode: str = "openai") -> dict[str, Any]:
        """Classify a single ticket for topic and validate sentiment."""

        text = ticket.get('original_message', '')
        existing_sentiment = ticket.get('sentiment__filter', 'Neutral')

        if mode == "openai" and self.openai_available:
            result = self.classify_with_openai(text, existing_sentiment)
        else:
            result = self.classify_with_rules(text, existing_sentiment)

        # Add results to ticket
        ticket_copy = ticket.copy()
        ticket_copy['topic__pred'] = result['topic']
        ticket_copy['topic_confidence'] = result['topic_confidence']
        ticket_copy['sentiment_validation'] = result['sentiment_validation']
        ticket_copy['classification_method'] = result['classification_method']

        return ticket_copy

    def classify_all_tickets(self, tickets: list[dict[str, Any]], mode: str = "openai") -> list[dict[str, Any]]:
        """Classify all tickets for topic and validate sentiment."""

        logger.info(f"Starting classification of {len(tickets)} tickets using {mode} mode...")

        classified_tickets = []
        for i, ticket in enumerate(tickets):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(tickets)} tickets...")

            classified_ticket = self.classify_ticket(ticket, mode)
            classified_tickets.append(classified_ticket)

        logger.info(f"Classification complete! Processed {len(classified_tickets)} tickets.")
        return classified_tickets

    def merge_least_popular_topics(self, classified_tickets: list[dict[str, Any]], num_to_merge: int = 5) -> list[dict[str, Any]]:
        """Merge the least popular topics into 'Other' category."""

        # Count topics
        topic_counts = Counter(ticket['topic__pred'] for ticket in classified_tickets)

        # Find least popular topics
        least_popular = [topic for topic, count in topic_counts.most_common()[:-num_to_merge-1:-1]]

        logger.info(f"Merging {len(least_popular)} least popular topics into 'Other': {least_popular}")

        # Update tickets
        merged_tickets = []
        for ticket in classified_tickets:
            ticket_copy = ticket.copy()
            if ticket_copy['topic__pred'] in least_popular:
                ticket_copy['original_topic'] = ticket_copy['topic__pred']
                ticket_copy['topic__pred'] = 'Other'
            merged_tickets.append(ticket_copy)

        return merged_tickets, least_popular

    def analyze_sentiment_per_topic(self, classified_tickets: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze sentiment validation and distribution per topic."""

        topic_analysis = {}

        # Get unique topics
        unique_topics = {ticket['topic__pred'] for ticket in classified_tickets}

        for topic in unique_topics:
            topic_tickets = [t for t in classified_tickets if t.get('topic__pred') == topic]

            if not topic_tickets:
                continue

            # Sentiment distribution
            sentiment_counts = Counter(t.get('sentiment__filter', 'Unknown') for t in topic_tickets)

            # Sentiment validation analysis
            validations = [t.get('sentiment_validation', {}) for t in topic_tickets]
            correct_count = sum(1 for v in validations if v.get('is_correct', False))
            incorrect_count = sum(1 for v in validations if not v.get('is_correct', True))
            avg_confidence = sum(v.get('confidence', 0) for v in validations) / len(validations) if validations else 0

            topic_analysis[topic] = {
                'total_tickets': len(topic_tickets),
                'sentiment_distribution': dict(sentiment_counts),
                'sentiment_validation': {
                    'correct': correct_count,
                    'incorrect': incorrect_count,
                    'accuracy': correct_count / len(topic_tickets) if topic_tickets else 0,
                    'avg_confidence': avg_confidence
                }
            }

            # Add merged topics info if this is "Other"
            if topic == 'Other':
                merged_topics = list({t.get('original_topic') for t in topic_tickets if 'original_topic' in t})
                topic_analysis[topic]['merged_topics'] = merged_topics

        return topic_analysis

    def analyze_results(self, classified_tickets: list[dict[str, Any]], merge_topics: bool = True) -> dict[str, Any]:
        """Analyze classification results and provide insights."""

        # Optionally merge least popular topics
        if merge_topics and len(classified_tickets) > 0:
            classified_tickets, merged_topics_list = self.merge_least_popular_topics(classified_tickets, num_to_merge=5)
        else:
            merged_topics_list = []

        # Topic analysis
        topic_counts = Counter(ticket['topic__pred'] for ticket in classified_tickets)
        topic_confidences = [ticket.get('topic_confidence', 0) for ticket in classified_tickets]

        # Sentiment validation analysis (overall)
        sentiment_validations = [ticket.get('sentiment_validation', {}) for ticket in classified_tickets]
        correct_sentiments = sum(1 for sv in sentiment_validations if sv.get('is_correct', False))
        sentiment_confidences = [sv.get('confidence', 0) for sv in sentiment_validations if sv.get('confidence')]

        # Per-topic sentiment analysis
        per_topic_analysis = self.analyze_sentiment_per_topic(classified_tickets)

        # Method analysis
        methods = Counter(ticket.get('classification_method', 'unknown') for ticket in classified_tickets)

        return {
            'topic_distribution': dict(topic_counts),
            'sentiment_accuracy': correct_sentiments / len(classified_tickets) if classified_tickets else 0,
            'methods_used': dict(methods),
            'avg_topic_confidence': sum(topic_confidences) / len(topic_confidences) if topic_confidences else 0,
            'avg_sentiment_confidence': sum(sentiment_confidences) / len(sentiment_confidences) if sentiment_confidences else 0,
            'total_tickets': len(classified_tickets),
            'correct_sentiments': correct_sentiments,
            'total_sentiments': len(classified_tickets),
            'merged_topics': merged_topics_list,
            'per_topic_sentiment_analysis': per_topic_analysis
        }


@app.command()
def main(
    input_file: str = typer.Option(..., "--input", "-i", help="Input JSON file with tickets"),
    output_file: str = typer.Option(..., "--output", "-o", help="Output JSON file with classifications"),
    mode: str = typer.Option("openai", "--mode", "-m", help="Classification mode: 'openai' or 'rules'")
):
    """Classify topics and validate sentiment in a single API call."""

    logger.info(f"Starting simple classification with mode: {mode}")

    # Load data
    data = load_json_data(input_file)
    tickets = data['tickets']

    # Validate data integrity
    validate_data_integrity(tickets)

    # Initialize classifier
    classifier = SimpleClassifier()

    # Classify everything
    classified_tickets = classifier.classify_all_tickets(tickets, mode)

    # Analyze results
    analysis = classifier.analyze_results(classified_tickets)

    # Prepare output data
    output_data = {
        "metadata": {
            "total_tickets": len(classified_tickets),
            "classification_mode": mode,
            "openai_available": classifier.openai_available,
            "topics": classifier.topics,
            "sentiment_labels": classifier.sentiment_labels,
            "analysis": analysis
        },
        "tickets": classified_tickets
    }

    # Save results
    save_json_data(output_data, output_file)

    # Print summary
    logger.info("Classification summary:")
    logger.info(f"  Total tickets: {len(classified_tickets)}")
    logger.info(f"  Methods used: {analysis['methods_used']}")
    logger.info(f"  Average topic confidence: {analysis['avg_topic_confidence']:.3f}")
    logger.info(f"  Overall sentiment accuracy: {analysis['sentiment_accuracy']:.1%}")
    logger.info(f"  Correct sentiments: {analysis['correct_sentiments']}/{analysis['total_sentiments']}")

    if analysis['merged_topics']:
        logger.info(f"  Merged topics into 'Other': {', '.join(analysis['merged_topics'])}")

    logger.info("\nTopic distribution:")
    for topic, count in sorted(analysis['topic_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(classified_tickets)) * 100
        logger.info(f"  {topic}: {count} ({percentage:.1f}%)")

    logger.info("\nPer-topic sentiment analysis:")
    for topic, topic_data in sorted(analysis['per_topic_sentiment_analysis'].items(),
                                    key=lambda x: x[1]['total_tickets'], reverse=True):
        logger.info(f"\n  {topic}:")
        logger.info(f"    Total tickets: {topic_data['total_tickets']}")
        logger.info(f"    Sentiment distribution: {topic_data['sentiment_distribution']}")
        logger.info(f"    Sentiment validation accuracy: {topic_data['sentiment_validation']['accuracy']:.1%}")
        logger.info(f"    Correct: {topic_data['sentiment_validation']['correct']}, "
                   f"Incorrect: {topic_data['sentiment_validation']['incorrect']}")
        logger.info(f"    Avg confidence: {topic_data['sentiment_validation']['avg_confidence']:.3f}")

        if topic == 'Other' and 'merged_topics' in topic_data:
            logger.info(f"    Merged topics: {', '.join(topic_data['merged_topics'])}")

    logger.info(f"\nClassification complete! Results saved to {output_file}")


if __name__ == "__main__":
    app()
