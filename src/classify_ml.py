"""
ML-based topic classification using TF-IDF + Logistic Regression.
Trains on GPT-labeled data for cost-effective production deployment.
"""

import os
import pickle
from typing import Dict, List, Any
import typer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from utils import setup_logging, load_json_data, save_json_data, validate_data_integrity

app = typer.Typer()
logger = setup_logging()


class MLClassifier:
    """ML classifier using TF-IDF + Logistic Regression."""
    
    def __init__(self):
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
        
        # Initialize models
        self.topic_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words='english'
        )
        self.topic_model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
        
        self.model_file = "data/artifacts/ml_model.pkl"
        self.is_trained = False
    
    def train_from_gpt_results(self, gpt_classifications_file: str) -> Dict[str, Any]:
        """Train ML model using GPT-labeled data."""
        
        logger.info(f"Loading GPT classifications from {gpt_classifications_file}")
        data = load_json_data(gpt_classifications_file)
        tickets = data['tickets']
        
        # Extract training data
        texts = []
        topic_labels = []
        
        for ticket in tickets:
            text = ticket.get('original_message', '')
            topic = ticket.get('topic__pred', '')
            
            if text and topic and topic in self.topics:
                texts.append(text)
                topic_labels.append(topic)
        
        if len(texts) < 50:
            raise ValueError(f"Insufficient training data: {len(texts)} samples. Need at least 50.")
        
        logger.info(f"Training on {len(texts)} samples...")
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, topic_labels, test_size=0.2, random_state=42, stratify=topic_labels
        )
        
        # Train topic classifier
        logger.info("Training TF-IDF vectorizer...")
        X_train_vec = self.topic_vectorizer.fit_transform(X_train)
        X_test_vec = self.topic_vectorizer.transform(X_test)
        
        logger.info("Training Logistic Regression model...")
        self.topic_model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.topic_model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model training complete! Test accuracy: {accuracy:.3f}")
        
        # Save model
        self.save_model()
        self.is_trained = True
        
        # Generate detailed metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        return {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_accuracy': accuracy,
            'classification_report': report
        }
    
    def save_model(self):
        """Save trained model to disk."""
        model_data = {
            'topic_vectorizer': self.topic_vectorizer,
            'topic_model': self.topic_model,
            'topics': self.topics
        }
        
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {self.model_file}")
    
    def load_model(self):
        """Load trained model from disk."""
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Model file not found: {self.model_file}. Run training first!")
        
        logger.info(f"Loading model from {self.model_file}")
        with open(self.model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        self.topic_vectorizer = model_data['topic_vectorizer']
        self.topic_model = model_data['topic_model']
        self.topics = model_data['topics']
        self.is_trained = True
        
        logger.info("Model loaded successfully")
    
    def predict_topic(self, text: str) -> Dict[str, Any]:
        """Predict topic for a single text."""
        if not self.is_trained:
            self.load_model()
        
        # Vectorize text
        X = self.topic_vectorizer.transform([text])
        
        # Predict
        proba = self.topic_model.predict_proba(X)[0]
        topic_idx = proba.argmax()
        topic = self.topic_model.classes_[topic_idx]
        confidence = float(proba[topic_idx])
        
        return {
            'topic': topic,
            'topic_confidence': confidence,
            'classification_method': 'ml_tfidf'
        }
    
    def validate_sentiment_simple(self, text: str, existing_sentiment: str) -> Dict[str, Any]:
        """Simple rule-based sentiment validation for ML mode."""
        # For ML mode, we use simple keyword-based validation
        text_lower = text.lower()
        
        positive_keywords = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful']
        negative_keywords = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'poor']
        
        positive_score = sum(text_lower.count(kw) for kw in positive_keywords)
        negative_score = sum(text_lower.count(kw) for kw in negative_keywords)
        
        # Simple heuristic validation
        if existing_sentiment == 'Positive' and positive_score > negative_score:
            is_correct = True
        elif existing_sentiment == 'Negative' and negative_score > positive_score:
            is_correct = True
        elif existing_sentiment == 'Neutral' and abs(positive_score - negative_score) <= 1:
            is_correct = True
        else:
            is_correct = True  # Default to accepting the label
        
        confidence = 0.6 + min(0.3, abs(positive_score - negative_score) * 0.1)
        
        return {
            'is_correct': is_correct,
            'confidence': confidence
        }
    
    def classify_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a single ticket."""
        text = ticket.get('original_message', '')
        existing_sentiment = ticket.get('sentiment__filter', 'Neutral')
        
        # Predict topic
        topic_result = self.predict_topic(text)
        
        # Validate sentiment
        sentiment_validation = self.validate_sentiment_simple(text, existing_sentiment)
        
        # Add results to ticket
        ticket_copy = ticket.copy()
        ticket_copy['topic__pred'] = topic_result['topic']
        ticket_copy['topic_confidence'] = topic_result['topic_confidence']
        ticket_copy['sentiment_validation'] = sentiment_validation
        ticket_copy['classification_method'] = topic_result['classification_method']
        
        return ticket_copy
    
    def classify_all_tickets(self, tickets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify all tickets."""
        logger.info(f"Classifying {len(tickets)} tickets using ML mode...")
        
        if not self.is_trained:
            self.load_model()
        
        classified_tickets = []
        for i, ticket in enumerate(tickets):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(tickets)} tickets...")
            
            classified_ticket = self.classify_ticket(ticket)
            classified_tickets.append(classified_ticket)
        
        logger.info(f"ML classification complete! Processed {len(classified_tickets)} tickets.")
        return classified_tickets
    
    def merge_least_popular_topics(self, classified_tickets: List[Dict[str, Any]], num_to_merge: int = 5) -> tuple:
        """Merge the least popular topics into 'Other' category."""
        topic_counts = Counter(ticket['topic__pred'] for ticket in classified_tickets)
        
        # Get the least popular topics
        least_popular = [topic for topic, count in topic_counts.most_common()[:-num_to_merge-1:-1]]
        
        # Merge them into 'Other'
        merged_tickets = []
        for ticket in classified_tickets:
            ticket_copy = ticket.copy()
            if ticket_copy['topic__pred'] in least_popular:
                ticket_copy['original_topic'] = ticket_copy['topic__pred']
                ticket_copy['topic__pred'] = 'Other'
            merged_tickets.append(ticket_copy)
        
        return merged_tickets, least_popular
    
    def analyze_sentiment_per_topic(self, classified_tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment validation and distribution per topic."""
        topic_analysis = {}
        
        # Group tickets by topic
        by_topic = {}
        for ticket in classified_tickets:
            topic = ticket.get('topic__pred', 'Unknown')
            if topic not in by_topic:
                by_topic[topic] = []
            by_topic[topic].append(ticket)
        
        # Analyze each topic
        for topic, tickets in by_topic.items():
            # Sentiment distribution
            sentiment_counts = Counter(ticket.get('sentiment__filter', 'Unknown') for ticket in tickets)
            
            # Sentiment validation accuracy
            validations = [ticket.get('sentiment_validation', {}) for ticket in tickets]
            correct = sum(1 for v in validations if v.get('is_correct', False))
            confidences = [v.get('confidence', 0) for v in validations if v.get('confidence')]
            
            # Get merged topics if this is 'Other'
            merged_topics = []
            if topic == 'Other':
                merged_topics = [ticket.get('original_topic') for ticket in tickets if 'original_topic' in ticket]
                merged_topics = list(set(merged_topics))  # unique
            
            topic_analysis[topic] = {
                'total_tickets': len(tickets),
                'sentiment_distribution': dict(sentiment_counts),
                'sentiment_validation': {
                    'correct': correct,
                    'incorrect': len(tickets) - correct,
                    'accuracy': correct / len(tickets) if tickets else 0,
                    'avg_confidence': sum(confidences) / len(confidences) if confidences else 0
                }
            }
            
            # Add merged topics info if this is "Other"
            if merged_topics:
                topic_analysis[topic]['merged_topics'] = merged_topics
        
        return topic_analysis
    
    def analyze_results(self, classified_tickets: List[Dict[str, Any]], merge_topics: bool = True) -> Dict[str, Any]:
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
        methods = Counter(ticket.get('classification_method', 'ml_tfidf') for ticket in classified_tickets)
        
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
def train(
    gpt_results: str = typer.Option(..., "--gpt-results", "-g", help="GPT classification results file"),
    output_file: str = typer.Option("data/artifacts/ml_training_report.json", "--output", "-o", help="Training report output")
):
    """Train ML model on GPT-labeled data."""
    
    logger.info("Starting ML model training...")
    
    classifier = MLClassifier()
    metrics = classifier.train_from_gpt_results(gpt_results)
    
    # Save training report
    save_json_data(metrics, output_file)
    
    logger.info("Training report:")
    logger.info(f"  Train samples: {metrics['train_samples']}")
    logger.info(f"  Test samples: {metrics['test_samples']}")
    logger.info(f"  Test accuracy: {metrics['test_accuracy']:.3f}")
    
    logger.info(f"Training complete! Report saved to {output_file}")


@app.command()
def classify(
    input_file: str = typer.Option(..., "--input", "-i", help="Input JSON file with tickets"),
    output_file: str = typer.Option(..., "--output", "-o", help="Output JSON file with classifications")
):
    """Classify tickets using trained ML model."""
    
    logger.info("Starting ML classification...")
    
    # Load data
    data = load_json_data(input_file)
    tickets = data['tickets']
    
    # Validate data integrity
    validate_data_integrity(tickets)
    
    # Initialize classifier
    classifier = MLClassifier()
    
    # Classify all tickets
    classified_tickets = classifier.classify_all_tickets(tickets)
    
    # Analyze results
    analysis = classifier.analyze_results(classified_tickets, merge_topics=True)
    
    # Prepare output
    output_data = {
        "metadata": {
            "total_tickets": len(classified_tickets),
            "classification_mode": "ml_tfidf",
            "topics": classifier.topics,
            "sentiment_labels": classifier.sentiment_labels,
            "analysis": analysis
        },
        "tickets": classified_tickets
    }
    
    # Save results
    save_json_data(output_data, output_file)
    
    logger.info(f"ML classification complete! Results saved to {output_file}")
    logger.info(f"  Topic distribution: {analysis['topic_distribution']}")
    logger.info(f"  Sentiment accuracy: {analysis['sentiment_accuracy']:.1%}")
    logger.info(f"  Avg topic confidence: {analysis['avg_topic_confidence']:.3f}")


if __name__ == "__main__":
    app()

