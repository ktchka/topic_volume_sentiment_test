"""
Compare classification methods: GPT vs Rules vs ML.
"""

from typing import Dict, Any
import typer

from utils import setup_logging, load_json_data, save_json_data

app = typer.Typer()
logger = setup_logging()


def load_results(file_path: str) -> Dict[str, Any]:
    """Load classification results."""
    try:
        return load_json_data(file_path)
    except Exception as e:
        logger.warning(f"Could not load {file_path}: {e}")
        return None


def calculate_agreement(results1: Dict, results2: Dict) -> float:
    """Calculate topic classification agreement between two methods."""
    if not results1 or not results2:
        return 0.0
    
    tickets1 = {t['original_message']: t['topic__pred'] for t in results1['tickets']}
    tickets2 = {t['original_message']: t['topic__pred'] for t in results2['tickets']}
    
    common_tickets = set(tickets1.keys()) & set(tickets2.keys())
    if not common_tickets:
        return 0.0
    
    agreements = sum(1 for msg in common_tickets if tickets1[msg] == tickets2[msg])
    return agreements / len(common_tickets)


def analyze_method(results: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    """Analyze results from a single method."""
    if not results:
        return {
            'method': method_name,
            'error': 'Results not available'
        }
    
    metadata = results.get('metadata', {})
    analysis = metadata.get('analysis', {})
    
    return {
        'method': method_name,
        'total_tickets': metadata.get('total_tickets', 0),
        'avg_topic_confidence': analysis.get('avg_topic_confidence', 0),
        'sentiment_accuracy': analysis.get('sentiment_accuracy', 0),
        'correct_sentiments': analysis.get('correct_sentiments', 0),
        'topic_distribution': analysis.get('topic_distribution', {})
    }


@app.command()
def compare(
    gpt_results: str = typer.Option("data/derived/classifications.json", "--gpt", help="GPT classification results"),
    ml_results: str = typer.Option("data/derived/classifications_ml.json", "--ml", help="ML classification results"),
    output_file: str = typer.Option("data/artifacts/method_comparison.json", "--output", "-o", help="Comparison output file")
):
    """Compare different classification methods."""
    
    logger.info("Loading classification results...")
    
    gpt_data = load_results(gpt_results)
    ml_data = load_results(ml_results)
    
    # Analyze each method
    gpt_analysis = analyze_method(gpt_data, "GPT-4o-mini")
    ml_analysis = analyze_method(ml_data, "ML (TF-IDF + LR)")
    
    # Calculate agreement
    gpt_ml_agreement = calculate_agreement(gpt_data, ml_data) if gpt_data and ml_data else 0.0
    
    # Compile comparison
    comparison = {
        'methods': {
            'gpt': gpt_analysis,
            'ml': ml_analysis
        },
        'agreement': {
            'gpt_vs_ml': gpt_ml_agreement
        },
        'summary': {
            'best_accuracy': max(
                gpt_analysis.get('sentiment_accuracy', 0),
                ml_analysis.get('sentiment_accuracy', 0)
            ),
            'best_confidence': max(
                gpt_analysis.get('avg_topic_confidence', 0),
                ml_analysis.get('avg_topic_confidence', 0)
            ),
            'highest_agreement': gpt_ml_agreement
        },
        'recommendations': generate_recommendations(gpt_analysis, ml_analysis)
    }
    
    # Save comparison
    save_json_data(comparison, output_file)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("CLASSIFICATION METHOD COMPARISON")
    logger.info("="*60)
    
    logger.info("\nGPT-4o-mini:")
    logger.info(f"  Topic Confidence: {gpt_analysis.get('avg_topic_confidence', 0):.3f}")
    logger.info(f"  Sentiment Accuracy: {gpt_analysis.get('sentiment_accuracy', 0):.1%}")
    logger.info("  Estimated Cost: ~$0.30 per 1000 tickets")
    
    logger.info("\nML (TF-IDF + Logistic Regression):")
    logger.info(f"  Topic Confidence: {ml_analysis.get('avg_topic_confidence', 0):.3f}")
    logger.info(f"  Sentiment Accuracy: {ml_analysis.get('sentiment_accuracy', 0):.1%}")
    logger.info("  Cost: Free (after training)")
    
    logger.info("\nAgreement:")
    logger.info(f"  GPT vs ML: {gpt_ml_agreement:.1%}")
    
    logger.info(f"\nComparison saved to {output_file}")


def generate_recommendations(gpt_analysis: Dict, ml_analysis: Dict) -> Dict[str, str]:
    """Generate recommendations based on comparison."""
    recommendations = {}
    
    gpt_acc = gpt_analysis.get('sentiment_accuracy', 0)
    ml_acc = ml_analysis.get('sentiment_accuracy', 0)
    
    if abs(gpt_acc - ml_acc) < 0.05:
        recommendations['accuracy'] = "ML model achieves comparable accuracy to GPT at zero marginal cost"
    elif gpt_acc > ml_acc:
        recommendations['accuracy'] = f"GPT has {(gpt_acc - ml_acc):.1%} higher accuracy - consider for high-stakes classification"
    else:
        recommendations['accuracy'] = "ML model outperforms GPT - use for production"
    
    recommendations['cost'] = "ML model recommended for production scale (>10k tickets/month)"
    recommendations['hybrid'] = "Consider hybrid: ML for high-confidence, GPT for uncertain cases"
    
    return recommendations


if __name__ == "__main__":
    app()

