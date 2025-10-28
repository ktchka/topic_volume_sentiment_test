"""
Generate simple validation report.
"""

import os
from typing import Dict, Any
import typer

from utils import setup_logging, load_json_data

app = typer.Typer()
logger = setup_logging()


class SimpleReportGenerator:
    """Generates simple validation reports."""
    
    def load_artifacts(self, artifacts_dir: str) -> Dict[str, Any]:
        """Load all validation artifacts."""
        artifacts = {}
        
        # Load classifications (contains per-topic sentiment analysis)
        # Try both absolute and relative paths
        classifications_file = "data/derived/classifications.json"
        if not os.path.exists(classifications_file):
            classifications_file = os.path.join(os.path.dirname(artifacts_dir.rstrip('/')), "derived", "classifications.json")
        
        if os.path.exists(classifications_file):
            logger.info(f"Loading classifications from {classifications_file}")
            artifacts['classifications'] = load_json_data(classifications_file)
        else:
            logger.warning(f"Classifications file not found: {classifications_file}")
        
        # Load volume validation
        volume_file = os.path.join(artifacts_dir, "volume_validation.json")
        if os.path.exists(volume_file):
            logger.info(f"Loading volume validation from {volume_file}")
            artifacts['volume'] = load_json_data(volume_file)
        else:
            logger.warning(f"Volume validation file not found: {volume_file}")
        
        return artifacts
    
    def generate_executive_summary(self, artifacts: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        
        summary = ["# Topic Volume & Sentiment Validation Report\n"]
        
        # Overall status
        volume_pass = artifacts.get('volume', {}).get('overall_pass', False)
        
        # Get sentiment accuracy from classifications
        sentiment_accuracy = 0.0
        if 'classifications' in artifacts and 'metadata' in artifacts['classifications']:
            sentiment_accuracy = artifacts['classifications']['metadata'].get('analysis', {}).get('sentiment_accuracy', 0.0)
        
        status_emoji = "✅ PASS" if volume_pass else "❌ FAIL"
        summary.append(f"**Volume Validation Status: {status_emoji}**\n")
        summary.append(f"**Overall Sentiment Validation Accuracy: {sentiment_accuracy:.1%}**\n")
        
        # Classification summary
        if 'classifications' in artifacts and 'metadata' in artifacts['classifications']:
            metadata = artifacts['classifications']['metadata']
            analysis = metadata.get('analysis', {})
            
            summary.append("## Classification Summary\n")
            summary.append(f"- **Total tickets**: {metadata.get('total_tickets', 0)}")
            summary.append(f"- **Classification mode**: {metadata.get('classification_mode', 'unknown')}")
            summary.append(f"- **Average topic confidence**: {analysis.get('avg_topic_confidence', 0):.3f}")
            summary.append(f"- **Average sentiment confidence**: {analysis.get('avg_sentiment_confidence', 0):.3f}")
            summary.append(f"- **Correct sentiments**: {analysis.get('correct_sentiments', 0)}/{analysis.get('total_sentiments', 0)}")
            
            if analysis.get('merged_topics'):
                summary.append(f"- **Merged topics into 'Other'**: {', '.join(analysis['merged_topics'])}")
            
            summary.append("")
        
        return "\n".join(summary)
    
    def generate_topic_distribution(self, artifacts: Dict[str, Any]) -> str:
        """Generate topic distribution section."""
        
        if 'classifications' not in artifacts:
            return "## Topic Distribution\n\nNo classification data available.\n"
        
        analysis = artifacts['classifications']['metadata'].get('analysis', {})
        topic_dist = analysis.get('topic_distribution', {})
        total_tickets = artifacts['classifications']['metadata'].get('total_tickets', 0)
        
        section = ["## Topic Distribution\n"]
        section.append("| Topic | Count | Percentage |")
        section.append("|-------|-------|------------|")
        
        # Sort by count descending
        for topic, count in sorted(topic_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_tickets * 100) if total_tickets > 0 else 0
            section.append(f"| {topic} | {count} | {percentage:.1f}% |")
        
        section.append("")
        return "\n".join(section)
    
    def generate_volume_validation(self, artifacts: Dict[str, Any]) -> str:
        """Generate volume validation section."""
        
        if 'volume' not in artifacts:
            return "## Volume Validation\n\nNo volume validation data available.\n"
        
        volume = artifacts['volume']
        
        section = ["## Volume Validation\n"]
        
        status = "✅ PASS" if volume.get('overall_pass', False) else "❌ FAIL"
        section.append(f"**Status**: {status}\n")
        section.append(f"**Max difference allowed**: {volume.get('max_difference_allowed', 0):.1f} percentage points\n")
        
        # Summary
        summary = volume.get('summary', {})
        section.append(f"- **Total topics**: {summary.get('total_topics', 0)}")
        section.append(f"- **Passed topics**: {summary.get('passed_topics', 0)}")
        section.append(f"- **Failed topics**: {summary.get('failed_topics', 0)}\n")
        
        # Per-topic results
        if 'topic_results' in volume:
            section.append("### Per-Topic Results\n")
            section.append("| Topic | System % | Reference % | Difference | Status |")
            section.append("|-------|----------|-------------|------------|--------|")
            
            for topic, result in sorted(volume['topic_results'].items(), 
                                       key=lambda x: x[1]['system_percentage'], reverse=True):
                status_icon = "✅" if result.get('pass', False) else "❌"
                section.append(f"| {topic} | {result['system_percentage']:.1f}% | "
                             f"{result['reference_percentage']:.1f}% | "
                             f"{result['difference']:.1f}pp | {status_icon} |")
            
            section.append("")
        
        return "\n".join(section)
    
    def generate_sentiment_analysis(self, artifacts: Dict[str, Any]) -> str:
        """Generate per-topic sentiment analysis section."""
        
        if 'classifications' not in artifacts:
            return "## Sentiment Analysis\n\nNo classification data available.\n"
        
        analysis = artifacts['classifications']['metadata'].get('analysis', {})
        per_topic = analysis.get('per_topic_sentiment_analysis', {})
        
        if not per_topic:
            return "## Sentiment Analysis\n\nNo per-topic sentiment data available.\n"
        
        section = ["## Per-Topic Sentiment Analysis\n"]
        
        # Sort by total tickets descending
        for topic, data in sorted(per_topic.items(), key=lambda x: x[1]['total_tickets'], reverse=True):
            section.append(f"### {topic}\n")
            
            # Basic stats
            section.append(f"- **Total tickets**: {data['total_tickets']}")
            
            # Sentiment distribution
            sentiment_dist = data.get('sentiment_distribution', {})
            section.append("- **Sentiment distribution**:")
            for sentiment, count in sorted(sentiment_dist.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / data['total_tickets'] * 100) if data['total_tickets'] > 0 else 0
                section.append(f"  - {sentiment}: {count} ({percentage:.1f}%)")
            
            # Validation accuracy
            validation = data.get('sentiment_validation', {})
            section.append(f"- **Sentiment validation accuracy**: {validation.get('accuracy', 0):.1%}")
            section.append(f"  - Correct: {validation.get('correct', 0)}")
            section.append(f"  - Incorrect: {validation.get('incorrect', 0)}")
            section.append(f"  - Avg confidence: {validation.get('avg_confidence', 0):.3f}")
            
            # Show merged topics if this is "Other"
            if topic == 'Other' and 'merged_topics' in data:
                section.append(f"- **Merged topics**: {', '.join(data['merged_topics'])}")
            
            section.append("")
        
        return "\n".join(section)
    
    def generate_recommendations(self, artifacts: Dict[str, Any]) -> str:
        """Generate recommendations based on results."""
        
        recommendations = ["## Recommendations\n"]
        
        # Volume validation recommendations
        if 'volume' in artifacts:
            volume = artifacts['volume']
            if not volume.get('overall_pass', False):
                failed_topics = [topic for topic, result in volume.get('topic_results', {}).items() 
                               if not result.get('pass', False)]
                
                if failed_topics:
                    recommendations.append("### Topic Volume Issues\n")
                    recommendations.append("The following topics have distribution differences exceeding the threshold:\n")
                    for topic in failed_topics:
                        recommendations.append(f"- **{topic}**")
                    recommendations.append("\n**Actions:**")
                    recommendations.append("- Review topic classification accuracy for these categories")
                    recommendations.append("- Verify reference distribution mapping is correct")
                    recommendations.append("- Consider if topic definitions need refinement\n")
        
        # Sentiment validation recommendations
        if 'classifications' in artifacts:
            analysis = artifacts['classifications']['metadata'].get('analysis', {})
            per_topic = analysis.get('per_topic_sentiment_analysis', {})
            
            # Find topics with low sentiment validation accuracy
            low_accuracy_topics = []
            for topic, data in per_topic.items():
                accuracy = data.get('sentiment_validation', {}).get('accuracy', 1.0)
                if accuracy < 0.80:  # Less than 80% accuracy
                    low_accuracy_topics.append((topic, accuracy))
            
            if low_accuracy_topics:
                recommendations.append("### Sentiment Validation Issues\n")
                recommendations.append("The following topics have low sentiment validation accuracy:\n")
                for topic, accuracy in sorted(low_accuracy_topics, key=lambda x: x[1]):
                    recommendations.append(f"- **{topic}**: {accuracy:.1%}")
                recommendations.append("\n**Actions:**")
                recommendations.append("- Review sentiment labels for these topics")
                recommendations.append("- Consider if sentiment definitions are appropriate for topic context")
                recommendations.append("- Verify sentiment labeling consistency\n")
        
        # General recommendations
        recommendations.append("### General Improvements\n")
        recommendations.append("- Regularly update and validate reference distributions")
        recommendations.append("- Monitor classification confidence scores over time")
        recommendations.append("- Consider topic-specific sentiment validation rules")
        recommendations.append("- Increase sample size for topics with low confidence\n")
        
        return "\n".join(recommendations)
    
    def generate_report(self, artifacts: Dict[str, Any], output_file: str) -> str:
        """Generate complete validation report."""
        
        logger.info("Generating validation report...")
        
        report_sections = [
            self.generate_executive_summary(artifacts),
            self.generate_topic_distribution(artifacts),
            self.generate_volume_validation(artifacts),
            self.generate_sentiment_analysis(artifacts),
            self.generate_recommendations(artifacts)
        ]
        
        report_content = "\n".join(report_sections)
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report generated: {output_file}")
        return report_content


@app.command()
def main(
    artifacts_dir: str = typer.Option(..., "--artifacts", "-a", help="Directory containing validation artifacts"),
    output_file: str = typer.Option("report.md", "--output", "-o", help="Output markdown file")
):
    """Generate simple validation report."""
    
    logger.info("Starting report generation...")
    
    # Load artifacts
    generator = SimpleReportGenerator()
    artifacts = generator.load_artifacts(artifacts_dir)
    
    if not artifacts:
        logger.error("No validation artifacts found!")
        return
    
    # Generate report
    generator.generate_report(artifacts, output_file)
    
    logger.info("Report generation complete!")


if __name__ == "__main__":
    app()
