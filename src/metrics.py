"""
Statistical metrics for topic volume and sentiment validation.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_fscore_support,
    accuracy_score
)
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps=0.5) -> float:
    """Calculate Jensen-Shannon Divergence between two distributions with Laplace smoothing."""
    # Ensure distributions are normalized
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Apply Laplace smoothing to avoid log(0) and make JSD more robust
    p = (p + eps) / (p.sum() + eps * len(p))
    q = (q + eps) / (q.sum() + eps * len(q))
    
    return jensenshannon(p, q)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Total Variation Distance between two distributions."""
    # Ensure distributions are normalized
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return 0.5 * np.sum(np.abs(p - q))


def chi_square_goodness_of_fit(observed: np.ndarray, 
                              expected: np.ndarray) -> Tuple[float, float]:
    """Calculate chi-square goodness of fit test."""
    # Ensure arrays have same length
    min_len = min(len(observed), len(expected))
    observed = observed[:min_len]
    expected = expected[:min_len]
    
    # Calculate chi-square statistic
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    
    # Calculate p-value
    degrees_of_freedom = len(observed) - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, degrees_of_freedom)
    
    return chi2_stat, p_value


def calculate_topic_volume_metrics(system_dist: Dict[str, float], 
                                 reference_dist: Dict[str, float], 
                                 topics: List[str]) -> Dict[str, Any]:
    """Calculate metrics for topic volume validation."""
    
    # Convert to arrays in consistent order
    system_array = np.array([system_dist.get(topic, 0) for topic in topics])
    reference_array = np.array([reference_dist.get(topic, 0) for topic in topics])
    
    # Calculate metrics
    jsd = jensen_shannon_divergence(system_array, reference_array)
    tvd = total_variation_distance(system_array, reference_array)
    
    # Calculate per-topic absolute differences
    abs_diffs = np.abs(system_array - reference_array) * 100  # Convert to percentage points
    max_abs_diff = np.max(abs_diffs)
    
    # Chi-square test
    observed_counts = system_array * np.sum(reference_array)  # Scale to match reference total
    expected_counts = reference_array * np.sum(reference_array)
    chi2_stat, chi2_pvalue = chi_square_goodness_of_fit(observed_counts, expected_counts)
    
    return {
        'jensen_shannon_divergence': jsd,
        'total_variation_distance': tvd,
        'max_abs_diff': max_abs_diff,
        'per_topic_abs_diffs': dict(zip(topics, abs_diffs)),
        'chi_square_statistic': chi2_stat,
        'chi_square_pvalue': chi2_pvalue
    }


def calculate_sentiment_metrics(y_true: List[str], y_pred: List[str], 
                              labels: List[str]) -> Dict[str, Any]:
    """Calculate sentiment classification metrics."""
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Calculate weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'support': support.tolist()
    }


def calculate_agreement_metrics(y1: List[str], y2: List[str], 
                              labels: List[str]) -> Dict[str, Any]:
    """Calculate agreement metrics between two labelers."""
    
    # Calculate agreement rate
    agreement_rate = np.mean([y1[i] == y2[i] for i in range(len(y1))])
    
    # Calculate per-class agreement
    per_class_agreement = {}
    for label in labels:
        label_indices = [i for i, y in enumerate(y1) if y == label]
        if label_indices:
            class_agreement = np.mean([y1[i] == y2[i] for i in label_indices])
            per_class_agreement[label] = class_agreement
        else:
            per_class_agreement[label] = 0.0
    
    # Calculate Cohen's Kappa
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(y1, y2)
    
    return {
        'agreement_rate': agreement_rate,
        'per_class_agreement': per_class_agreement,
        'cohens_kappa': kappa
    }


def calculate_prevalence_estimation(y_true: List[str], y_pred: List[str], 
                                  labels: List[str]) -> Dict[str, Any]:
    """Calculate prevalence estimation metrics."""
    
    # Calculate true prevalence
    true_counts = [y_true.count(label) for label in labels]
    true_prevalence = np.array(true_counts) / len(y_true)
    
    # Calculate predicted prevalence
    pred_counts = [y_pred.count(label) for label in labels]
    pred_prevalence = np.array(pred_counts) / len(y_pred)
    
    # Calculate prevalence gaps
    prevalence_gaps = np.abs(true_prevalence - pred_prevalence) * 100  # Convert to percentage points
    max_gap = np.max(prevalence_gaps)
    
    # Calculate mean absolute error
    mae = np.mean(prevalence_gaps)
    
    return {
        'true_prevalence': dict(zip(labels, true_prevalence)),
        'predicted_prevalence': dict(zip(labels, pred_prevalence)),
        'prevalence_gaps': dict(zip(labels, prevalence_gaps)),
        'max_gap': max_gap,
        'mean_absolute_error': mae
    }


def validate_thresholds(metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, bool]:
    """Validate metrics against thresholds."""
    
    validation_results = {}
    
    # Topic volume thresholds
    if 'topic_volume' in config:
        tv_thresholds = config['topic_volume']
        
        if 'max_jsd_overall' in tv_thresholds:
            validation_results['jsd_threshold'] = (
                metrics.get('jensen_shannon_divergence', 0) <= tv_thresholds['max_jsd_overall']
            )
        
        if 'max_abs_pp_diff_per_topic' in tv_thresholds:
            validation_results['max_diff_threshold'] = (
                metrics.get('max_abs_diff', 0) <= tv_thresholds['max_abs_pp_diff_per_topic']
            )
        
        if 'min_chi2_pvalue' in tv_thresholds:
            validation_results['chi2_threshold'] = (
                metrics.get('chi_square_pvalue', 0) >= tv_thresholds['min_chi2_pvalue']
            )
    
    # Sentiment thresholds
    if 'sentiment' in config:
        sent_thresholds = config['sentiment']
        
        if 'min_macro_f1' in sent_thresholds:
            validation_results['macro_f1_threshold'] = (
                metrics.get('macro_f1', 0) >= sent_thresholds['min_macro_f1']
            )
        
        if 'min_class_f1' in sent_thresholds:
            # Check if all classes meet minimum F1 threshold
            per_class = metrics.get('per_class', {})
            class_f1s = [per_class[label]['f1'] for label in per_class.keys()]
            validation_results['class_f1_threshold'] = all(
                f1 >= sent_thresholds['min_class_f1'] for f1 in class_f1s
            )
        
        if 'max_prevalence_gap' in sent_thresholds:
            validation_results['prevalence_gap_threshold'] = (
                metrics.get('max_gap', 0) <= sent_thresholds['max_prevalence_gap']
            )
    
    return validation_results
