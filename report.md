# Topic Volume & Sentiment Validation Report

**Volume Validation Status: ✅ PASS**

**Overall Sentiment Validation Accuracy: 81.6%**

## Classification Summary

- **Total tickets**: 678
- **Classification mode**: openai
- **Average topic confidence**: 0.858
- **Average sentiment confidence**: 0.887
- **Correct sentiments**: 553/678
- **Merged topics into 'Other'**: Cancellation & Refunds, Payment Problems, Search & Filtering, Interface & Navigation, Booking Process Issues

## Topic Distribution

| Topic | Count | Percentage |
|-------|-------|------------|
| Other | 306 | 45.1% |
| App Stability & Performance | 135 | 19.9% |
| Unexpected Charges & Pricing | 121 | 17.8% |
| Customer Service | 116 | 17.1% |

## Volume Validation

**Status**: ✅ PASS

**Max difference allowed**: 15.0 percentage points

- **Total topics**: 9
- **Passed topics**: 9
- **Failed topics**: 0

### Per-Topic Results

| Topic | System % | Reference % | Difference | Status |
|-------|----------|-------------|------------|--------|
| App Stability & Performance | 19.9% | 10.0% | 9.9pp | ✅ |
| Unexpected Charges & Pricing | 17.8% | 15.0% | 2.8pp | ✅ |
| Customer Service | 17.1% | 8.0% | 9.1pp | ✅ |
| Booking Process Issues | 15.8% | 25.0% | 9.2pp | ✅ |
| Interface & Navigation | 13.1% | 13.0% | 0.1pp | ✅ |
| Search & Filtering | 10.2% | 0.0% | 10.2pp | ✅ |
| Payment Problems | 3.1% | 0.0% | 3.1pp | ✅ |
| Cancellation & Refunds | 2.9% | 5.0% | 2.1pp | ✅ |
| Data & Privacy | 0.0% | 0.0% | 0.0pp | ✅ |

## Per-Topic Sentiment Analysis

### Other

- **Total tickets**: 306
- **Sentiment distribution**:
  - Neutral: 118 (38.6%)
  - Positive: 101 (33.0%)
  - Negative: 87 (28.4%)
- **Sentiment validation accuracy**: 83.7%
  - Correct: 256
  - Incorrect: 50
  - Avg confidence: 0.891
- **Merged topics**: Booking Process Issues, Search & Filtering, Cancellation & Refunds, Payment Problems, Interface & Navigation

### App Stability & Performance

- **Total tickets**: 135
- **Sentiment distribution**:
  - Negative: 54 (40.0%)
  - Neutral: 53 (39.3%)
  - Positive: 28 (20.7%)
- **Sentiment validation accuracy**: 73.3%
  - Correct: 99
  - Incorrect: 36
  - Avg confidence: 0.882

### Unexpected Charges & Pricing

- **Total tickets**: 121
- **Sentiment distribution**:
  - Negative: 52 (43.0%)
  - Positive: 44 (36.4%)
  - Neutral: 25 (20.7%)
- **Sentiment validation accuracy**: 78.5%
  - Correct: 95
  - Incorrect: 26
  - Avg confidence: 0.892

### Customer Service

- **Total tickets**: 116
- **Sentiment distribution**:
  - Positive: 59 (50.9%)
  - Neutral: 29 (25.0%)
  - Negative: 28 (24.1%)
- **Sentiment validation accuracy**: 88.8%
  - Correct: 103
  - Incorrect: 13
  - Avg confidence: 0.874

## Recommendations

### Sentiment Validation Issues

The following topics have low sentiment validation accuracy:

- **App Stability & Performance**: 73.3%
- **Unexpected Charges & Pricing**: 78.5%

**Actions:**
- Review sentiment labels for these topics
- Consider if sentiment definitions are appropriate for topic context
- Verify sentiment labeling consistency

### General Improvements

- Regularly update and validate reference distributions
- Monitor classification confidence scores over time
- Consider topic-specific sentiment validation rules
- Increase sample size for topics with low confidence
