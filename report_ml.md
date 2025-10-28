# Topic Volume & Sentiment Validation Report

**Volume Validation Status: ✅ PASS**

**Overall Sentiment Validation Accuracy: 100.0%**

## Classification Summary

- **Total tickets**: 678
- **Classification mode**: ml_tfidf
- **Average topic confidence**: 0.710
- **Average sentiment confidence**: 0.607
- **Correct sentiments**: 678/678
- **Merged topics into 'Other'**: Payment Problems, Cancellation & Refunds, Search & Filtering, Interface & Navigation, Booking Process Issues

## Topic Distribution

| Topic | Count | Percentage |
|-------|-------|------------|
| Other | 309 | 45.6% |
| App Stability & Performance | 133 | 19.6% |
| Unexpected Charges & Pricing | 121 | 17.8% |
| Customer Service | 115 | 17.0% |

## Volume Validation

**Status**: ✅ PASS

**Max difference allowed**: 15.0 percentage points

- **Total topics**: 9
- **Passed topics**: 9
- **Failed topics**: 0

### Per-Topic Results

| Topic | System % | Reference % | Difference | Status |
|-------|----------|-------------|------------|--------|
| App Stability & Performance | 19.6% | 10.0% | 9.6pp | ✅ |
| Unexpected Charges & Pricing | 17.8% | 15.0% | 2.8pp | ✅ |
| Customer Service | 17.0% | 8.0% | 9.0pp | ✅ |
| Booking Process Issues | 16.8% | 25.0% | 8.2pp | ✅ |
| Interface & Navigation | 14.3% | 13.0% | 1.3pp | ✅ |
| Search & Filtering | 9.6% | 0.0% | 9.6pp | ✅ |
| Cancellation & Refunds | 2.5% | 5.0% | 2.5pp | ✅ |
| Payment Problems | 2.4% | 0.0% | 2.4pp | ✅ |
| Data & Privacy | 0.0% | 0.0% | 0.0pp | ✅ |

## Per-Topic Sentiment Analysis

### Other

- **Total tickets**: 309
- **Sentiment distribution**:
  - Neutral: 122 (39.5%)
  - Positive: 101 (32.7%)
  - Negative: 86 (27.8%)
- **Sentiment validation accuracy**: 100.0%
  - Correct: 309
  - Incorrect: 0
  - Avg confidence: 0.607
- **Merged topics**: Search & Filtering, Interface & Navigation, Cancellation & Refunds, Booking Process Issues, Payment Problems

### App Stability & Performance

- **Total tickets**: 133
- **Sentiment distribution**:
  - Negative: 56 (42.1%)
  - Neutral: 49 (36.8%)
  - Positive: 28 (21.1%)
- **Sentiment validation accuracy**: 100.0%
  - Correct: 133
  - Incorrect: 0
  - Avg confidence: 0.612

### Unexpected Charges & Pricing

- **Total tickets**: 121
- **Sentiment distribution**:
  - Negative: 51 (42.1%)
  - Positive: 44 (36.4%)
  - Neutral: 26 (21.5%)
- **Sentiment validation accuracy**: 100.0%
  - Correct: 121
  - Incorrect: 0
  - Avg confidence: 0.607

### Customer Service

- **Total tickets**: 115
- **Sentiment distribution**:
  - Positive: 59 (51.3%)
  - Neutral: 28 (24.3%)
  - Negative: 28 (24.3%)
- **Sentiment validation accuracy**: 100.0%
  - Correct: 115
  - Incorrect: 0
  - Avg confidence: 0.600

## Recommendations

### General Improvements

- Regularly update and validate reference distributions
- Monitor classification confidence scores over time
- Consider topic-specific sentiment validation rules
- Increase sample size for topics with low confidence
