# Simple Topic Volume & Sentiment Validation

This project validates a system that classifies Booking.com reviews into 9 topics and validates existing sentiment labels.

## Overview

The system processes 678 Booking.com review tickets and:
1. **Classifies topics** using OpenAI API (ONE API call per ticket)
2. **Validates existing sentiment labels** - checks if the pre-labeled sentiment makes sense
3. **Validates topic volume distribution** against reference distributions

## Key Simplifications

- **ONE API call per ticket** - Classifies topic AND validates sentiment in single call
- **Topics defined in code** - No complex YAML configuration files
- **Simple validation** - Just pass/fail based on reasonable thresholds
- **No complex metrics** - Focus on practical validation results

## Data Structure

- **Input**: `booking_reviews_678_varlen_clean.json` (678 tickets)
- **Reference**: `results_booking_reviews_678_varlen_clean.json` (gold standard analysis)
- **Fields**: `original_message`, `message_text`, `sentiment__filter`
- **Derived**: `topic__pred`, `topic_confidence`, `sentiment_validation`

## 9 Topic Categories

Topics are defined directly in code (no YAML needed):

1. **Unexpected Charges & Pricing** - Price changes, hidden fees, promo code failures
2. **App Stability & Performance** - App crashes, freezing, slow loading
3. **Booking Process Issues** - Problems with the booking flow
4. **Customer Service** - Support interactions and service quality
5. **Payment Problems** - Transaction failures, billing issues
6. **Search & Filtering** - Search functionality and filtering options
7. **Cancellation & Refunds** - Cancellation process and refund requests
8. **Interface & Navigation** - UI/UX issues and navigation problems
9. **Data & Privacy** - Data handling and privacy concerns

## Project Structure

```
src/
  classify_and_validate.py  # ONE file: topics + sentiment validation
  validate_volume_simple.py # Simple volume validation
  generate_report.py        # Simple validation report
  metrics.py                # Basic statistical metrics
  utils.py                  # Common utilities
data/
  booking_reviews_678_varlen_clean.json  # Input data
  results_booking_reviews_678_varlen_clean.json  # Reference data
  derived/classifications.json           # Classification results
  artifacts/volume_validation.json       # Volume validation results
  artifacts/classification_cache.json    # API response cache
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set OpenAI API key (optional - for OpenAI classification):**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Quick Start

```bash
# Run complete pipeline with OpenAI API (requires API key)
make all

# Run complete pipeline with rule-based classification (no API key needed)
make all-rules

# Run individual steps
make classify         # Classify topics and validate sentiment
make classify-rules   # Classify with rules fallback
make validate-volume  # Validate topic volume distribution
make generate-report  # Generate report
```

### Step-by-Step Usage

#### Step 1: Classify Topics and Validate Sentiment
```bash
# Using OpenAI API (default, requires API key)
python src/classify_and_validate.py \
  --input booking_reviews_678_varlen_clean.json \
  --output data/derived/classifications.json \
  --mode openai

# Using rule-based approach (no API key needed)
python src/classify_and_validate.py \
  --input booking_reviews_678_varlen_clean.json \
  --output data/derived/classifications.json \
  --mode rules
```

#### Step 2: Validate Topic Volume Distribution
```bash
python src/validate_volume_simple.py \
  --input data/derived/classifications.json \
  --reference results_booking_reviews_678_varlen_clean.json \
  --output data/artifacts/volume_validation.json
```

#### Step 3: Generate Report
```bash
python src/generate_report.py \
  --artifacts data/artifacts/ \
  --output report.md
```

## API Call Structure

Each ticket gets ONE API call that returns:

```json
{
  "topic": "Unexpected Charges & Pricing",
  "topic_confidence": 0.85,
  "sentiment_validation": {
    "is_correct": true,
    "confidence": 0.90
  }
}
```

## Validation Logic

### Topic Volume Validation
- Compare topic distribution against reference
- Allow up to 15% point difference per topic
- Simple pass/fail result

### Sentiment Validation
- Check if existing sentiment labels make sense for the review content
- Provide confidence scores for validation
- Track overall sentiment accuracy

### Advanced Analysis

#### Topic Merging
- Automatically merges the **5 least popular topics** into an "Other" category
- Preserves original topic information for reference
- Simplifies analysis by focusing on major topics

#### Per-Topic Sentiment Analysis
For each topic (including "Other"), the system calculates:
- **Sentiment distribution**: How many Positive/Negative/Neutral reviews per topic
- **Validation accuracy**: What % of sentiment labels are correct for this topic
- **Confidence scores**: Average confidence in sentiment validation
- **Merged topics info**: Which topics were merged into "Other"

Example output:
```json
{
  "App Stability & Performance": {
    "total_tickets": 100,
    "sentiment_distribution": {
      "Negative": 85,
      "Neutral": 10,
      "Positive": 5
    },
    "sentiment_validation": {
      "correct": 92,
      "incorrect": 8,
      "accuracy": 0.92,
      "avg_confidence": 0.88
    }
  },
  "Other": {
    "total_tickets": 50,
    "merged_topics": ["Data & Privacy", "Search & Filtering", ...],
    "sentiment_distribution": {...},
    "sentiment_validation": {...}
  }
}
```

## Caching

The system caches OpenAI API responses using MD5 hashes to avoid redundant API calls, reducing costs and improving performance.

## Output Files

- `data/derived/classifications.json` - Classification results
- `data/artifacts/volume_validation.json` - Volume validation results
- `report.md` - Human-readable validation report