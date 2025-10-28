# Simple Topic Volume & Sentiment Validation

This project validates a system that classifies Booking.com reviews into 9 topics and validates existing sentiment labels.

## Current implementation

In this project, I built a simplified topic classification and sentiment validation system for Booking.com customer reviews. The system processes 678 review tickets and performs two main tasks:

**Task 1: Topic Classification**  
I implemented a unified classifier that uses GPT-4o-mini to classify reviews into 9 predefined topics (Unexpected Charges & Pricing, App Stability & Performance, Booking Process Issues, Customer Service, Payment Problems, Search & Filtering, Cancellation & Refunds, Interface & Navigation, and Data & Privacy). The key optimization is that topic classification and sentiment validation occur in a single API call rather than separate requests, reducing costs by 50%.

**Task 2: Sentiment Validation**  
Instead of classifying sentiment from scratch, the system validates existing sentiment labels (Positive, Negative, Neutral) by asking GPT-4o-mini whether the pre-labeled sentiment is correct for each review. This approach returns a boolean validation result with confidence scores.

**Key Technical Features:**
- **Response caching**: All API responses are cached using MD5 hashes to avoid redundant calls and reduce costs
- **Fallback mechanism**: Rule-based classification using keyword matching when API calls fail
- **Topic merging**: Automatically merges the 5 least popular topics into an "Other" category to simplify analysis
- **Per-topic sentiment analysis**: Calculates sentiment distribution (Positive/Negative/Neutral counts) and validation accuracy for each topic
- **Volume validation**: Compares the system's topic distribution against reference data with a 15 percentage point threshold per topic

**Architecture:**
The system is structured into three main components: 
- `classify_and_validate.py` (unified classification),
- `validate_volume_simple.py` (topic distribution validation),
- `generate_report.py` (markdown report generation). 
Configuration is simplified with topics defined directly in code rather than YAML files, and the pipeline is orchestrated through a Makefile with targets for classification, validation, and reporting.

**Validation Approach:**
Volume validation uses simple percentage point comparison (threshold: 15pp per topic) for interpretability and actionability - stakeholders can immediately understand which topics diverge and by how much. Sentiment validation focuses on accuracy (percentage of correct validations) and confidence scores rather than distribution matching, as the validation task is to check if existing labels are correct, not to match artificial distributions.

### Results

**Classification Performance:**
- **Total tickets processed**: 678
- **Classification mode**: GPT-4o-mini with OpenAI API
- **Average topic confidence**: 0.858
- **Average sentiment confidence**: 0.887

**Volume Validation:**
- **Status**: ✅ PASS
- **All 9 topics**: Within 15 percentage point threshold
- **Largest difference**: 10.2pp (Search & Filtering)

**Sentiment Validation:**
- **Overall accuracy**: 81.6% (553/678 correct)
- **Best performing topic**: Customer Service (88.8% accuracy)
- **Lowest performing topic**: App Stability & Performance (73.3% accuracy)

**Topic Distribution:**
- **Top 4 topics**: Other (45.1%), App Stability (19.9%), Pricing (17.8%), Customer Service (17.1%)
- **Merged into "Other"**: 5 least common topics (Cancellation & Refunds, Payment Problems, Search & Filtering, Interface & Navigation, Booking Process Issues)

**Key Findings:**
- System successfully validates topic volume distribution against reference data
- Sentiment validation accuracy varies by topic (73-89%), with Customer Service showing highest agreement
- GPT-4o-mini demonstrates strong confidence (0.89 average) in sentiment validation decisions

## Important note

For production I'd make a different design:
Firstly, I'd use tiktoken to estimate costs for labeling a representative sample (e.g., 3000 tickets). For production I'd use Gpt-4o-mini as it is 16x cheaper and sufficient for classification tasks. 

Although GPT models are able to achieve high accuracy as expert annotators and labellers (https://arxiv.org/html/2403.09097v1#S7, https://medium.com/data-science/bootstrapping-labels-with-gpt-4-8dc85ab5026d), after GPT labeling, I'd have humans validate a random sample (5-10%) to check quality and calculate inter-annotator agreement (Cohen's kappa) between GPT and humans. 

Then I'd train on high-confidence GPT labels first and have humans review low-confidence predictions to improve training data. 

Depending on resources, I'd fine-tune sentence-transformers or use a BERT variant (DistilBERT or ALBERT). As part of a continuous improvement, I'd set up a flagging system for predictions with lower confidence for human review, use corrections to retrain model monthly or bi-monthly and conduct A/B tests before deployment. 


## Technical details

### Data Structure

- **Input**: `booking_reviews_678_varlen_clean.json` (678 tickets)
- **Reference**: `results_booking_reviews_678_varlen_clean.json` (gold standard analysis)
- **Fields**: `original_message`, `message_text`, `sentiment__filter`
- **Derived**: `topic__pred`, `topic_confidence`, `sentiment_validation`

### 9 Topic Categories

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

The 5 least common topics are merged into category 'Other'.

## Installation

1. **Run setup (installs dependencies and creates directories):**
```bash
make setup
```

Or manually:
```bash
pip install -r requirements.txt
mkdir -p data/derived data/artifacts
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

