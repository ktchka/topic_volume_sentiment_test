# Simple Topic Volume & Sentiment Validation

This project validates a system that classifies Booking.com reviews into 9 topics and validates existing sentiment labels.

## Current implementation

In this project, I built a simplified topic classification and sentiment validation system for Booking.com customer reviews. The system processes 678 review tickets and performs two main tasks:

**Task 1: Topic Classification**
I implemented **three classification approaches** with different tradeoffs:
1. **GPT-4o-mini**: Uses OpenAI API to classify reviews into 9 predefined topics (Unexpected Charges & Pricing, App Stability & Performance, Booking Process Issues, Customer Service, Payment Problems, Search & Filtering, Cancellation & Refunds, Interface & Navigation, and Data & Privacy). The key optimization is that topic classification and sentiment validation occur in a single API call rather than separate requests, reducing costs by 50%.
2. **ML (TF-IDF + Logistic Regression)**: Trains on GPT-labeled data (542 train, 136 test) and achieves 94.9% topic classification accuracy. Provides 100x faster inference at zero marginal cost, suitable for production scale deployment.
3. **Rule-based**: Keyword matching fallback when API calls fail or for simple cases.

**Task 2: Sentiment Validation**
Instead of classifying sentiment from scratch, the system validates existing sentiment labels (Positive, Negative, Neutral) by asking GPT-4o-mini whether the pre-labeled sentiment is correct for each review. This approach returns a boolean validation result with confidence scores. The ML variant uses simple rule-based heuristics for sentiment validation.

**Key Technical Features:**
- **Response caching**: All API responses are cached using MD5 hashes to avoid redundant calls and reduce costs
- **Multiple classification modes**: GPT (accurate, expensive), ML (fast, cheap), Rules (fastest, fallback)
- **ML training pipeline**: Automatically trains on GPT-labeled data to create production-ready classifier
- **Topic merging**: Automatically merges the 5 least popular topics into an "Other" category to simplify analysis
- **Per-topic sentiment analysis**: Calculates sentiment distribution (Positive/Negative/Neutral counts) and validation accuracy for each topic
- **Volume validation**: Compares the system's topic distribution against reference data with a 15 percentage point threshold per topic

**Architecture:**
The system is structured into four main components:
- `classify_and_validate.py` (GPT + rule-based classification),
- `classify_ml.py` (ML classifier with training pipeline),
- `validate_volume_simple.py` (topic distribution validation),
- `generate_report.py` (markdown report generation).

Configuration is simplified with topics defined directly in code rather than YAML files, and the pipeline is orchestrated through a Makefile with targets for classification, validation, and reporting.

**Validation Approach:**
Volume validation uses simple percentage point comparison (threshold: 15pp per topic) for interpretability and actionability - stakeholders can immediately understand which topics diverge and by how much. Sentiment validation focuses on accuracy (percentage of correct validations) and confidence scores rather than distribution matching, as the validation task is to check if existing labels are correct, not to match artificial distributions.

> **ML Data Limitation:** The ML model was trained on only 678 tickets (insufficient for making any decisions). The 94.9% test accuracy is a proof-of-concept demonstrating feasibility, not production-validated performance.

### Methods Comparison

| Method | Topic Accuracy | Sentiment Validation | Speed | Cost | When to Use |
|--------|----------------|---------------------|-------|------|-------------|
| **GPT-4o-mini** | Baseline | 81.6% | ~30-60s | ~$0.30/1k | High-stakes, complex cases |
| **ML (TF-IDF + LR)** | 94.9% | 100.0% | <1s | Free* | Production scale (>10k/month) |
| **Rules** | ~60-70% | ~100%** | <1s | Free | Fallback only |

*ML requires initial training on GPT-labeled data
**Rules achieve 100% because they accept all existing labels as correct
*ML approach requires initial training on GPT-labeled data

> The details can be read in the [GPT report](report.md) and [ML report](report_ml.md)

## Important note

For production I'd make a different design:
Firstly, I'd use tiktoken to estimate costs for labeling a representative sample (e.g., 3000 tickets). For production I'd use Gpt-4o-mini as it is 16x cheaper and sufficient for classification tasks.

Although GPT models are able to achieve high accuracy as expert annotators and labellers (https://arxiv.org/html/2403.09097v1#S7, https://medium.com/data-science/bootstrapping-labels-with-gpt-4-8dc85ab5026d), after GPT labeling, I'd have humans validate a random sample (5-10%) to check quality and calculate inter-annotator agreement (Cohen's kappa) between GPT and humans. For the automatic selection of the best prompt, I'd use [DSPy](https://dspy.ai/).

Then I'd train on high-confidence GPT labels first and have humans review low-confidence predictions to improve training data.

Depending on resources, I'd fine-tune sentence-transformers or use a BERT variant (DistilBERT or ALBERT). As part of a continuous improvement, I'd set up a flagging system for predictions with lower confidence for human review, use corrections to retrain model monthly or bi-monthly and conduct A/B tests before deployment.

**Why not in this test?**
- 8-hour constraint: prioritized working end-to-end pipeline
- ML model would require labeled training set (could use GPT-labeled data, but adds complexity) and more data
- Focus on system design, validation pipeline, and code quality

## Technical details


### Key Technical Decisions

**Why single API call?**
Combining topic + sentiment in one call reduces cost by 50% and latency by ~50%.

**Why simple validation metrics?**
15pp threshold and accuracy % are interpretable by stakeholders. Complex statistical metrics (JSD, TVD) were deemed unnecessary for this use case. Moreover, I believe that ~600 messages are not enough to measure statistical significance of more complex metrics.

**Why ML variant?**
Demonstrates production scalability path: GPT for initial labeling → ML for cost-effective deployment at scale.

**Why topic merging?**
Rare topics (<5% each) merged to "Other" prevents metric skewing and focuses analysis on major issues.
9 predefined categories: Unexpected Charges & Pricing, App Stability & Performance, Booking Process Issues, Customer Service, Payment Problems, Search & Filtering, Cancellation & Refunds, Interface & Navigation, Data & Privacy.

### Installation

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

### Usage

#### Quick Start

```bash
# Run complete pipeline with GPT-4o-mini (requires API key)
make all

# Run complete pipeline with rule-based classification (no API key needed)
make all-rules

# Run complete pipeline with ML model (requires training first)
make all-ml

# Run individual steps
make classify         # Classify topics and validate sentiment (GPT)
make classify-rules   # Classify with rules fallback
make train-ml         # Train ML model on GPT-labeled data
make classify-ml      # Classify using trained ML model
make compare          # Compare GPT vs ML methods
make validate-volume  # Validate topic volume distribution
make generate-report  # Generate report
```

## Development

### Code quality
```bash
make lint          # Run linter with auto-fix
make format        # Format code
make check-format  # Check without modifying
```

### Pre-commit hooks
Hooks run automatically on `git commit`. To run manually:
```bash
pre-commit run --all-files
```

The project uses:
- **ruff**: Fast Python linter and formatter (10-100x faster than pylint/flake8/black)
- **pre-commit**: Automatic checks on commit (linting, formatting, YAML/JSON validation)

## Output files
### Output files

`report.md` / `report_ml.md` - Validation reports
