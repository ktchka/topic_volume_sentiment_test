# Simple Topic Volume & Sentiment Validation Makefile

.PHONY: help setup classify classify-rules classify-ml train-ml validate-volume generate-report compare all all-ml clean

# Default target
help:
	@echo "Simple Topic Volume & Sentiment Validation"
	@echo "=========================================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup            - Install dependencies and create directories"
	@echo "  classify         - Classify topics and validate sentiment (GPT-4o-mini)"
	@echo "  classify-rules   - Classify topics and validate sentiment with rules fallback"
	@echo "  train-ml         - Train ML model on GPT-labeled data"
	@echo "  classify-ml      - Classify using trained ML model (TF-IDF + LR)"
	@echo "  validate-volume  - Validate topic volume distribution"
	@echo "  generate-report  - Generate validation report"
	@echo "  compare          - Compare GPT vs ML classification methods"
	@echo "  all              - Run complete validation pipeline (GPT)"
	@echo "  all-ml           - Run complete validation pipeline (ML)"
	@echo "  clean            - Clean up generated files"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Set OPENAI_API_KEY environment variable (for OpenAI mode)"
	@echo "  - Run 'make setup' first"

# Setup project
setup:
	@echo "Setting up project with uv..."
	uv venv
	uv pip install -r requirements.txt
	mkdir -p data/derived data/artifacts
	@echo "Setup complete!"
	@echo "To activate the virtual environment, run: source .venv/bin/activate"

# Classify topics and validate sentiment (ONE API call)
classify:
	@echo "Classifying topics and validating sentiment (ONE API call)..."
	uv run python src/classify_and_validate.py \
		--input booking_reviews_678_varlen_clean.json \
		--output data/derived/classifications.json \
		--mode openai
	@echo "Classification complete!"

# Classify with rules fallback
classify-rules:
	@echo "Classifying topics and validating sentiment with rules fallback..."
	uv run python src/classify_and_validate.py \
		--input booking_reviews_678_varlen_clean.json \
		--output data/derived/classifications.json \
		--mode rules
	@echo "Classification complete!"

# Train ML model on GPT-labeled data
train-ml:
	@echo "Training ML model on GPT-labeled data..."
	uv run python src/classify_ml.py train \
		--gpt-results data/derived/classifications.json \
		--output data/artifacts/ml_training_report.json
	@echo "ML model training complete!"

# Classify using trained ML model
classify-ml:
	@echo "Classifying topics using trained ML model..."
	uv run python src/classify_ml.py classify \
		--input booking_reviews_678_varlen_clean.json \
		--output data/derived/classifications_ml.json
	@echo "ML classification complete!"

# Validate topic volume distribution
validate-volume:
	@echo "Validating topic volume distribution..."
	uv run python src/validate_volume_simple.py \
		--input data/derived/classifications.json \
		--reference results_booking_reviews_678_varlen_clean.json \
		--output data/artifacts/volume_validation.json
	@echo "Volume validation complete!"

# Generate validation report
generate-report:
	@echo "Generating validation report..."
	uv run python src/generate_report.py \
		--artifacts data/artifacts/ \
		--output report.md
	@echo "Report generated: report.md"

# Compare classification methods
compare:
	@echo "Comparing classification methods..."
	uv run python src/compare_methods.py \
		--gpt data/derived/classifications.json \
		--ml data/derived/classifications_ml.json \
		--output data/artifacts/method_comparison.json
	@echo "Comparison complete!"

# Run complete validation pipeline
all: setup classify validate-volume generate-report
	@echo "Complete validation pipeline finished!"
	@echo "Check report.md for results"

# Run complete validation pipeline with rules fallback
all-rules: setup classify-rules validate-volume generate-report
	@echo "Complete validation pipeline (rules) finished!"
	@echo "Check report.md for results"

# Run complete validation pipeline with ML
all-ml: setup classify train-ml classify-ml
	@echo "Training ML model..."
	@make train-ml
	@echo "Running ML classification and validation..."
	uv run python src/validate_volume_simple.py \
		--input data/derived/classifications_ml.json \
		--reference results_booking_reviews_678_varlen_clean.json \
		--output data/artifacts/volume_validation_ml.json
	uv run python src/generate_report.py \
		--artifacts data/artifacts/ \
		--classifications data/derived/classifications_ml.json \
		--output report_ml.md
	@echo "Complete ML validation pipeline finished!"
	@echo "Check report_ml.md for results"

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -rf data/derived/*.json
	rm -rf data/artifacts/*.json
	rm -rf data/artifacts/*.png
	rm -f report.md
	@echo "Cleanup complete!"

# Quick test run
test:
	@echo "Running quick test..."
	uv run python src/classify_and_validate.py \
		--input booking_reviews_678_varlen_clean.json \
		--output data/derived/test_classifications.json \
		--mode rules
	uv run python src/validate_volume_simple.py \
		--input data/derived/test_classifications.json \
		--reference results_booking_reviews_678_varlen_clean.json \
		--output data/artifacts/test_volume.json
	@echo "Test run complete!"