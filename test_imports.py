#!/usr/bin/env python3
"""Quick smoke test to verify all imports work after ruff formatting."""

try:
    from src.classify_and_validate import SimpleClassifier
    print("✓ GPT classifier imports OK")
    
    from src.classify_ml import MLClassifier
    print("✓ ML classifier imports OK")
    
    from src.validate_volume_simple import SimpleVolumeValidator
    print("✓ Volume validator imports OK")
    
    from src.generate_report import SimpleReportGenerator
    print("✓ Report generator imports OK")
    
    from src.compare_methods import load_results
    print("✓ Compare methods imports OK")
    
    from src.utils import setup_logging, load_json_data
    print("✓ Utils imports OK")
    
    from src.metrics import jensen_shannon_divergence
    print("✓ Metrics imports OK")
    
    print("\n🎉 All imports successful! Code is working correctly.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    exit(1)
