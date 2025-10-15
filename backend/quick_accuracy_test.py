"""
Quick accuracy test with lower confidence threshold
"""

from enhanced_text_analysis_service import EnhancedTextAnalysisService, EnhancedAnalysisConfig

print("🔧 Quick Accuracy Test with Lower Threshold")
print("=" * 50)

# Create service with lower confidence threshold
config = EnhancedAnalysisConfig(
    confidence_threshold=50.0,  # Much lower threshold
    max_processing_time=5.0
)

service = EnhancedTextAnalysisService(config)

# Simple test cases
test_cases = [
    "I have experence in Python programing.",
    "I work with Javascrip and HTML developement.",
    "I am proficent in React framwork.",
    "I understand databse design and querry optimization."
]

for i, text in enumerate(test_cases, 1):
    print(f"\n--- Test {i} ---")
    print(f"Text: {text}")
    
    result = service.analyze_text(text)
    
    print(f"Detected {len(result.typos)} typos:")
    for typo in result.typos:
        print(f"  - '{typo.word}' → '{typo.suggestion}' ({typo.confidence_score:.1f}%)")
    
    if not result.typos:
        print("  No typos detected")

service.cleanup()
print("\n✅ Test completed")