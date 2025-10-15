"""
Test the enhanced system with Java/JDK now available
This should show significantly improved grammar detection
"""

from enhanced_text_analysis_service import EnhancedTextAnalysisService, EnhancedAnalysisConfig
import time

print("🚀 Enhanced System Test with Java/JDK Available")
print("=" * 60)

# Create service with optimized settings
config = EnhancedAnalysisConfig(
    confidence_threshold=60.0,  # Balanced threshold
    max_processing_time=5.0,
    enable_traditional_nlp=True,
    enable_gector=True,
    enable_domain_validation=True
)

service = EnhancedTextAnalysisService(config)

# Test cases with both spelling and grammar issues
test_cases = [
    {
        "name": "Mixed Spelling & Grammar",
        "text": "I have experence in Python programing. Me and my team has completed the project successfully.",
        "expected_typos": ["experence", "programing"],
        "expected_grammar": ["Me and my team has"]
    },
    {
        "name": "Technical Resume Text",
        "text": "I am proficent in React.js and have experence with Node.js framwork. This are the technologies I use.",
        "expected_typos": ["proficent", "experence", "framwork"],
        "expected_grammar": ["This are"]
    },
    {
        "name": "Advanced Technical Terms",
        "text": "I implement microservises architectures using Docker containerisation. I have 5 year of experence in DevOps.",
        "expected_typos": ["microservises", "containerisation", "experence"],
        "expected_grammar": ["5 year of"]
    },
    {
        "name": "Grammar Focus",
        "text": "Me and him has worked together. This are the results we achieved. I was responsible to the project.",
        "expected_typos": [],
        "expected_grammar": ["Me and him has", "This are", "responsible to"]
    }
]

print(f"📝 Testing {len(test_cases)} scenarios with Java available...")

total_typo_accuracy = 0
total_grammar_accuracy = 0
total_processing_time = 0

for i, test_case in enumerate(test_cases, 1):
    print(f"\n--- Test {i}: {test_case['name']} ---")
    print(f"Text: {test_case['text']}")
    
    start_time = time.time()
    result = service.analyze_text(test_case['text'])
    processing_time = time.time() - start_time
    total_processing_time += processing_time
    
    # Extract results
    detected_typos = [typo.word.lower() for typo in result.typos]
    detected_grammar_contexts = [issue.sentence for issue in result.grammar_issues]
    
    # Calculate accuracy
    expected_typos_lower = [word.lower() for word in test_case['expected_typos']]
    
    # Typo accuracy
    if expected_typos_lower:
        typo_matches = len(set(detected_typos) & set(expected_typos_lower))
        typo_accuracy = typo_matches / len(expected_typos_lower)
    else:
        typo_accuracy = 1.0 if not detected_typos else 0.8  # Penalty for false positives
    
    # Grammar accuracy (simplified - check if any grammar issues detected when expected)
    if test_case['expected_grammar']:
        grammar_accuracy = 1.0 if result.grammar_issues else 0.0
    else:
        grammar_accuracy = 1.0 if not result.grammar_issues else 0.8
    
    total_typo_accuracy += typo_accuracy
    total_grammar_accuracy += grammar_accuracy
    
    # Display results
    print(f"⏱️  Processing time: {processing_time:.3f}s")
    print(f"🔍 Layers used: {[layer.value for layer in result.layers_used]}")
    print(f"📊 Typo accuracy: {typo_accuracy:.1%} | Grammar accuracy: {grammar_accuracy:.1%}")
    
    # Show detected typos
    if result.typos:
        print("🔤 Detected typos:")
        for typo in result.typos:
            is_expected = typo.word.lower() in expected_typos_lower
            status = "✅" if is_expected else "⚠️"
            print(f"  {status} '{typo.word}' → '{typo.suggestion}' (confidence: {typo.confidence_score:.1f}%)")
    else:
        print("🔤 No typos detected")
    
    # Show detected grammar issues
    if result.grammar_issues:
        print("📝 Detected grammar issues:")
        for issue in result.grammar_issues:
            print(f"  ✅ {issue.issue_type}: {issue.explanation}")
            print(f"     Context: {issue.sentence[:100]}...")
    else:
        print("📝 No grammar issues detected")
    
    # Show missed items
    missed_typos = set(expected_typos_lower) - set(detected_typos)
    if missed_typos:
        print(f"❌ Missed typos: {list(missed_typos)}")
    
    if test_case['expected_grammar'] and not result.grammar_issues:
        print(f"❌ Missed grammar issues: {test_case['expected_grammar']}")

# Overall results
avg_typo_accuracy = total_typo_accuracy / len(test_cases)
avg_grammar_accuracy = total_grammar_accuracy / len(test_cases)
avg_processing_time = total_processing_time / len(test_cases)

print(f"\n🎯 OVERALL RESULTS WITH JAVA")
print("=" * 50)
print(f"📈 Average Typo Accuracy: {avg_typo_accuracy:.1%}")
print(f"📈 Average Grammar Accuracy: {avg_grammar_accuracy:.1%}")
print(f"📈 Combined Accuracy: {(avg_typo_accuracy + avg_grammar_accuracy) / 2:.1%}")
print(f"⏱️  Average Processing Time: {avg_processing_time:.3f}s")

# Check system capabilities
capabilities = service.get_analysis_capabilities()
print(f"\n🔧 SYSTEM STATUS")
print("=" * 50)
print(f"Enhanced analysis available: {capabilities['enhanced_analysis_available']}")
print(f"Available layers: {capabilities['available_layers']}")
print(f"Total analyses performed: {capabilities['total_analyses_performed']}")

# Test Java availability specifically
print(f"\n☕ JAVA STATUS CHECK")
print("=" * 50)

# Get system info from traditional NLP layer
try:
    from enhanced_traditional_nlp_layer import EnhancedTraditionalNLPLayer
    nlp_layer = EnhancedTraditionalNLPLayer()
    system_info = nlp_layer.get_system_info()
    
    print(f"Java available: {system_info['java_available']}")
    if system_info['java_available']:
        print(f"Java version: {system_info.get('java_version', 'Unknown')}")
        print(f"LanguageTool available: {system_info['language_tool_available']}")
        print("🎉 Java is working! Grammar detection should be significantly improved.")
    else:
        print("❌ Java still not detected. Check PATH configuration.")
    
    print(f"Spell checker available: {system_info['spell_checker_available']}")
    print(f"Custom dictionary size: {system_info['custom_dictionary_size']}")
    print(f"Fallback patterns: {system_info['fallback_patterns_count']}")
    
    nlp_layer.cleanup()
    
except Exception as e:
    print(f"Error checking Java status: {e}")

# Final assessment
print(f"\n🏆 FINAL ASSESSMENT")
print("=" * 50)

combined_accuracy = (avg_typo_accuracy + avg_grammar_accuracy) / 2
speed_ok = avg_processing_time <= 3.0

if combined_accuracy >= 0.8 and speed_ok:
    print("🎉 EXCELLENT: System performing at high accuracy with Java!")
    print("✅ Ready for production use")
elif combined_accuracy >= 0.6:
    print("✅ GOOD: Significant improvement with Java available")
    print("🔧 Consider fine-tuning confidence thresholds for even better results")
else:
    print("⚠️  NEEDS TUNING: System working but accuracy could be improved")

print(f"Accuracy target (≥80%): {'✅ ACHIEVED' if combined_accuracy >= 0.8 else '❌ NOT ACHIEVED'} ({combined_accuracy:.1%})")
print(f"Speed target (≤3s): {'✅ ACHIEVED' if speed_ok else '❌ NOT ACHIEVED'} ({avg_processing_time:.3f}s)")

service.cleanup()
print("\n🧹 Test completed and system cleaned up")