"""
Comprehensive test for the enhanced typo detection system.
Tests all components and validates world-class accuracy.
"""

import time
import logging
from typing import List, Dict, Any

# Configure logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_system():
    """Test the complete enhanced system"""
    print("🚀 Starting Enhanced Typo Detection System Test")
    print("=" * 60)
    
    try:
        # Import the enhanced service
        from enhanced_text_analysis_service import EnhancedTextAnalysisService, EnhancedAnalysisConfig
        
        print("✅ Enhanced service imports successful")
        
        # Create enhanced service
        config = EnhancedAnalysisConfig(
            enable_traditional_nlp=True,
            enable_gector=True,
            enable_domain_validation=True,
            confidence_threshold=75.0,
            max_processing_time=5.0,
            parallel_processing=True,
            cache_enabled=True
        )
        
        service = EnhancedTextAnalysisService(config)
        print("✅ Enhanced service initialized")
        
        # Test cases with known errors
        test_cases = [
            {
                "name": "Technical Resume Text",
                "text": "I have experence in Python, Javascript, and React framwork. I worked with AWS and Docker containrs.",
                "expected_typos": ["experence", "framwork", "containrs"],
                "expected_grammar": []
            },
            {
                "name": "Grammar Issues",
                "text": "Me and my team has completed the project. This are the results we achieved.",
                "expected_typos": [],
                "expected_grammar": ["Me and my team has", "This are"]
            },
            {
                "name": "Mixed Issues",
                "text": "I am proficent in machine learnin and have 5 year of experence in data scince.",
                "expected_typos": ["proficent", "learnin", "experence", "scince"],
                "expected_grammar": ["5 year of"]
            },
            {
                "name": "Clean Text",
                "text": "I have extensive experience in software development using Python and JavaScript.",
                "expected_typos": [],
                "expected_grammar": []
            }
        ]
        
        print(f"\n📝 Running {len(test_cases)} test cases...")
        
        total_accuracy = 0.0
        total_processing_time = 0.0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {test_case['name']} ---")
            print(f"Text: {test_case['text']}")
            
            start_time = time.time()
            
            # Analyze text
            result = service.analyze_text(
                text=test_case['text'],
                check_spelling=True,
                check_grammar=True
            )
            
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            # Extract detected issues
            detected_typos = [typo.word for typo in result.typos]
            detected_grammar = [issue.sentence for issue in result.grammar_issues]
            
            print(f"⏱️  Processing time: {processing_time:.3f}s")
            print(f"🔍 Layers used: {[layer.value for layer in result.layers_used]}")
            print(f"📊 Status: {result.processing_status.value}")
            print(f"💾 Cache hits: {result.cache_hits}")
            
            # Calculate accuracy
            typo_accuracy = calculate_accuracy(detected_typos, test_case['expected_typos'])
            grammar_accuracy = calculate_accuracy(detected_grammar, test_case['expected_grammar'])
            overall_accuracy = (typo_accuracy + grammar_accuracy) / 2
            
            total_accuracy += overall_accuracy
            
            print(f"📈 Typo accuracy: {typo_accuracy:.1%}")
            print(f"📈 Grammar accuracy: {grammar_accuracy:.1%}")
            print(f"📈 Overall accuracy: {overall_accuracy:.1%}")
            
            # Show detected issues
            if result.typos:
                print("🔤 Detected typos:")
                for typo in result.typos:
                    print(f"  - '{typo.word}' → '{typo.suggestion}' (confidence: {typo.confidence_score:.1f}%)")
            
            if result.grammar_issues:
                print("📝 Detected grammar issues:")
                for issue in result.grammar_issues:
                    print(f"  - {issue.issue_type}: {issue.explanation}")
        
        # Overall results
        avg_accuracy = total_accuracy / len(test_cases)
        avg_processing_time = total_processing_time / len(test_cases)
        
        print(f"\n🎯 OVERALL RESULTS")
        print("=" * 40)
        print(f"Average accuracy: {avg_accuracy:.1%}")
        print(f"Average processing time: {avg_processing_time:.3f}s")
        print(f"Target accuracy (85%): {'✅ ACHIEVED' if avg_accuracy >= 0.85 else '❌ NOT ACHIEVED'}")
        print(f"Target speed (<3s): {'✅ ACHIEVED' if avg_processing_time < 3.0 else '❌ NOT ACHIEVED'}")
        
        # Test system capabilities
        print(f"\n🔧 SYSTEM CAPABILITIES")
        print("=" * 40)
        
        capabilities = service.get_analysis_capabilities()
        print(f"Enhanced analysis available: {capabilities['enhanced_analysis_available']}")
        print(f"Available layers: {capabilities['available_layers']}")
        print(f"Cache enabled: {capabilities['cache_enabled']}")
        print(f"Parallel processing: {capabilities['parallel_processing']}")
        
        # Test performance report
        print(f"\n📊 PERFORMANCE REPORT")
        print("=" * 40)
        
        performance = service.get_performance_report()
        service_metrics = performance['service_metrics']
        print(f"Total analyses: {service_metrics['total_analyses']}")
        print(f"Average processing time: {service_metrics['average_processing_time']:.3f}s")
        print(f"Uptime: {service_metrics['uptime_seconds']:.1f}s")
        
        # Test system health
        print(f"\n🏥 SYSTEM HEALTH")
        print("=" * 40)
        
        health = service.validate_system_health()
        print(f"Overall status: {health['overall_status']}")
        if health['issues']:
            print("Issues:")
            for issue in health['issues']:
                print(f"  - {issue}")
        if health['recommendations']:
            print("Recommendations:")
            for rec in health['recommendations']:
                print(f"  - {rec}")
        
        # Test optimization
        print(f"\n⚡ PERFORMANCE OPTIMIZATION")
        print("=" * 40)
        
        optimization_result = service.optimize_performance()
        print(f"Engine optimized: {optimization_result.get('engine_optimized', False)}")
        print(f"Cache optimized: {optimization_result.get('cache_optimized', False)}")
        
        # Final assessment
        print(f"\n🏆 FINAL ASSESSMENT")
        print("=" * 40)
        
        if avg_accuracy >= 0.85 and avg_processing_time < 3.0:
            print("🎉 SUCCESS: Enhanced system meets all requirements!")
            print("✅ Accuracy target achieved (≥85%)")
            print("✅ Performance target achieved (<3s)")
        else:
            print("⚠️  PARTIAL SUCCESS: Some targets not met")
            if avg_accuracy < 0.85:
                print(f"❌ Accuracy below target: {avg_accuracy:.1%} < 85%")
            if avg_processing_time >= 3.0:
                print(f"❌ Performance below target: {avg_processing_time:.3f}s ≥ 3s")
        
        # Cleanup
        service.cleanup()
        print("\n🧹 System cleanup completed")
        
        return {
            'success': avg_accuracy >= 0.85 and avg_processing_time < 3.0,
            'accuracy': avg_accuracy,
            'processing_time': avg_processing_time,
            'capabilities': capabilities,
            'health': health
        }
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return {'success': False, 'error': 'Import failed'}
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def calculate_accuracy(detected: List[str], expected: List[str]) -> float:
    """Calculate accuracy using F1 score"""
    if not expected and not detected:
        return 1.0  # Perfect if both empty
    
    if not expected:
        return 0.0 if detected else 1.0
    
    if not detected:
        return 0.0
    
    # Convert to sets for comparison
    detected_set = set(word.lower() for word in detected)
    expected_set = set(word.lower() for word in expected)
    
    # Calculate precision, recall, F1
    true_positives = len(detected_set & expected_set)
    false_positives = len(detected_set - expected_set)
    false_negatives = len(expected_set - detected_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def test_fallback_scenarios():
    """Test fallback scenarios when dependencies are missing"""
    print("\n🔄 Testing fallback scenarios...")
    
    try:
        from enhanced_traditional_nlp_layer import EnhancedTraditionalNLPLayer
        
        # Test traditional NLP layer (should work without Java)
        layer = EnhancedTraditionalNLPLayer()
        
        test_text = "I have experence in programing and sofware development."
        
        from enhanced_models import AnalysisConfig
        config = AnalysisConfig()
        
        results = layer.detect(test_text, config)
        
        print(f"✅ Traditional NLP fallback working: {len(results)} results")
        
        # Test system info
        system_info = layer.get_system_info()
        print(f"Java available: {system_info['java_available']}")
        print(f"Spell checker available: {system_info['spell_checker_available']}")
        print(f"Fallback patterns: {system_info['fallback_patterns_count']}")
        
        layer.cleanup()
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")

if __name__ == "__main__":
    # Run main test
    result = test_enhanced_system()
    
    # Run fallback tests
    test_fallback_scenarios()
    
    # Print final result
    print(f"\n{'='*60}")
    if result.get('success'):
        print("🎉 ALL TESTS PASSED - ENHANCED SYSTEM IS READY!")
    else:
        print("⚠️  TESTS COMPLETED WITH ISSUES")
    print(f"{'='*60}")