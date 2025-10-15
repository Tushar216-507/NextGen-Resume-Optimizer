"""
Quick demonstration of the Enhanced Typo Detection System
"""

from enhanced_text_analysis_service import EnhancedTextAnalysisService, EnhancedAnalysisConfig
import time

print('🚀 Enhanced Typo Detection System - Quick Demo')
print('=' * 50)

# Create service
config = EnhancedAnalysisConfig(confidence_threshold=70.0)
service = EnhancedTextAnalysisService(config)

# Test text with obvious errors
test_text = 'I have experence in Python and Javascript. Me and my team has completed the project.'

print(f'📝 Analyzing: {test_text}')
print()

start = time.time()
result = service.analyze_text(test_text)
end = time.time()

print(f'⏱️  Processing time: {end-start:.2f}s')
print(f'🔍 Layers used: {[layer.value for layer in result.layers_used]}')
print(f'📊 Status: {result.processing_status.value}')
print(f'🎯 Total issues found: {len(result.typos) + len(result.grammar_issues)}')
print()

if result.typos:
    print('🔤 Typos detected:')
    for typo in result.typos:
        print(f'  - "{typo.word}" → "{typo.suggestion}" (confidence: {typo.confidence_score:.1f}%)')

if result.grammar_issues:
    print('📝 Grammar issues detected:')
    for issue in result.grammar_issues:
        print(f'  - {issue.issue_type}: {issue.explanation}')

print()
print('✅ Enhanced system is working!')

# Show capabilities
capabilities = service.get_analysis_capabilities()
print(f'🔧 Available layers: {capabilities["available_layers"]}')
print(f'📈 Total analyses performed: {capabilities["total_analyses_performed"]}')

service.cleanup()
print('🧹 Cleanup completed')