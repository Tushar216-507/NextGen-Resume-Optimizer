"""
Comprehensive IT Technical Terms Accuracy Test
Tests the enhanced typo detection system with IT-related terms from beginner to advanced levels.
"""

import time
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Test case for IT technical terms"""
    level: str
    category: str
    text: str
    expected_typos: List[str]
    expected_corrections: List[str]
    description: str

def create_it_test_cases() -> List[TestCase]:
    """Create comprehensive IT test cases from beginner to advanced"""
    
    return [
        # BEGINNER LEVEL - Basic IT Terms
        TestCase(
            level="Beginner",
            category="Basic Programming",
            text="I have experence in programing with Phyton and Javascrip. I know HTML and CSS for web developement.",
            expected_typos=["experence", "programing", "Phyton", "Javascrip", "developement"],
            expected_corrections=["experience", "programming", "Python", "JavaScript", "development"],
            description="Basic programming languages with common misspellings"
        ),
        
        TestCase(
            level="Beginner",
            category="Web Technologies",
            text="I can create websits using HTML, CSS, and basic Javascrip. I understand responive design principles.",
            expected_typos=["websits", "Javascrip", "responive"],
            expected_corrections=["websites", "JavaScript", "responsive"],
            description="Web development fundamentals"
        ),
        
        TestCase(
            level="Beginner",
            category="Database Basics",
            text="I have worked with SQL databses and understand basic querry operations like SELECT and INSERT.",
            expected_typos=["databses", "querry"],
            expected_corrections=["databases", "query"],
            description="Basic database terminology"
        ),
        
        # INTERMEDIATE LEVEL - Frameworks and Tools
        TestCase(
            level="Intermediate",
            category="Frontend Frameworks",
            text="I am proficent in React.js and have experence with Angular framwork. I use Webpack for bundeling and Babel for transpilation.",
            expected_typos=["proficent", "experence", "framwork", "bundeling"],
            expected_corrections=["proficient", "experience", "framework", "bundling"],
            description="Frontend frameworks and build tools"
        ),
        
        TestCase(
            level="Intermediate",
            category="Backend Technologies",
            text="I have experence with Node.js and Express framwork. I can build RESTfull APIs and work with MongoDB databse.",
            expected_typos=["experence", "framwork", "RESTfull", "databse"],
            expected_corrections=["experience", "framework", "RESTful", "database"],
            description="Backend development stack"
        ),
        
        TestCase(
            level="Intermediate",
            category="Version Control",
            text="I am familar with Git version controll and use GitHub for colaboration. I understand branching and mergeing strategies.",
            expected_typos=["familar", "controll", "colaboration", "mergeing"],
            expected_corrections=["familiar", "control", "collaboration", "merging"],
            description="Version control systems"
        ),
        
        TestCase(
            level="Intermediate",
            category="Cloud Basics",
            text="I have experence with AWS servises like EC2 and S3. I understand cloud computeing concepts and deployement strategies.",
            expected_typos=["experence", "servises", "computeing", "deployement"],
            expected_corrections=["experience", "services", "computing", "deployment"],
            description="Basic cloud computing"
        ),
        
        # ADVANCED LEVEL - Complex Technologies
        TestCase(
            level="Advanced",
            category="Microservices Architecture",
            text="I have experence designing microservises architectures using Docker containerisation and Kubernetes orchestraton. I implement service meshes with Istio.",
            expected_typos=["experence", "microservises", "containerisation", "orchestraton"],
            expected_corrections=["experience", "microservices", "containerization", "orchestration"],
            description="Advanced microservices and containerization"
        ),
        
        TestCase(
            level="Advanced",
            category="DevOps and CI/CD",
            text="I implement CI/CD piplines using Jenkins and GitLab CI. I use Terraform for infrastrucure as code and Ansible for configuraton management.",
            expected_typos=["piplines", "infrastrucure", "configuraton"],
            expected_corrections=["pipelines", "infrastructure", "configuration"],
            description="DevOps tools and practices"
        ),
        
        TestCase(
            level="Advanced",
            category="Machine Learning",
            text="I have experence with TensorFlow and PyTorch for deep lerning. I implement neural netwoks and use Scikit-learn for machne learning algoritms.",
            expected_typos=["experence", "lerning", "netwoks", "machne", "algoritms"],
            expected_corrections=["experience", "learning", "networks", "machine", "algorithms"],
            description="Machine learning and AI technologies"
        ),
        
        TestCase(
            level="Advanced",
            category="Distributed Systems",
            text="I design scalabel distributed systems using Apache Kafka for mesaging and Redis for cacheing. I implement event-driven architecures with eventual consistancy.",
            expected_typos=["scalabel", "mesaging", "cacheing", "architecures", "consistancy"],
            expected_corrections=["scalable", "messaging", "caching", "architectures", "consistency"],
            description="Distributed systems and event-driven architecture"
        ),
        
        TestCase(
            level="Advanced",
            category="Security and Cryptography",
            text="I implement OAuth2 authentification and JWT tokens for securty. I understand encription algorithms and have experence with penetraton testing.",
            expected_typos=["authentification", "securty", "encription", "experence", "penetraton"],
            expected_corrections=["authentication", "security", "encryption", "experience", "penetration"],
            description="Security and cryptography concepts"
        ),
        
        # EXPERT LEVEL - Cutting-edge Technologies
        TestCase(
            level="Expert",
            category="Blockchain and Web3",
            text="I develop smart contracs on Ethereum using Solidity. I understand concensus algoritms and have experence with decentralised applications (DApps).",
            expected_typos=["contracs", "concensus", "algoritms", "experence", "decentralised"],
            expected_corrections=["contracts", "consensus", "algorithms", "experience", "decentralized"],
            description="Blockchain and decentralized technologies"
        ),
        
        TestCase(
            level="Expert",
            category="Quantum Computing",
            text="I have experence with quantum computeing using Qiskit and understand quantum algoritms. I work with qubits and quantum entanglement phenomina.",
            expected_typos=["experence", "computeing", "algoritms", "phenomina"],
            expected_corrections=["experience", "computing", "algorithms", "phenomena"],
            description="Quantum computing concepts"
        ),
        
        TestCase(
            level="Expert",
            category="Advanced AI/ML",
            text="I implement transformers architecures and work with large languag models (LLMs). I have experence with reinforcment learning and generativ adversarial networks.",
            expected_typos=["architecures", "languag", "experence", "reinforcment", "generativ"],
            expected_corrections=["architectures", "language", "experience", "reinforcement", "generative"],
            description="Advanced AI and machine learning"
        ),
        
        # MIXED COMPLEXITY - Real Resume Scenarios
        TestCase(
            level="Mixed",
            category="Full Stack Resume",
            text="Senior Full Stack Developer with 5+ years experence in React, Node.js, and cloud technolgies. Proficent in microservises architecture and DevOps practises. Led team in implementng CI/CD piplines and containerisation strategies.",
            expected_typos=["experence", "technolgies", "proficent", "microservises", "practises", "implementng", "piplines", "containerisation"],
            expected_corrections=["experience", "technologies", "proficient", "microservices", "practices", "implementing", "pipelines", "containerization"],
            description="Realistic senior developer resume text"
        ),
        
        TestCase(
            level="Mixed",
            category="Data Science Resume",
            text="Data Scientist with experence in machne learning, deep lerning, and big data analytics. Skilled in Python, TensorFlow, and Apache Spark. Developed predicitv models and implemented real-time data procesing piplines.",
            expected_typos=["experence", "machne", "lerning", "predicitv", "procesing", "piplines"],
            expected_corrections=["experience", "machine", "learning", "predictive", "processing", "pipelines"],
            description="Data science professional resume"
        )
    ]

def test_it_accuracy():
    """Test IT technical terms accuracy across all levels"""
    
    print("🚀 IT Technical Terms Accuracy Test")
    print("=" * 60)
    
    try:
        # Import enhanced service
        from enhanced_text_analysis_service import EnhancedTextAnalysisService, EnhancedAnalysisConfig
        
        # Create enhanced service with optimized settings for technical terms
        config = EnhancedAnalysisConfig(
            enable_traditional_nlp=True,
            enable_gector=True,
            enable_domain_validation=True,
            confidence_threshold=70.0,  # Lower threshold to catch more technical terms
            max_processing_time=5.0,
            parallel_processing=True,
            cache_enabled=True
        )
        
        service = EnhancedTextAnalysisService(config)
        print("✅ Enhanced service initialized")
        
        # Get test cases
        test_cases = create_it_test_cases()
        
        # Results tracking
        results_by_level = {}
        overall_results = []
        
        print(f"\n📝 Testing {len(test_cases)} IT technical term scenarios...")
        print("=" * 60)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test {i}: {test_case.level} - {test_case.category} ---")
            print(f"Description: {test_case.description}")
            print(f"Text: {test_case.text}")
            
            start_time = time.time()
            
            # Analyze text
            result = service.analyze_text(
                text=test_case.text,
                check_spelling=True,
                check_grammar=True
            )
            
            processing_time = time.time() - start_time
            
            # Extract detected typos
            detected_typos = [typo.word.lower() for typo in result.typos]
            detected_corrections = [typo.suggestion.lower() for typo in result.typos]
            
            # Calculate accuracy metrics
            expected_typos_lower = [word.lower() for word in test_case.expected_typos]
            expected_corrections_lower = [word.lower() for word in test_case.expected_corrections]
            
            # Typo detection accuracy
            typo_precision, typo_recall, typo_f1 = calculate_metrics(detected_typos, expected_typos_lower)
            
            # Correction accuracy (how many corrections are correct)
            correction_accuracy = calculate_correction_accuracy(
                detected_typos, detected_corrections, 
                expected_typos_lower, expected_corrections_lower
            )
            
            # Store results
            test_result = {
                'level': test_case.level,
                'category': test_case.category,
                'typo_precision': typo_precision,
                'typo_recall': typo_recall,
                'typo_f1': typo_f1,
                'correction_accuracy': correction_accuracy,
                'processing_time': processing_time,
                'detected_count': len(detected_typos),
                'expected_count': len(expected_typos_lower),
                'layers_used': len(result.layers_used)
            }
            
            overall_results.append(test_result)
            
            # Group by level
            if test_case.level not in results_by_level:
                results_by_level[test_case.level] = []
            results_by_level[test_case.level].append(test_result)
            
            # Display results
            print(f"⏱️  Processing time: {processing_time:.3f}s")
            print(f"🔍 Layers used: {[layer.value for layer in result.layers_used]}")
            print(f"🎯 Expected typos: {len(expected_typos_lower)} | Detected: {len(detected_typos)}")
            print(f"📊 Typo Detection - Precision: {typo_precision:.1%} | Recall: {typo_recall:.1%} | F1: {typo_f1:.1%}")
            print(f"✅ Correction Accuracy: {correction_accuracy:.1%}")
            
            # Show detailed results
            if result.typos:
                print("🔤 Detected typos:")
                for typo in result.typos:
                    is_expected = typo.word.lower() in expected_typos_lower
                    status = "✅" if is_expected else "❌"
                    print(f"  {status} '{typo.word}' → '{typo.suggestion}' (confidence: {typo.confidence_score:.1f}%)")
            
            # Show missed typos
            missed_typos = set(expected_typos_lower) - set(detected_typos)
            if missed_typos:
                print(f"❌ Missed typos: {list(missed_typos)}")
        
        # Calculate overall statistics
        print(f"\n🎯 OVERALL RESULTS BY LEVEL")
        print("=" * 60)
        
        level_summaries = {}
        for level, level_results in results_by_level.items():
            avg_f1 = sum(r['typo_f1'] for r in level_results) / len(level_results)
            avg_correction = sum(r['correction_accuracy'] for r in level_results) / len(level_results)
            avg_time = sum(r['processing_time'] for r in level_results) / len(level_results)
            
            level_summaries[level] = {
                'avg_f1': avg_f1,
                'avg_correction': avg_correction,
                'avg_time': avg_time,
                'test_count': len(level_results)
            }
            
            print(f"\n{level} Level ({len(level_results)} tests):")
            print(f"  📈 Average F1 Score: {avg_f1:.1%}")
            print(f"  ✅ Average Correction Accuracy: {avg_correction:.1%}")
            print(f"  ⏱️  Average Processing Time: {avg_time:.3f}s")
        
        # Overall system performance
        overall_f1 = sum(r['typo_f1'] for r in overall_results) / len(overall_results)
        overall_correction = sum(r['correction_accuracy'] for r in overall_results) / len(overall_results)
        overall_time = sum(r['processing_time'] for r in overall_results) / len(overall_results)
        
        print(f"\n🏆 SYSTEM-WIDE PERFORMANCE")
        print("=" * 60)
        print(f"📊 Overall F1 Score: {overall_f1:.1%}")
        print(f"✅ Overall Correction Accuracy: {overall_correction:.1%}")
        print(f"⏱️  Average Processing Time: {overall_time:.3f}s")
        print(f"🧪 Total Tests: {len(overall_results)}")
        
        # Performance by complexity
        print(f"\n📈 PERFORMANCE BY COMPLEXITY")
        print("=" * 60)
        
        complexity_order = ["Beginner", "Intermediate", "Advanced", "Expert", "Mixed"]
        for level in complexity_order:
            if level in level_summaries:
                summary = level_summaries[level]
                print(f"{level:12} | F1: {summary['avg_f1']:6.1%} | Correction: {summary['avg_correction']:6.1%} | Time: {summary['avg_time']:6.3f}s")
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT")
        print("=" * 60)
        
        target_f1 = 0.85
        target_correction = 0.80
        target_time = 3.0
        
        f1_achieved = overall_f1 >= target_f1
        correction_achieved = overall_correction >= target_correction
        time_achieved = overall_time <= target_time
        
        print(f"F1 Score Target (≥85%): {'✅ ACHIEVED' if f1_achieved else '❌ NOT ACHIEVED'} ({overall_f1:.1%})")
        print(f"Correction Target (≥80%): {'✅ ACHIEVED' if correction_achieved else '❌ NOT ACHIEVED'} ({overall_correction:.1%})")
        print(f"Speed Target (≤3s): {'✅ ACHIEVED' if time_achieved else '❌ NOT ACHIEVED'} ({overall_time:.3f}s)")
        
        if f1_achieved and correction_achieved and time_achieved:
            print("\n🎉 EXCELLENT: All targets achieved! System ready for production.")
        elif (overall_f1 >= 0.70 and overall_correction >= 0.70):
            print("\n✅ GOOD: System performs well, minor optimizations possible.")
        else:
            print("\n⚠️  NEEDS IMPROVEMENT: Consider tuning confidence thresholds or adding more technical terms.")
        
        # Cleanup
        service.cleanup()
        
        return {
            'overall_f1': overall_f1,
            'overall_correction': overall_correction,
            'overall_time': overall_time,
            'level_summaries': level_summaries,
            'success': f1_achieved and correction_achieved and time_achieved
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def calculate_metrics(detected: List[str], expected: List[str]) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score"""
    if not expected and not detected:
        return 1.0, 1.0, 1.0
    
    if not expected:
        return 0.0 if detected else 1.0, 1.0, 0.0 if detected else 1.0
    
    if not detected:
        return 1.0, 0.0, 0.0
    
    detected_set = set(detected)
    expected_set = set(expected)
    
    true_positives = len(detected_set & expected_set)
    false_positives = len(detected_set - expected_set)
    false_negatives = len(expected_set - detected_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def calculate_correction_accuracy(detected_typos: List[str], detected_corrections: List[str],
                                expected_typos: List[str], expected_corrections: List[str]) -> float:
    """Calculate how accurate the corrections are"""
    if not detected_typos:
        return 1.0 if not expected_typos else 0.0
    
    correct_corrections = 0
    total_corrections = len(detected_typos)
    
    # Create mapping of expected typo -> correction
    expected_mapping = {}
    for i in range(min(len(expected_typos), len(expected_corrections))):
        expected_mapping[expected_typos[i]] = expected_corrections[i]
    
    # Check each detected correction
    for i in range(len(detected_typos)):
        detected_typo = detected_typos[i]
        detected_correction = detected_corrections[i] if i < len(detected_corrections) else ""
        
        # If this typo was expected and correction matches
        if detected_typo in expected_mapping:
            expected_correction = expected_mapping[detected_typo]
            if detected_correction.lower() == expected_correction.lower():
                correct_corrections += 1
    
    return correct_corrections / total_corrections if total_corrections > 0 else 0.0

if __name__ == "__main__":
    result = test_it_accuracy()
    
    print(f"\n{'='*60}")
    if result.get('success'):
        print("🎉 IT TECHNICAL TERMS TEST COMPLETED SUCCESSFULLY!")
    else:
        print("⚠️  IT TECHNICAL TERMS TEST COMPLETED WITH MIXED RESULTS")
    print(f"{'='*60}")