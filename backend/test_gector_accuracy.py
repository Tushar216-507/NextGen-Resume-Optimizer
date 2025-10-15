"""
Test script to compare accuracy between GECToR and traditional NLP approaches.

This script evaluates and compares the accuracy of GECToR-based analysis
against traditional NLP methods for spelling and grammar correction.
"""
import pytest
import time
import pandas as pd
from typing import List, Dict, Any

from text_analysis_service import TextAnalysisService
from gector_analysis_service import GECToRAnalysisService
from models import ValidatedTypoResult, ValidatedGrammarResult, ValidationStatus

# Test data with known errors
TEST_CASES = [
    {
        "text": "I have five years of experence in software developement and machine learnin.",
        "expected_typos": ["experence", "developement", "learnin"],
        "expected_grammar": []
    },
    {
        "text": "Me and my team has completed the project ahead of schedule.",
        "expected_typos": [],
        "expected_grammar": ["Me and my team has"]
    },
    {
        "text": "Proficient in Python, Javascript, and React framwork.",
        "expected_typos": ["framwork"],
        "expected_grammar": []
    },
    {
        "text": "Led a team of 5 developers to delivered the product on time.",
        "expected_typos": [],
        "expected_grammar": ["to delivered"]
    },
    {
        "text": "Implemented a algoritm for optimizing database querys.",
        "expected_typos": ["algoritm", "querys"],
        "expected_grammar": ["a algoritm"]
    }
]

def calculate_metrics(detected: List[str], expected: List[str]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        detected: List of detected errors
        expected: List of expected errors
        
    Returns:
        Dictionary with precision, recall, and F1 metrics
    """
    true_positives = len(set(detected) & set(expected))
    false_positives = len(set(detected) - set(expected))
    false_negatives = len(set(expected) - set(detected))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def extract_typos_from_results(results: List[ValidatedTypoResult]) -> List[str]:
    """Extract original typo words from results"""
    return [result.word for result in results]

def extract_grammar_from_results(results: List[ValidatedGrammarResult]) -> List[str]:
    """Extract grammar issues from results"""
    return [result.sentence for result in results]

def test_accuracy_comparison():
    """Test to compare accuracy between GECToR and traditional NLP"""
    
    # Initialize services
    traditional_service = TextAnalysisService(enable_enhanced_analysis=True)
    gector_service = GECToRAnalysisService()
    
    # Results storage
    results = []
    
    # Process each test case
    for i, test_case in enumerate(TEST_CASES):
        text = test_case["text"]
        expected_typos = test_case["expected_typos"]
        expected_grammar = test_case["expected_grammar"]
        
        # Traditional NLP analysis
        traditional_start = time.time()
        traditional_analysis = traditional_service.analyze_with_confidence(text)
        traditional_time = time.time() - traditional_start
        
        traditional_typos = extract_typos_from_results(traditional_analysis.typos)
        traditional_grammar = extract_grammar_from_results(traditional_analysis.grammar_issues)
        
        # GECToR analysis
        gector_start = time.time()
        gector_typos, gector_grammar = gector_service.analyze_text(text)
        gector_time = time.time() - gector_start
        
        gector_typos_list = extract_typos_from_results(gector_typos)
        gector_grammar_list = extract_grammar_from_results(gector_grammar)
        
        # Calculate metrics
        traditional_typo_metrics = calculate_metrics(traditional_typos, expected_typos)
        traditional_grammar_metrics = calculate_metrics(traditional_grammar, expected_grammar)
        
        gector_typo_metrics = calculate_metrics(gector_typos_list, expected_typos)
        gector_grammar_metrics = calculate_metrics(gector_grammar_list, expected_grammar)
        
        # Store results
        results.append({
            "test_case": i + 1,
            "text": text,
            "traditional_typo_precision": traditional_typo_metrics["precision"],
            "traditional_typo_recall": traditional_typo_metrics["recall"],
            "traditional_typo_f1": traditional_typo_metrics["f1"],
            "traditional_grammar_precision": traditional_grammar_metrics["precision"],
            "traditional_grammar_recall": traditional_grammar_metrics["recall"],
            "traditional_grammar_f1": traditional_grammar_metrics["f1"],
            "traditional_time": traditional_time,
            "gector_typo_precision": gector_typo_metrics["precision"],
            "gector_typo_recall": gector_typo_metrics["recall"],
            "gector_typo_f1": gector_typo_metrics["f1"],
            "gector_grammar_precision": gector_grammar_metrics["precision"],
            "gector_grammar_recall": gector_grammar_metrics["recall"],
            "gector_grammar_f1": gector_grammar_metrics["f1"],
            "gector_time": gector_time
        })
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Calculate average metrics
    avg_traditional_typo_f1 = df["traditional_typo_f1"].mean()
    avg_traditional_grammar_f1 = df["traditional_grammar_f1"].mean()
    avg_traditional_time = df["traditional_time"].mean()
    
    avg_gector_typo_f1 = df["gector_typo_f1"].mean()
    avg_gector_grammar_f1 = df["gector_grammar_f1"].mean()
    avg_gector_time = df["gector_time"].mean()
    
    # Print summary
    print("\n=== ACCURACY COMPARISON RESULTS ===")
    print(f"Traditional NLP - Typo F1: {avg_traditional_typo_f1:.2f}, Grammar F1: {avg_traditional_grammar_f1:.2f}, Time: {avg_traditional_time:.3f}s")
    print(f"GECToR - Typo F1: {avg_gector_typo_f1:.2f}, Grammar F1: {avg_gector_grammar_f1:.2f}, Time: {avg_gector_time:.3f}s")
    print(f"Improvement - Typo: {(avg_gector_typo_f1 - avg_traditional_typo_f1) * 100:.1f}%, Grammar: {(avg_gector_grammar_f1 - avg_traditional_grammar_f1) * 100:.1f}%")
    print(f"Speed Difference: {(avg_gector_time / avg_traditional_time):.1f}x (>1 means GECToR is slower)")
    
    # Save detailed results to CSV
    df.to_csv("accuracy_comparison_results.csv", index=False)
    
    # Assert that GECToR performs better overall
    assert avg_gector_typo_f1 + avg_gector_grammar_f1 > avg_traditional_typo_f1 + avg_traditional_grammar_f1, \
        "GECToR should provide better overall accuracy than traditional NLP"

if __name__ == "__main__":
    test_accuracy_comparison()