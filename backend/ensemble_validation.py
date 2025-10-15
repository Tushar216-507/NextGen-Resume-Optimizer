"""
Ensemble validation system for enhanced typo detection.
Implements sophisticated voting algorithms and conflict resolution strategies.
"""

import logging
import statistics
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
from enum import Enum

from enhanced_models import (
    LayeredDetectionResult, EnsembleResult, DetectionLayer, 
    AnalysisConfig
)
from models import ValidationStatus
from core_interfaces import IEnsembleValidator
from confidence_scoring import AdvancedConfidenceScorer

logger = logging.getLogger(__name__)

class VotingStrategy(str, Enum):
    """Voting strategies for ensemble validation"""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    CONFIDENCE_BASED = "confidence_based"
    ADAPTIVE = "adaptive"
    CONSENSUS = "consensus"

class ConflictResolutionMethod(str, Enum):
    """Methods for resolving conflicts between layers"""
    HIGHEST_CONFIDENCE = "highest_confidence"
    MOST_RELIABLE_LAYER = "most_reliable_layer"
    WEIGHTED_AVERAGE = "weighted_average"
    DOMAIN_PRIORITY = "domain_priority"
    CONSERVATIVE = "conservative"

@dataclass
class LayerVote:
    """Represents a vote from a detection layer"""
    layer: DetectionLayer
    suggestion: str
    confidence: float
    weight: float
    reliability: float
    
class VotingResult:
    """Result of ensemble voting process"""
    def __init__(self):
        self.winning_suggestion: str = ""
        self.final_confidence: float = 0.0
        self.vote_distribution: Dict[str, float] = {}
        self.layer_votes: Dict[DetectionLayer, LayerVote] = {}
        self.conflict_detected: bool = False
        self.resolution_method: str = ""
        self.explanation: str = ""

class LayerReliabilityManager:
    """Manages reliability scores and weights for different layers"""
    
    def __init__(self):
        # Base reliability scores (can be updated based on performance)
        self.base_reliability = {
            DetectionLayer.TRADITIONAL_NLP: 0.75,
            DetectionLayer.GECTOR_TRANSFORMER: 0.85,
            DetectionLayer.DOMAIN_VALIDATION: 0.90,
            DetectionLayer.ENSEMBLE: 1.0
        }
        
        # Dynamic reliability tracking
        self.performance_history: Dict[DetectionLayer, List[float]] = defaultdict(list)
        self.current_reliability: Dict[DetectionLayer, float] = self.base_reliability.copy()
        
        # Layer weights for different contexts
        self.context_weights = {
            'technical': {
                DetectionLayer.DOMAIN_VALIDATION: 1.2,
                DetectionLayer.TRADITIONAL_NLP: 0.8,
                DetectionLayer.GECTOR_TRANSFORMER: 1.0
            },
            'general': {
                DetectionLayer.DOMAIN_VALIDATION: 0.9,
                DetectionLayer.TRADITIONAL_NLP: 1.0,
                DetectionLayer.GECTOR_TRANSFORMER: 1.1
            },
            'formal': {
                DetectionLayer.DOMAIN_VALIDATION: 0.8,
                DetectionLayer.TRADITIONAL_NLP: 1.1,
                DetectionLayer.GECTOR_TRANSFORMER: 1.2
            }
        }
    
    def get_layer_weight(self, layer: DetectionLayer, context: str = 'general') -> float:
        """Get weight for a layer in specific context"""
        base_weight = self.context_weights.get(context, {}).get(layer, 1.0)
        reliability_factor = self.current_reliability.get(layer, 0.8)
        return base_weight * reliability_factor
    
    def update_layer_performance(self, layer: DetectionLayer, accuracy: float) -> None:
        """Update layer performance history"""
        self.performance_history[layer].append(accuracy)
        
        # Keep only recent history (last 100 evaluations)
        if len(self.performance_history[layer]) > 100:
            self.performance_history[layer] = self.performance_history[layer][-100:]
        
        # Update current reliability with exponential moving average
        if len(self.performance_history[layer]) >= 5:
            recent_performance = statistics.mean(self.performance_history[layer][-10:])
            current = self.current_reliability.get(layer, self.base_reliability[layer])
            self.current_reliability[layer] = 0.8 * current + 0.2 * recent_performance
    
    def get_reliability_score(self, layer: DetectionLayer) -> float:
        """Get current reliability score for a layer"""
        return self.current_reliability.get(layer, 0.5)

class ConflictDetector:
    """Detects and analyzes conflicts between layer suggestions"""
    
    def __init__(self):
        self.similarity_threshold = 0.7
        self.confidence_gap_threshold = 20.0  # Percentage points
    
    def detect_conflicts(self, layer_results: List[LayeredDetectionResult]) -> Dict[str, Any]:
        """
        Detect conflicts between layer suggestions.
        
        Returns:
            Dictionary with conflict analysis
        """
        if len(layer_results) < 2:
            return {'has_conflict': False, 'conflict_type': None}
        
        # Group suggestions by similarity
        suggestion_groups = self._group_similar_suggestions(layer_results)
        
        # Analyze conflict patterns
        conflict_analysis = {
            'has_conflict': len(suggestion_groups) > 1,
            'conflict_type': self._classify_conflict_type(suggestion_groups),
            'suggestion_groups': suggestion_groups,
            'confidence_variance': self._calculate_confidence_variance(layer_results),
            'layer_agreement': self._calculate_layer_agreement(layer_results)
        }
        
        return conflict_analysis
    
    def _group_similar_suggestions(self, results: List[LayeredDetectionResult]) -> List[List[LayeredDetectionResult]]:
        """Group results by suggestion similarity"""
        groups = []
        
        for result in results:
            if not result.suggestions:
                continue
            
            primary_suggestion = result.suggestions[0]
            placed = False
            
            # Try to place in existing group
            for group in groups:
                if group and self._are_suggestions_similar(primary_suggestion, group[0].suggestions[0]):
                    group.append(result)
                    placed = True
                    break
            
            # Create new group if not placed
            if not placed:
                groups.append([result])
        
        return groups
    
    def _are_suggestions_similar(self, suggestion1: str, suggestion2: str) -> bool:
        """Check if two suggestions are similar"""
        if suggestion1.lower() == suggestion2.lower():
            return True
        
        # Calculate similarity using edit distance
        distance = self._levenshtein_distance(suggestion1.lower(), suggestion2.lower())
        max_len = max(len(suggestion1), len(suggestion2))
        
        if max_len == 0:
            return True
        
        similarity = 1 - (distance / max_len)
        return similarity >= self.similarity_threshold
    
    def _classify_conflict_type(self, suggestion_groups: List[List[LayeredDetectionResult]]) -> Optional[str]:
        """Classify the type of conflict"""
        if len(suggestion_groups) <= 1:
            return None
        
        # Analyze confidence levels
        group_confidences = []
        for group in suggestion_groups:
            avg_confidence = statistics.mean([
                max(result.confidence_scores) if result.confidence_scores else 0
                for result in group
            ])
            group_confidences.append(avg_confidence)
        
        confidence_gap = max(group_confidences) - min(group_confidences)
        
        if confidence_gap > self.confidence_gap_threshold:
            return "high_confidence_disagreement"
        elif len(suggestion_groups) == 2:
            return "binary_disagreement"
        else:
            return "multiple_disagreement"
    
    def _calculate_confidence_variance(self, results: List[LayeredDetectionResult]) -> float:
        """Calculate variance in confidence scores"""
        confidences = []
        for result in results:
            if result.confidence_scores:
                confidences.append(max(result.confidence_scores))
        
        if len(confidences) < 2:
            return 0.0
        
        return statistics.variance(confidences)
    
    def _calculate_layer_agreement(self, results: List[LayeredDetectionResult]) -> float:
        """Calculate agreement level between layers (0-1)"""
        if len(results) < 2:
            return 1.0
        
        suggestions = []
        for result in results:
            if result.suggestions:
                suggestions.append(result.suggestions[0].lower())
        
        if not suggestions:
            return 0.0
        
        # Calculate pairwise agreement
        agreements = 0
        total_pairs = 0
        
        for i in range(len(suggestions)):
            for j in range(i + 1, len(suggestions)):
                total_pairs += 1
                if self._are_suggestions_similar(suggestions[i], suggestions[j]):
                    agreements += 1
        
        return agreements / total_pairs if total_pairs > 0 else 0.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

class VotingAlgorithm:
    """Implements various voting algorithms for ensemble validation"""
    
    def __init__(self, reliability_manager: LayerReliabilityManager):
        self.reliability_manager = reliability_manager
    
    def majority_vote(self, votes: List[LayerVote]) -> VotingResult:
        """Simple majority voting"""
        result = VotingResult()
        
        # Count votes for each suggestion
        vote_counts = Counter(vote.suggestion for vote in votes)
        
        if vote_counts:
            result.winning_suggestion = vote_counts.most_common(1)[0][0]
            result.final_confidence = statistics.mean([
                vote.confidence for vote in votes 
                if vote.suggestion == result.winning_suggestion
            ])
            
            # Calculate vote distribution
            total_votes = len(votes)
            result.vote_distribution = {
                suggestion: count / total_votes 
                for suggestion, count in vote_counts.items()
            }
        
        result.explanation = f"Majority vote winner: {result.winning_suggestion}"
        return result
    
    def weighted_vote(self, votes: List[LayerVote], context: str = 'general') -> VotingResult:
        """Weighted voting based on layer reliability and context"""
        result = VotingResult()
        
        # Calculate weighted scores for each suggestion
        suggestion_scores = defaultdict(float)
        suggestion_weights = defaultdict(float)
        
        for vote in votes:
            weight = self.reliability_manager.get_layer_weight(vote.layer, context)
            weighted_score = vote.confidence * weight
            
            suggestion_scores[vote.suggestion] += weighted_score
            suggestion_weights[vote.suggestion] += weight
        
        if suggestion_scores:
            # Find winning suggestion
            result.winning_suggestion = max(suggestion_scores.keys(), key=suggestion_scores.get)
            
            # Calculate final confidence as weighted average
            total_weight = suggestion_weights[result.winning_suggestion]
            result.final_confidence = suggestion_scores[result.winning_suggestion] / max(total_weight, 1.0)
            
            # Calculate vote distribution
            total_score = sum(suggestion_scores.values())
            result.vote_distribution = {
                suggestion: score / total_score 
                for suggestion, score in suggestion_scores.items()
            }
        
        result.explanation = f"Weighted vote winner: {result.winning_suggestion} (weight: {total_weight:.2f})"
        return result
    
    def confidence_based_vote(self, votes: List[LayerVote]) -> VotingResult:
        """Voting based purely on confidence scores"""
        result = VotingResult()
        
        if votes:
            # Find vote with highest confidence
            best_vote = max(votes, key=lambda v: v.confidence)
            result.winning_suggestion = best_vote.suggestion
            result.final_confidence = best_vote.confidence
            
            # Calculate distribution based on confidence
            total_confidence = sum(vote.confidence for vote in votes)
            if total_confidence > 0:
                confidence_by_suggestion = defaultdict(float)
                for vote in votes:
                    confidence_by_suggestion[vote.suggestion] += vote.confidence
                
                result.vote_distribution = {
                    suggestion: conf / total_confidence 
                    for suggestion, conf in confidence_by_suggestion.items()
                }
        
        result.explanation = f"Highest confidence winner: {result.winning_suggestion} ({result.final_confidence:.1f}%)"
        return result
    
    def adaptive_vote(self, votes: List[LayerVote], conflict_analysis: Dict[str, Any], 
                     context: str = 'general') -> VotingResult:
        """Adaptive voting that chooses strategy based on conflict analysis"""
        
        # Choose strategy based on conflict characteristics
        if not conflict_analysis['has_conflict']:
            # No conflict - use simple average
            return self._consensus_vote(votes)
        
        conflict_type = conflict_analysis['conflict_type']
        confidence_variance = conflict_analysis['confidence_variance']
        layer_agreement = conflict_analysis['layer_agreement']
        
        if confidence_variance > 400:  # High variance in confidence
            return self.confidence_based_vote(votes)
        elif layer_agreement < 0.3:  # Low agreement
            return self.weighted_vote(votes, context)
        elif conflict_type == "binary_disagreement":
            return self._resolve_binary_conflict(votes, context)
        else:
            return self.weighted_vote(votes, context)
    
    def _consensus_vote(self, votes: List[LayerVote]) -> VotingResult:
        """Consensus voting for cases with no conflict"""
        result = VotingResult()
        
        if votes:
            # Use the most common suggestion
            suggestions = [vote.suggestion for vote in votes]
            result.winning_suggestion = Counter(suggestions).most_common(1)[0][0]
            
            # Average confidence of matching votes
            matching_votes = [vote for vote in votes if vote.suggestion == result.winning_suggestion]
            result.final_confidence = statistics.mean([vote.confidence for vote in matching_votes])
            
            result.explanation = f"Consensus winner: {result.winning_suggestion}"
        
        return result
    
    def _resolve_binary_conflict(self, votes: List[LayerVote], context: str) -> VotingResult:
        """Resolve binary conflicts using domain knowledge"""
        result = VotingResult()
        
        # Group votes by suggestion
        suggestion_groups = defaultdict(list)
        for vote in votes:
            suggestion_groups[vote.suggestion].append(vote)
        
        if len(suggestion_groups) == 2:
            suggestions = list(suggestion_groups.keys())
            
            # Compare groups
            group1_votes = suggestion_groups[suggestions[0]]
            group2_votes = suggestion_groups[suggestions[1]]
            
            # Calculate weighted scores
            score1 = sum(vote.confidence * self.reliability_manager.get_layer_weight(vote.layer, context) 
                        for vote in group1_votes)
            score2 = sum(vote.confidence * self.reliability_manager.get_layer_weight(vote.layer, context) 
                        for vote in group2_votes)
            
            if score1 > score2:
                result.winning_suggestion = suggestions[0]
                result.final_confidence = statistics.mean([vote.confidence for vote in group1_votes])
            else:
                result.winning_suggestion = suggestions[1]
                result.final_confidence = statistics.mean([vote.confidence for vote in group2_votes])
            
            result.explanation = f"Binary conflict resolved: {result.winning_suggestion} (score: {max(score1, score2):.1f})"
        
        return result

class AdvancedEnsembleValidator(IEnsembleValidator):
    """
    Advanced ensemble validator with sophisticated voting and conflict resolution.
    """
    
    def __init__(self, confidence_scorer: Optional[AdvancedConfidenceScorer] = None):
        self.confidence_scorer = confidence_scorer or AdvancedConfidenceScorer()
        self.reliability_manager = LayerReliabilityManager()
        self.conflict_detector = ConflictDetector()
        self.voting_algorithm = VotingAlgorithm(self.reliability_manager)
        
        # Configuration
        self.default_voting_strategy = VotingStrategy.ADAPTIVE
        self.min_confidence_threshold = 60.0
        self.consensus_threshold = 0.8
        
        logger.info("Advanced ensemble validator initialized")
    
    def validate_ensemble(self, layer_results: List[LayeredDetectionResult],
                         config: AnalysisConfig) -> EnsembleResult:
        """
        Validate and combine results from multiple layers.
        
        Args:
            layer_results: Results from different detection layers
            config: Analysis configuration
            
        Returns:
            EnsembleResult with final decision
        """
        if not layer_results:
            return self._create_empty_result("No layer results provided")
        
        # Filter results by confidence threshold
        filtered_results = [
            result for result in layer_results 
            if result.confidence_scores and max(result.confidence_scores) >= self.min_confidence_threshold
        ]
        
        if not filtered_results:
            return self._create_empty_result("No results meet confidence threshold")
        
        # Convert to votes
        votes = self._convert_to_votes(filtered_results)
        
        if not votes:
            return self._create_empty_result("No valid votes generated")
        
        # Detect conflicts
        conflict_analysis = self.conflict_detector.detect_conflicts(filtered_results)
        
        # Determine context
        context = self._analyze_context(layer_results)
        
        # Perform voting
        voting_result = self._perform_voting(votes, conflict_analysis, context, config)
        
        # Create ensemble result
        ensemble_result = self._create_ensemble_result(voting_result, conflict_analysis, layer_results)
        
        logger.debug(
            f"Ensemble validation completed: {ensemble_result.final_suggestion}",
            extra={
                'final_suggestion': ensemble_result.final_suggestion,
                'ensemble_confidence': ensemble_result.ensemble_confidence,
                'conflict_detected': conflict_analysis['has_conflict'],
                'layers_count': len(layer_results)
            }
        )
        
        return ensemble_result
    
    def _convert_to_votes(self, results: List[LayeredDetectionResult]) -> List[LayerVote]:
        """Convert layer results to votes"""
        votes = []
        
        for result in results:
            if not result.suggestions or not result.confidence_scores:
                continue
            
            # Use the best suggestion from each layer
            best_suggestion = result.suggestions[0]
            best_confidence = result.confidence_scores[0] if result.confidence_scores else 0
            
            # Get layer reliability
            reliability = self.reliability_manager.get_reliability_score(result.layer_name)
            
            vote = LayerVote(
                layer=result.layer_name,
                suggestion=best_suggestion,
                confidence=best_confidence,
                weight=1.0,  # Will be adjusted by voting algorithm
                reliability=reliability
            )
            
            votes.append(vote)
        
        return votes
    
    def _analyze_context(self, results: List[LayeredDetectionResult]) -> str:
        """Analyze context to determine voting strategy"""
        # Simple context analysis based on detection methods and content
        technical_indicators = 0
        formal_indicators = 0
        
        for result in results:
            context_lower = result.context.lower() if result.context else ""
            
            # Check for technical context
            if any(indicator in context_lower for indicator in [
                'programming', 'software', 'api', 'database', 'framework'
            ]):
                technical_indicators += 1
            
            # Check for formal context
            if any(indicator in context_lower for indicator in [
                'resume', 'professional', 'experience', 'education'
            ]):
                formal_indicators += 1
        
        if technical_indicators > formal_indicators:
            return 'technical'
        elif formal_indicators > technical_indicators:
            return 'formal'
        else:
            return 'general'
    
    def _perform_voting(self, votes: List[LayerVote], conflict_analysis: Dict[str, Any],
                       context: str, config: AnalysisConfig) -> VotingResult:
        """Perform voting using the appropriate strategy"""
        
        # Choose voting strategy
        strategy = self.default_voting_strategy
        
        # Override strategy based on configuration or conflict analysis
        if hasattr(config, 'voting_strategy'):
            strategy = config.voting_strategy
        
        # Execute voting
        if strategy == VotingStrategy.MAJORITY:
            return self.voting_algorithm.majority_vote(votes)
        elif strategy == VotingStrategy.WEIGHTED:
            return self.voting_algorithm.weighted_vote(votes, context)
        elif strategy == VotingStrategy.CONFIDENCE_BASED:
            return self.voting_algorithm.confidence_based_vote(votes)
        elif strategy == VotingStrategy.ADAPTIVE:
            return self.voting_algorithm.adaptive_vote(votes, conflict_analysis, context)
        else:
            # Default to weighted voting
            return self.voting_algorithm.weighted_vote(votes, context)
    
    def _create_ensemble_result(self, voting_result: VotingResult, 
                              conflict_analysis: Dict[str, Any],
                              original_results: List[LayeredDetectionResult]) -> EnsembleResult:
        """Create final ensemble result"""
        
        # Determine validation status
        validation_status = ValidationStatus.VALIDATED
        if conflict_analysis['has_conflict']:
            if conflict_analysis['layer_agreement'] < 0.5:
                validation_status = ValidationStatus.CROSS_VALIDATED
            else:
                validation_status = ValidationStatus.VALIDATED
        
        # Create layer votes dictionary
        layer_votes = {}
        for layer_result in original_results:
            if layer_result.confidence_scores:
                layer_votes[layer_result.layer_name] = max(layer_result.confidence_scores)
        
        # Generate explanation
        explanation = self._generate_explanation(voting_result, conflict_analysis, original_results)
        
        # Determine conflict resolution method
        resolution_method = voting_result.resolution_method or "adaptive_voting"
        
        return EnsembleResult(
            final_suggestion=voting_result.winning_suggestion,
            ensemble_confidence=voting_result.final_confidence,
            layer_votes=layer_votes,
            explanation=explanation,
            validation_status=validation_status.value,
            conflict_resolution_method=resolution_method
        )
    
    def _generate_explanation(self, voting_result: VotingResult, 
                            conflict_analysis: Dict[str, Any],
                            original_results: List[LayeredDetectionResult]) -> str:
        """Generate human-readable explanation"""
        explanation_parts = []
        
        # Basic result
        explanation_parts.append(f"Ensemble selected '{voting_result.winning_suggestion}' with {voting_result.final_confidence:.1f}% confidence")
        
        # Layer participation
        layer_names = [result.layer_name.value for result in original_results]
        explanation_parts.append(f"Based on input from {len(layer_names)} layers: {', '.join(layer_names)}")
        
        # Conflict information
        if conflict_analysis['has_conflict']:
            conflict_type = conflict_analysis['conflict_type']
            agreement = conflict_analysis['layer_agreement']
            explanation_parts.append(f"Resolved {conflict_type} with {agreement:.1%} layer agreement")
        else:
            explanation_parts.append("All layers in agreement")
        
        # Voting details
        if voting_result.explanation:
            explanation_parts.append(voting_result.explanation)
        
        return ". ".join(explanation_parts)
    
    def _create_empty_result(self, reason: str) -> EnsembleResult:
        """Create empty ensemble result with explanation"""
        return EnsembleResult(
            final_suggestion="",
            ensemble_confidence=0.0,
            layer_votes={},
            explanation=f"No ensemble result: {reason}",
            validation_status=ValidationStatus.FILTERED.value,
            conflict_resolution_method="none"
        )
    
    def resolve_conflicts(self, conflicting_results: List[LayeredDetectionResult]) -> str:
        """Resolve conflicts between different layer suggestions"""
        if not conflicting_results:
            return ""
        
        # Use confidence-based resolution as fallback
        votes = self._convert_to_votes(conflicting_results)
        voting_result = self.voting_algorithm.confidence_based_vote(votes)
        
        return voting_result.winning_suggestion
    
    def calculate_ensemble_confidence(self, layer_results: List[LayeredDetectionResult],
                                    weights: Dict[DetectionLayer, float]) -> float:
        """Calculate ensemble confidence from layer results"""
        if not layer_results:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for result in layer_results:
            if result.confidence_scores:
                confidence = max(result.confidence_scores)
                weight = weights.get(result.layer_name, 1.0)
                
                weighted_sum += confidence * weight
                total_weight += weight
        
        return weighted_sum / max(total_weight, 1.0)
    
    def update_layer_performance(self, layer: DetectionLayer, accuracy: float) -> None:
        """Update layer performance for future ensemble decisions"""
        self.reliability_manager.update_layer_performance(layer, accuracy)
        logger.debug(f"Updated {layer} performance: {accuracy:.2f}")
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble validation statistics"""
        return {
            'layer_reliability': {
                layer.value: self.reliability_manager.get_reliability_score(layer)
                for layer in DetectionLayer
            },
            'voting_strategy': self.default_voting_strategy.value,
            'confidence_threshold': self.min_confidence_threshold,
            'consensus_threshold': self.consensus_threshold
        }