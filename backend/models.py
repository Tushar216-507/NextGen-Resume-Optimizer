from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class TypoResult(BaseModel):
    """Model for spelling/typo detection results"""
    word: str
    suggestion: str
    position: Optional[int] = None

class GrammarResult(BaseModel):
    """Model for grammar issue detection results"""
    sentence: str
    suggestion: str
    issue_type: str
    position: Optional[int] = None

class AnalysisSummary(BaseModel):
    """Summary statistics for text analysis"""
    total_typos: int
    total_grammar_issues: int
    word_count: int
    readability_score: Optional[float] = None

class AnalysisResult(BaseModel):
    """Complete analysis result"""
    typos: List[TypoResult]
    grammar_issues: List[GrammarResult]
    summary: AnalysisSummary
    processing_time: float

class AnalyzeRequest(BaseModel):
    """Request model for analyze endpoint"""
    text: str
    check_spelling: bool = True
    check_grammar: bool = True
    language: str = "en"

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    error_code: str
    details: Optional[str] = None
    timestamp: datetime = datetime.now()

# Enhanced models for improved accuracy

class ValidationStatus(str, Enum):
    """Status of suggestion validation"""
    VALIDATED = "validated"
    CROSS_VALIDATED = "cross_validated"
    FILTERED = "filtered"
    PENDING = "pending"

class ProfessionalLevel(str, Enum):
    """Professional level indicators"""
    ENTRY_LEVEL = "entry_level"
    MID_LEVEL = "mid_level"
    SENIOR_LEVEL = "senior_level"
    EXECUTIVE = "executive"

class FormattingStyle(str, Enum):
    """Resume formatting styles"""
    TRADITIONAL = "traditional"
    MODERN = "modern"
    CREATIVE = "creative"
    TECHNICAL = "technical"

class ValidatedTypoResult(BaseModel):
    """Enhanced typo result with confidence and validation"""
    word: str
    suggestion: str
    confidence_score: float
    explanation: str
    context: str
    validation_status: ValidationStatus
    position: Optional[int] = None

class ValidatedGrammarResult(BaseModel):
    """Enhanced grammar result with confidence and validation"""
    sentence: str
    suggestion: str
    confidence_score: float
    explanation: str
    issue_type: str
    rule_category: str
    validation_status: ValidationStatus
    position: Optional[int] = None

class ConfidenceMetrics(BaseModel):
    """Metrics about confidence scoring"""
    average_confidence: float
    high_confidence_suggestions: int
    filtered_low_confidence: int
    validation_pass_rate: float

class TextSection(BaseModel):
    """Represents a section of resume text"""
    section_type: str
    content: str
    start_position: int
    end_position: int
    formatting_indicators: List[str]

class ResumeContext(BaseModel):
    """Context information about the resume"""
    sections: Dict[str, TextSection]
    formatting_style: FormattingStyle
    professional_level: ProfessionalLevel
    industry_indicators: List[str]
    detected_technologies: List[str]

class EnhancedAnalysisSummary(BaseModel):
    """Enhanced summary with confidence metrics"""
    total_typos: int
    total_grammar_issues: int
    word_count: int
    readability_score: Optional[float] = None
    confidence_metrics: ConfidenceMetrics
    context_analysis: ResumeContext

class EnhancedAnalysisResult(BaseModel):
    """Enhanced analysis result with confidence scoring"""
    typos: List[ValidatedTypoResult]
    grammar_issues: List[ValidatedGrammarResult]
    summary: EnhancedAnalysisSummary
    processing_time: float
    analysis_version: str = "enhanced_v1.0"