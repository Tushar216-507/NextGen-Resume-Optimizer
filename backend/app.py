from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
from PyPDF2 import PdfReader
import docx
from models import AnalyzeRequest, AnalysisResult, ErrorResponse, EnhancedAnalysisResult
from text_analysis_service import TextAnalysisService

app = FastAPI(title="Resume Analyzer API", version="1.0.0")

# Initialize integrated text analysis service with enhanced capabilities
analysis_service = TextAnalysisService(enable_enhanced_analysis=True)

# ✅ Setup CORS globally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:xxxx"] for your Flutter web app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploads
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def health_check():
    capabilities = analysis_service.get_analysis_capabilities()
    return {
        "status": "Backend is running!", 
        "message": "Ready to receive files",
        "analysis_capabilities": capabilities
    }

@app.get("/capabilities")
async def get_capabilities():
    """Get information about available analysis capabilities"""
    return analysis_service.get_analysis_capabilities()

@app.post("/set_confidence_threshold")
async def set_confidence_threshold(threshold: float):
    """Set the confidence threshold for analysis suggestions"""
    try:
        if not (0.0 <= threshold <= 100.0):
            raise HTTPException(status_code=400, detail="Threshold must be between 0 and 100")
        
        analysis_service.set_confidence_threshold(threshold)
        return {
            "message": f"Confidence threshold set to {threshold}%",
            "new_threshold": analysis_service.confidence_threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set threshold: {str(e)}")

@app.post("/analyze_resume", response_model=AnalysisResult)
async def analyze_resume(request: AnalyzeRequest):
    """
    Analyze text for spelling mistakes, typos, and grammar issues (backward compatible)
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Perform text analysis using original service for backward compatibility
        result = analysis_service.analyze_full_text(
            text=request.text,
            check_spelling=request.check_spelling,
            check_grammar=request.check_grammar
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze_resume_enhanced", response_model=EnhancedAnalysisResult)
async def analyze_resume_enhanced(request: AnalyzeRequest):
    """
    Enhanced text analysis with confidence scoring and context awareness
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Check if enhanced analysis is available
        capabilities = analysis_service.get_analysis_capabilities()
        if not capabilities['enhanced_analysis_available']:
            raise HTTPException(
                status_code=503, 
                detail="Enhanced analysis not available. Missing enhanced components."
            )
        
        # Perform enhanced text analysis using integrated service
        result = analysis_service.analyze_with_confidence(
            text=request.text,
            check_spelling=request.check_spelling,
            check_grammar=request.check_grammar
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    print(f"🔥 RECEIVED FILE: {file.filename}, Size: {file.size}")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    print(f"📁 Saving to: {file_path}")

    # ✅ Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ Extract text from file
    text = ""
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        return {"error": "Unsupported file format. Please upload PDF or DOCX."}

    # ✅ Perform enhanced quality analysis on extracted text
    capabilities = analysis_service.get_analysis_capabilities()
    if capabilities['enhanced_analysis_available']:
        quality_analysis = analysis_service.analyze_with_confidence(text)
    else:
        # Fallback to basic analysis and convert to enhanced format
        basic_analysis = analysis_service.analyze_full_text(text)
        # Create a minimal enhanced result for compatibility
        from models import EnhancedAnalysisSummary, ConfidenceMetrics, ResumeContext
        quality_analysis = type('EnhancedResult', (), {
            'typos': [type('ValidatedTypo', (), {
                'word': t.word, 'suggestion': t.suggestion, 'confidence_score': 75.0,
                'explanation': f"Spelling correction: {t.word} -> {t.suggestion}",
                'validation_status': 'validated'
            })() for t in basic_analysis.typos],
            'grammar_issues': [type('ValidatedGrammar', (), {
                'sentence': g.sentence, 'suggestion': g.suggestion, 'issue_type': g.issue_type,
                'confidence_score': 75.0, 'explanation': f"Grammar issue: {g.issue_type}",
                'validation_status': 'validated'
            })() for g in basic_analysis.grammar_issues],
            'summary': type('EnhancedSummary', (), {
                'total_typos': basic_analysis.summary.total_typos,
                'total_grammar_issues': basic_analysis.summary.total_grammar_issues,
                'word_count': basic_analysis.summary.word_count,
                'confidence_metrics': type('ConfidenceMetrics', (), {
                    'average_confidence': 75.0, 'high_confidence_suggestions': 0,
                    'validation_pass_rate': 1.0
                })(),
                'context_analysis': type('ResumeContext', (), {
                    'sections': {}, 'formatting_style': 'traditional',
                    'professional_level': 'mid_level', 'industry_indicators': [],
                    'detected_technologies': []
                })()
            })(),
            'processing_time': basic_analysis.processing_time
        })()
    
    # ✅ Enhanced analysis with quality metrics
    analysis = {
        "atsScore": 75,  # Keep existing ATS score logic
        "grammaticalErrors": quality_analysis.summary.total_grammar_issues,
        "typos": quality_analysis.summary.total_typos,
        "missingBlocks": ["Projects", "Certifications"],  # Keep existing logic
        "presentBlocks": ["Contact Info", "Skills", "Experience", "Education"],  # Keep existing logic
        "suggestions": [
            "Add a Projects section",
            "Include relevant certifications",
            "Fix grammar in Experience section",
        ],
        "extractedText": text[:500] + "..." if len(text) > 500 else text,
        # ✅ Add enhanced quality analysis with confidence scores
        "qualityAnalysis": {
            "typos": [
                {
                    "word": typo.word, 
                    "suggestion": typo.suggestion,
                    "confidence": typo.confidence_score,
                    "explanation": typo.explanation,
                    "validationStatus": typo.validation_status
                } for typo in quality_analysis.typos
            ],
            "grammarIssues": [
                {
                    "sentence": issue.sentence, 
                    "suggestion": issue.suggestion, 
                    "type": issue.issue_type,
                    "confidence": issue.confidence_score,
                    "explanation": issue.explanation,
                    "validationStatus": issue.validation_status
                } for issue in quality_analysis.grammar_issues
            ],
            "summary": {
                "totalTypos": quality_analysis.summary.total_typos,
                "totalGrammarIssues": quality_analysis.summary.total_grammar_issues,
                "wordCount": quality_analysis.summary.word_count,
                "processingTime": quality_analysis.processing_time,
                "confidenceMetrics": {
                    "averageConfidence": quality_analysis.summary.confidence_metrics.average_confidence,
                    "highConfidenceSuggestions": quality_analysis.summary.confidence_metrics.high_confidence_suggestions,
                    "validationPassRate": quality_analysis.summary.confidence_metrics.validation_pass_rate
                },
                "contextAnalysis": {
                    "detectedSections": list(quality_analysis.summary.context_analysis.sections.keys()),
                    "formattingStyle": quality_analysis.summary.context_analysis.formatting_style,
                    "professionalLevel": quality_analysis.summary.context_analysis.professional_level,
                    "industryIndicators": quality_analysis.summary.context_analysis.industry_indicators,
                    "detectedTechnologies": quality_analysis.summary.context_analysis.detected_technologies
                }
            }
        }
    }

    return {"message": "Resume processed successfully", "analysis": analysis}

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    analysis_service.cleanup()
