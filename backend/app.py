from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
from PyPDF2 import PdfReader
import docx
from models import AnalyzeRequest, AnalysisResult, ErrorResponse, EnhancedAnalysisResult
from text_analysis_service import TextAnalysisService
from enhanced_text_analysis_service import EnhancedTextAnalysisService, EnhancedAnalysisConfig

app = FastAPI(title="Resume Analyzer API", version="2.0.0")

# Initialize both services for backward compatibility and enhanced features
try:
    # Enhanced service with world-class accuracy
    enhanced_config = EnhancedAnalysisConfig(
        enable_traditional_nlp=True,
        enable_gector=True,
        enable_domain_validation=True,
        confidence_threshold=80.0,
        max_processing_time=3.0,
        parallel_processing=True,
        cache_enabled=True
    )
    enhanced_analysis_service = EnhancedTextAnalysisService(enhanced_config)
    ENHANCED_SERVICE_AVAILABLE = True
    print("✅ Enhanced analysis service initialized successfully")
except Exception as e:
    print(f"⚠️ Enhanced service initialization failed: {e}")
    ENHANCED_SERVICE_AVAILABLE = False

# Fallback to original service
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

def validate_job_type(job_type: str) -> str:
    """Validate and sanitize job type parameter"""
    valid_job_types = {
        'software_engineer', 'frontend_developer', 'backend_developer', 'full_stack_developer',
        'mobile_developer', 'data_scientist', 'data_engineer', 'machine_learning_engineer',
        'devops_engineer', 'cloud_engineer', 'security_engineer', 'qa_engineer',
        'ui_ux_designer', 'product_manager', 'engineering_manager', 'solutions_architect',
        'database_administrator', 'other'
    }
    
    if job_type and job_type in valid_job_types:
        return job_type
    else:
        print(f"⚠️ Invalid job type '{job_type}', defaulting to 'other'")
        return "other"

def calculate_job_specific_bonus(text: str, job_type: str) -> int:
    """Calculate ATS score bonus based on job-specific keywords and requirements"""
    text_lower = text.lower()
    bonus = 0
    
    # Technical keywords by job type
    job_keywords = {
        'software_engineer': ['python', 'java', 'javascript', 'react', 'node.js', 'git', 'agile', 'api', 'database'],
        'frontend_developer': ['react', 'vue', 'angular', 'javascript', 'typescript', 'css', 'html', 'responsive', 'ui'],
        'backend_developer': ['python', 'java', 'node.js', 'api', 'database', 'sql', 'microservices', 'rest', 'graphql'],
        'full_stack_developer': ['react', 'node.js', 'python', 'javascript', 'api', 'database', 'git', 'agile'],
        'mobile_developer': ['react native', 'flutter', 'swift', 'kotlin', 'ios', 'android', 'mobile', 'app store'],
        'data_scientist': ['python', 'r', 'machine learning', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'sql', 'statistics'],
        'data_engineer': ['python', 'sql', 'spark', 'hadoop', 'etl', 'pipeline', 'aws', 'kafka', 'airflow'],
        'machine_learning_engineer': ['python', 'tensorflow', 'pytorch', 'scikit-learn', 'mlops', 'docker', 'kubernetes'],
        'devops_engineer': ['docker', 'kubernetes', 'aws', 'jenkins', 'terraform', 'ansible', 'ci/cd', 'monitoring'],
        'cloud_engineer': ['aws', 'azure', 'gcp', 'terraform', 'kubernetes', 'docker', 'serverless', 'infrastructure'],
        'security_engineer': ['cybersecurity', 'penetration testing', 'vulnerability', 'encryption', 'firewall', 'compliance'],
        'qa_engineer': ['testing', 'automation', 'selenium', 'junit', 'quality assurance', 'bug tracking', 'test cases'],
        'ui_ux_designer': ['figma', 'sketch', 'adobe', 'user experience', 'wireframes', 'prototyping', 'design systems'],
        'product_manager': ['product management', 'roadmap', 'stakeholder', 'requirements', 'analytics', 'user stories'],
        'engineering_manager': ['team lead', 'management', 'mentoring', 'project management', 'technical leadership'],
        'solutions_architect': ['architecture', 'system design', 'scalability', 'microservices', 'cloud architecture'],
        'database_administrator': ['sql', 'mysql', 'postgresql', 'mongodb', 'database optimization', 'backup', 'recovery'],
    }
    
    keywords = job_keywords.get(job_type, [])
    
    # Count keyword matches
    for keyword in keywords:
        if keyword in text_lower:
            bonus += 2  # 2 points per relevant keyword
    
    # Cap bonus at 20 points
    return min(20, bonus)

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
    if ENHANCED_SERVICE_AVAILABLE:
        return enhanced_analysis_service.get_analysis_capabilities()
    else:
        return analysis_service.get_analysis_capabilities()

@app.post("/set_confidence_threshold")
async def set_confidence_threshold(threshold: float):
    """Set the confidence threshold for analysis suggestions"""
    try:
        if not (0.0 <= threshold <= 100.0):
            raise HTTPException(status_code=400, detail="Threshold must be between 0 and 100")
        
        # Update both services
        if ENHANCED_SERVICE_AVAILABLE:
            enhanced_analysis_service.set_confidence_threshold(threshold)
            new_threshold = enhanced_analysis_service.config.confidence_threshold
        else:
            analysis_service.set_confidence_threshold(threshold)
            new_threshold = analysis_service.confidence_threshold
        
        return {
            "message": f"Confidence threshold set to {threshold}%",
            "new_threshold": new_threshold,
            "enhanced_service_used": ENHANCED_SERVICE_AVAILABLE
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

@app.post("/analyze_resume_enhanced")
async def analyze_resume_enhanced(request: AnalyzeRequest):
    """
    Enhanced text analysis with world-class multi-layer detection
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Use enhanced service if available
        if ENHANCED_SERVICE_AVAILABLE:
            result = enhanced_analysis_service.analyze_text(
                text=request.text,
                check_spelling=request.check_spelling,
                check_grammar=request.check_grammar
            )
            
            # Convert to API response format
            return {
                "typos": [
                    {
                        "word": typo.word,
                        "suggestion": typo.suggestion,
                        "confidence": typo.confidence_score,
                        "explanation": typo.explanation,
                        "context": typo.context,
                        "position": typo.position,
                        "validation_status": typo.validation_status,
                        "layer_results": len(typo.layer_results) if typo.layer_results else 0
                    } for typo in result.typos
                ],
                "grammar_issues": [
                    {
                        "sentence": issue.sentence,
                        "suggestion": issue.suggestion,
                        "confidence": issue.confidence_score,
                        "explanation": issue.explanation,
                        "issue_type": issue.issue_type,
                        "position": issue.position,
                        "validation_status": issue.validation_status,
                        "layer_results": len(issue.layer_results) if issue.layer_results else 0
                    } for issue in result.grammar_issues
                ],
                "summary": {
                    "total_typos": len(result.typos),
                    "total_grammar_issues": len(result.grammar_issues),
                    "processing_time": result.processing_time,
                    "layers_used": [layer.value for layer in result.layers_used],
                    "processing_status": result.processing_status.value,
                    "cache_hits": result.cache_hits,
                    "total_suggestions": result.total_suggestions,
                    "high_confidence_issues": result.high_confidence_issues
                },
                "analysis_version": "enhanced_v2.0"
            }
        else:
            # Fallback to original enhanced analysis
            capabilities = analysis_service.get_analysis_capabilities()
            if not capabilities['enhanced_analysis_available']:
                raise HTTPException(
                    status_code=503, 
                    detail="Enhanced analysis not available. Missing enhanced components."
                )
            
            result = analysis_service.analyze_with_confidence(
                text=request.text,
                check_spelling=request.check_spelling,
                check_grammar=request.check_grammar
            )
            
            return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...), job_type: str = Form("other")):
    print(f"🔥 RECEIVED FILE: {file.filename}, Size: {file.size}")
    
    # Validate and sanitize job type
    validated_job_type = validate_job_type(job_type)
    print(f"🎯 JOB TYPE: {validated_job_type}")
    
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
    
    # ✅ Enhanced analysis with quality metrics and job-specific scoring
    try:
        # Calculate job-specific ATS score
        base_ats_score = 75
        job_specific_bonus = calculate_job_specific_bonus(text, validated_job_type)
        final_ats_score = min(100, base_ats_score + job_specific_bonus)
    except Exception as e:
        print(f"⚠️ Error in job-specific analysis: {e}, using base score")
        final_ats_score = 75
    
    analysis = {
        "atsScore": final_ats_score,
        "jobType": validated_job_type,
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
    try:
        if ENHANCED_SERVICE_AVAILABLE:
            enhanced_analysis_service.cleanup()
        analysis_service.cleanup()
        print("✅ System cleanup completed successfully")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

@app.get("/system/health")
async def get_system_health():
    """Get comprehensive system health information"""
    try:
        if ENHANCED_SERVICE_AVAILABLE:
            return enhanced_analysis_service.validate_system_health()
        else:
            return {"status": "basic", "message": "Enhanced service not available"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/system/performance")
async def get_performance_report():
    """Get detailed performance report"""
    try:
        if ENHANCED_SERVICE_AVAILABLE:
            return enhanced_analysis_service.get_performance_report()
        else:
            return {"message": "Performance reporting requires enhanced service"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance report: {str(e)}")

@app.post("/system/optimize")
async def optimize_system():
    """Optimize system performance"""
    try:
        if ENHANCED_SERVICE_AVAILABLE:
            result = enhanced_analysis_service.optimize_performance()
            return {"message": "System optimization completed", "results": result}
        else:
            return {"message": "System optimization requires enhanced service"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System optimization failed: {str(e)}")

@app.get("/system/statistics")
async def get_system_statistics():
    """Get comprehensive system statistics"""
    try:
        if ENHANCED_SERVICE_AVAILABLE:
            return enhanced_analysis_service.get_system_statistics()
        else:
            return {
                "service_info": {"version": "basic_v1.0"},
                "message": "Full statistics require enhanced service"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
