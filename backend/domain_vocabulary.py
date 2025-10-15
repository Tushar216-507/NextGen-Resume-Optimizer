"""
Comprehensive domain vocabulary system for resume-specific typo detection.
Includes technical terms, industry jargon, company names, and contextual validation.
"""

import re
import json
import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import requests
from datetime import datetime, timedelta

from enhanced_models import ValidationResult
try:
    from enhanced_models import ValidationRule
except ImportError:
    # Fallback ValidationRule if not available
    from dataclasses import dataclass
    @dataclass
    class ValidationRule:
        rule_name: str
        rule_type: str
        pattern: str = None
        weight: float = 1.0
        enabled: bool = True
from core_interfaces import IDomainValidator

logger = logging.getLogger(__name__)

@dataclass
class TechnicalTerm:
    """Represents a technical term with metadata"""
    term: str
    variations: List[str]
    category: str
    confidence: float
    context_patterns: List[str]
    industry: Optional[str] = None
    
class IndustryContext:
    """Context information for different industries"""
    
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    MARKETING = "marketing"
    EDUCATION = "education"
    MANUFACTURING = "manufacturing"
    CONSULTING = "consulting"
    RETAIL = "retail"
    
    @classmethod
    def get_all_industries(cls) -> List[str]:
        return [
            cls.TECHNOLOGY, cls.FINANCE, cls.HEALTHCARE, cls.MARKETING,
            cls.EDUCATION, cls.MANUFACTURING, cls.CONSULTING, cls.RETAIL
        ]

class ComprehensiveDomainVocabulary:
    """
    Comprehensive domain vocabulary with advanced validation capabilities.
    Covers technical terms, frameworks, tools, companies, and industry-specific jargon.
    """
    
    def __init__(self, auto_update: bool = True):
        self.auto_update = auto_update
        self.last_update = None
        
        # Core vocabularies
        self.technical_terms: Dict[str, TechnicalTerm] = {}
        self.frameworks_and_tools: Dict[str, TechnicalTerm] = {}
        self.programming_languages: Dict[str, TechnicalTerm] = {}
        self.company_names: Set[str] = set()
        self.certifications: Set[str] = set()
        self.job_titles: Set[str] = set()
        self.industry_terms: Dict[str, Set[str]] = defaultdict(set)
        
        # Validation rules
        self.validation_rules: List[ValidationRule] = []
        
        # Context patterns
        self.context_patterns: Dict[str, List[str]] = {}
        
        # Initialize vocabularies
        self._initialize_technical_terms()
        self._initialize_frameworks_and_tools()
        self._initialize_programming_languages()
        self._initialize_company_names()
        self._initialize_certifications()
        self._initialize_job_titles()
        self._initialize_industry_terms()
        self._initialize_validation_rules()
        self._initialize_context_patterns()
        
        logger.info(f"Domain vocabulary initialized with {len(self.technical_terms)} technical terms")
    
    def _initialize_technical_terms(self):
        """Initialize comprehensive technical terms database"""
        terms_data = [
            # Cloud & Infrastructure
            ("aws", ["amazon web services", "amazon aws"], "cloud", 0.95, ["cloud", "infrastructure"]),
            ("azure", ["microsoft azure"], "cloud", 0.95, ["cloud", "microsoft"]),
            ("gcp", ["google cloud platform", "google cloud"], "cloud", 0.95, ["cloud", "google"]),
            ("kubernetes", ["k8s"], "orchestration", 0.90, ["container", "orchestration"]),
            ("docker", ["containerization"], "containerization", 0.95, ["container", "deployment"]),
            ("terraform", [], "infrastructure", 0.90, ["infrastructure", "iac"]),
            ("ansible", [], "automation", 0.85, ["automation", "configuration"]),
            ("jenkins", [], "ci_cd", 0.85, ["ci", "cd", "pipeline"]),
            ("gitlab", ["gitlab-ci"], "ci_cd", 0.85, ["git", "ci", "cd"]),
            ("circleci", ["circle ci"], "ci_cd", 0.80, ["ci", "cd"]),
            
            # Programming & Development
            ("api", ["apis", "rest api", "restful api"], "development", 0.95, ["development", "integration"]),
            ("restful", ["rest"], "api", 0.90, ["api", "web service"]),
            ("graphql", [], "api", 0.85, ["api", "query"]),
            ("microservices", ["micro services"], "architecture", 0.90, ["architecture", "distributed"]),
            ("serverless", [], "architecture", 0.85, ["cloud", "lambda"]),
            ("oauth", ["oauth2", "oauth 2.0"], "security", 0.85, ["authentication", "security"]),
            ("jwt", ["json web token"], "security", 0.80, ["token", "authentication"]),
            ("ssl", ["tls", "https"], "security", 0.90, ["security", "encryption"]),
            
            # Databases
            ("postgresql", ["postgres"], "database", 0.90, ["database", "sql"]),
            ("mysql", [], "database", 0.90, ["database", "sql"]),
            ("mongodb", ["mongo"], "database", 0.85, ["database", "nosql"]),
            ("redis", [], "database", 0.85, ["cache", "memory"]),
            ("elasticsearch", ["elastic search"], "search", 0.80, ["search", "analytics"]),
            ("cassandra", [], "database", 0.75, ["database", "nosql"]),
            ("dynamodb", ["dynamo db"], "database", 0.80, ["aws", "nosql"]),
            
            # Frontend Technologies
            ("javascript", ["js"], "programming", 0.95, ["frontend", "web"]),
            ("typescript", ["ts"], "programming", 0.90, ["frontend", "javascript"]),
            ("html", ["html5"], "markup", 0.95, ["frontend", "web"]),
            ("css", ["css3"], "styling", 0.95, ["frontend", "styling"]),
            ("scss", ["sass"], "styling", 0.85, ["css", "preprocessing"]),
            ("webpack", [], "build_tool", 0.80, ["bundling", "build"]),
            ("babel", [], "transpiler", 0.75, ["javascript", "transpiling"]),
            ("eslint", [], "linting", 0.75, ["code quality", "linting"]),
            
            # Backend Technologies
            ("nodejs", ["node.js", "node js"], "runtime", 0.90, ["backend", "javascript"]),
            ("express", ["expressjs", "express.js"], "framework", 0.85, ["nodejs", "web"]),
            ("fastapi", [], "framework", 0.80, ["python", "api"]),
            ("django", [], "framework", 0.85, ["python", "web"]),
            ("flask", [], "framework", 0.80, ["python", "web"]),
            ("spring", ["spring boot"], "framework", 0.85, ["java", "enterprise"]),
            ("laravel", [], "framework", 0.80, ["php", "web"]),
            
            # Data Science & ML
            ("tensorflow", [], "ml_framework", 0.85, ["machine learning", "ai"]),
            ("pytorch", [], "ml_framework", 0.85, ["machine learning", "ai"]),
            ("scikit-learn", ["sklearn"], "ml_library", 0.80, ["machine learning", "python"]),
            ("pandas", [], "data_library", 0.85, ["data analysis", "python"]),
            ("numpy", [], "data_library", 0.85, ["numerical", "python"]),
            ("matplotlib", [], "visualization", 0.75, ["plotting", "python"]),
            ("seaborn", [], "visualization", 0.70, ["plotting", "python"]),
            ("jupyter", ["jupyter notebook"], "tool", 0.80, ["data science", "notebook"]),
            
            # DevOps & Monitoring
            ("prometheus", [], "monitoring", 0.75, ["monitoring", "metrics"]),
            ("grafana", [], "visualization", 0.75, ["monitoring", "dashboard"]),
            ("elk", ["elasticsearch logstash kibana"], "logging", 0.70, ["logging", "analytics"]),
            ("splunk", [], "logging", 0.70, ["logging", "analytics"]),
            ("nagios", [], "monitoring", 0.65, ["monitoring", "alerting"]),
            ("datadog", [], "monitoring", 0.70, ["monitoring", "apm"]),
            
            # Testing
            ("junit", [], "testing", 0.80, ["java", "unit testing"]),
            ("pytest", [], "testing", 0.80, ["python", "testing"]),
            ("jest", [], "testing", 0.80, ["javascript", "testing"]),
            ("selenium", [], "testing", 0.85, ["automation", "web testing"]),
            ("cypress", [], "testing", 0.75, ["e2e", "testing"]),
            ("postman", [], "testing", 0.80, ["api", "testing"]),
            
            # Methodologies
            ("agile", [], "methodology", 0.90, ["project management", "scrum"]),
            ("scrum", [], "methodology", 0.85, ["agile", "project management"]),
            ("kanban", [], "methodology", 0.80, ["agile", "workflow"]),
            ("devops", [], "methodology", 0.90, ["development", "operations"]),
            ("ci/cd", ["cicd", "continuous integration"], "methodology", 0.85, ["automation", "deployment"]),
            ("tdd", ["test driven development"], "methodology", 0.75, ["testing", "development"]),
            ("bdd", ["behavior driven development"], "methodology", 0.70, ["testing", "development"]),
        ]
        
        for term_data in terms_data:
            term, variations, category, confidence, patterns = term_data
            self.technical_terms[term.lower()] = TechnicalTerm(
                term=term,
                variations=variations,
                category=category,
                confidence=confidence,
                context_patterns=patterns
            )
    
    def _initialize_frameworks_and_tools(self):
        """Initialize frameworks and tools vocabulary"""
        frameworks_data = [
            # Frontend Frameworks
            ("react", ["reactjs", "react.js"], "frontend_framework", 0.95, ["frontend", "component"]),
            ("angular", ["angularjs"], "frontend_framework", 0.90, ["frontend", "typescript"]),
            ("vue", ["vuejs", "vue.js"], "frontend_framework", 0.85, ["frontend", "progressive"]),
            ("svelte", [], "frontend_framework", 0.75, ["frontend", "compiler"]),
            ("nextjs", ["next.js"], "frontend_framework", 0.80, ["react", "ssr"]),
            ("nuxtjs", ["nuxt.js"], "frontend_framework", 0.75, ["vue", "ssr"]),
            ("gatsby", [], "frontend_framework", 0.70, ["react", "static"]),
            
            # Mobile Frameworks
            ("react-native", ["react native"], "mobile_framework", 0.85, ["mobile", "react"]),
            ("flutter", [], "mobile_framework", 0.85, ["mobile", "dart"]),
            ("ionic", [], "mobile_framework", 0.75, ["mobile", "hybrid"]),
            ("xamarin", [], "mobile_framework", 0.70, ["mobile", "microsoft"]),
            
            # Backend Frameworks
            ("rails", ["ruby on rails"], "backend_framework", 0.80, ["ruby", "web"]),
            ("symfony", [], "backend_framework", 0.70, ["php", "web"]),
            ("aspnet", ["asp.net"], "backend_framework", 0.80, ["microsoft", "web"]),
            ("gin", [], "backend_framework", 0.70, ["go", "web"]),
            ("fiber", [], "backend_framework", 0.65, ["go", "web"]),
            
            # Development Tools
            ("git", [], "version_control", 0.95, ["version control", "repository"]),
            ("github", [], "platform", 0.90, ["git", "collaboration"]),
            ("bitbucket", [], "platform", 0.75, ["git", "atlassian"]),
            ("jira", [], "project_management", 0.85, ["project", "tracking"]),
            ("confluence", [], "documentation", 0.75, ["wiki", "documentation"]),
            ("slack", [], "communication", 0.85, ["team", "communication"]),
            ("teams", ["microsoft teams"], "communication", 0.80, ["microsoft", "collaboration"]),
            
            # IDEs and Editors
            ("vscode", ["visual studio code"], "editor", 0.90, ["development", "editor"]),
            ("intellij", ["intellij idea"], "ide", 0.85, ["java", "development"]),
            ("pycharm", [], "ide", 0.80, ["python", "development"]),
            ("webstorm", [], "ide", 0.75, ["javascript", "development"]),
            ("sublime", ["sublime text"], "editor", 0.70, ["text", "editor"]),
            ("atom", [], "editor", 0.65, ["text", "editor"]),
        ]
        
        for framework_data in frameworks_data:
            term, variations, category, confidence, patterns = framework_data
            self.frameworks_and_tools[term.lower()] = TechnicalTerm(
                term=term,
                variations=variations,
                category=category,
                confidence=confidence,
                context_patterns=patterns
            )
    
    def _initialize_programming_languages(self):
        """Initialize programming languages vocabulary"""
        languages_data = [
            ("python", [], "programming_language", 0.95, ["backend", "data science"]),
            ("java", [], "programming_language", 0.95, ["enterprise", "backend"]),
            ("javascript", ["js"], "programming_language", 0.95, ["frontend", "web"]),
            ("typescript", ["ts"], "programming_language", 0.90, ["frontend", "typed"]),
            ("csharp", ["c#"], "programming_language", 0.90, ["microsoft", "backend"]),
            ("cplusplus", ["c++"], "programming_language", 0.85, ["systems", "performance"]),
            ("go", ["golang"], "programming_language", 0.85, ["backend", "concurrent"]),
            ("rust", [], "programming_language", 0.80, ["systems", "memory safe"]),
            ("kotlin", [], "programming_language", 0.80, ["android", "jvm"]),
            ("swift", [], "programming_language", 0.80, ["ios", "apple"]),
            ("php", [], "programming_language", 0.85, ["web", "backend"]),
            ("ruby", [], "programming_language", 0.80, ["web", "scripting"]),
            ("scala", [], "programming_language", 0.75, ["jvm", "functional"]),
            ("r", [], "programming_language", 0.80, ["statistics", "data science"]),
            ("matlab", [], "programming_language", 0.70, ["numerical", "engineering"]),
            ("dart", [], "programming_language", 0.70, ["flutter", "mobile"]),
        ]
        
        for lang_data in languages_data:
            term, variations, category, confidence, patterns = lang_data
            self.programming_languages[term.lower()] = TechnicalTerm(
                term=term,
                variations=variations,
                category=category,
                confidence=confidence,
                context_patterns=patterns
            )
    
    def _initialize_company_names(self):
        """Initialize major company names"""
        companies = [
            # Tech Giants
            "google", "microsoft", "amazon", "apple", "facebook", "meta",
            "netflix", "tesla", "uber", "airbnb", "spotify", "twitter",
            "linkedin", "github", "gitlab", "atlassian", "salesforce",
            "oracle", "ibm", "intel", "nvidia", "amd", "qualcomm",
            
            # Cloud Providers
            "aws", "azure", "gcp", "digitalocean", "linode", "vultr",
            "heroku", "vercel", "netlify", "cloudflare",
            
            # Consulting & Services
            "accenture", "deloitte", "pwc", "kpmg", "ey", "mckinsey",
            "bcg", "bain", "capgemini", "tcs", "infosys", "wipro",
            
            # Startups & Unicorns
            "stripe", "shopify", "zoom", "slack", "notion", "figma",
            "canva", "discord", "twitch", "reddit", "pinterest",
            
            # Financial
            "jpmorgan", "goldman sachs", "morgan stanley", "blackrock",
            "visa", "mastercard", "paypal", "square", "robinhood"
        ]
        
        self.company_names.update(companies)
    
    def _initialize_certifications(self):
        """Initialize professional certifications"""
        certifications = [
            # Cloud Certifications
            "aws certified solutions architect", "aws certified developer",
            "aws certified sysops administrator", "azure fundamentals",
            "azure administrator", "azure developer", "azure solutions architect",
            "google cloud professional", "gcp associate cloud engineer",
            
            # Programming Certifications
            "oracle certified java programmer", "microsoft certified",
            "python institute certification", "javascript certification",
            
            # Project Management
            "pmp", "project management professional", "scrum master",
            "certified scrum master", "csm", "safe", "agile certification",
            
            # Security
            "cissp", "ceh", "certified ethical hacker", "comptia security+",
            "cisa", "cism", "gsec",
            
            # Data & Analytics
            "tableau certified", "power bi certification", "google analytics",
            "hadoop certification", "spark certification",
            
            # DevOps
            "docker certified", "kubernetes certification", "jenkins certification",
            "terraform certification", "ansible certification"
        ]
        
        self.certifications.update(certifications)
    
    def _initialize_job_titles(self):
        """Initialize common job titles"""
        job_titles = [
            # Engineering
            "software engineer", "senior software engineer", "staff engineer",
            "principal engineer", "engineering manager", "tech lead",
            "frontend developer", "backend developer", "full stack developer",
            "mobile developer", "ios developer", "android developer",
            
            # Data & AI
            "data scientist", "senior data scientist", "data engineer",
            "machine learning engineer", "ai engineer", "data analyst",
            "business intelligence analyst", "research scientist",
            
            # DevOps & Infrastructure
            "devops engineer", "site reliability engineer", "sre",
            "cloud engineer", "infrastructure engineer", "platform engineer",
            "security engineer", "network engineer",
            
            # Product & Design
            "product manager", "senior product manager", "product owner",
            "ui designer", "ux designer", "product designer",
            "design systems engineer", "user researcher",
            
            # Leadership
            "cto", "chief technology officer", "vp engineering",
            "director of engineering", "head of engineering",
            "architect", "solutions architect", "enterprise architect",
            
            # QA & Testing
            "qa engineer", "test engineer", "automation engineer",
            "quality assurance analyst", "sdet"
        ]
        
        self.job_titles.update(job_titles)
    
    def _initialize_industry_terms(self):
        """Initialize industry-specific terms"""
        industry_terms = {
            IndustryContext.TECHNOLOGY: {
                "saas", "paas", "iaas", "b2b", "b2c", "mvp", "poc",
                "scalability", "performance", "optimization", "refactoring",
                "technical debt", "code review", "pair programming",
                "open source", "proprietary", "enterprise", "startup"
            },
            IndustryContext.FINANCE: {
                "fintech", "blockchain", "cryptocurrency", "trading",
                "algorithmic trading", "risk management", "compliance",
                "kyc", "aml", "pci dss", "sox", "gdpr", "regulatory"
            },
            IndustryContext.HEALTHCARE: {
                "hipaa", "ehr", "emr", "telemedicine", "healthtech",
                "medical devices", "fda", "clinical trials", "patient data"
            },
            IndustryContext.MARKETING: {
                "martech", "crm", "marketing automation", "seo", "sem",
                "social media", "content management", "analytics",
                "conversion optimization", "a/b testing"
            },
            IndustryContext.EDUCATION: {
                "edtech", "lms", "learning management system", "mooc",
                "e-learning", "educational technology", "student information system"
            }
        }
        
        self.industry_terms.update(industry_terms)
    
    def _initialize_validation_rules(self):
        """Initialize validation rules for domain terms"""
        rules = [
            ValidationRule(
                rule_name="technical_term_context",
                rule_type="context_validation",
                weight=0.9,
                enabled=True
            ),
            ValidationRule(
                rule_name="company_name_capitalization",
                rule_type="capitalization",
                pattern=r"^[A-Z][a-z]*(?:\s[A-Z][a-z]*)*$",
                weight=0.8,
                enabled=True
            ),
            ValidationRule(
                rule_name="certification_format",
                rule_type="format_validation",
                weight=0.85,
                enabled=True
            ),
            ValidationRule(
                rule_name="programming_language_context",
                rule_type="context_validation",
                weight=0.9,
                enabled=True
            )
        ]
        
        self.validation_rules.extend(rules)
    
    def _initialize_context_patterns(self):
        """Initialize context patterns for better validation"""
        patterns = {
            "technical_skills": [
                r"skills?:?\s*",
                r"technologies?:?\s*",
                r"programming languages?:?\s*",
                r"frameworks?:?\s*",
                r"tools?:?\s*"
            ],
            "experience": [
                r"experience with",
                r"worked with",
                r"used",
                r"implemented",
                r"developed using"
            ],
            "project_description": [
                r"built using",
                r"developed with",
                r"implemented in",
                r"created using"
            ],
            "job_requirements": [
                r"required:?\s*",
                r"must have:?\s*",
                r"looking for:?\s*",
                r"qualifications?:?\s*"
            ]
        }
        
        self.context_patterns.update(patterns)

class EnhancedDomainValidator(IDomainValidator):
    """
    Enhanced domain validator with comprehensive vocabulary and intelligent validation.
    """
    
    def __init__(self, vocabulary: Optional[ComprehensiveDomainVocabulary] = None):
        self.vocabulary = vocabulary or ComprehensiveDomainVocabulary()
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.context_analyzer = ContextAnalyzer()
        
        logger.info("Enhanced domain validator initialized")
    
    def is_valid_technical_term(self, word: str, context: str) -> bool:
        """
        Check if a word is a valid technical term in the given context.
        
        Args:
            word: Word to validate
            context: Surrounding text context
            
        Returns:
            True if the word is a valid technical term
        """
        word_lower = word.lower()
        
        # Check direct matches
        if word_lower in self.vocabulary.technical_terms:
            return self._validate_term_context(word_lower, context, "technical")
        
        if word_lower in self.vocabulary.frameworks_and_tools:
            return self._validate_term_context(word_lower, context, "framework")
        
        if word_lower in self.vocabulary.programming_languages:
            return self._validate_term_context(word_lower, context, "language")
        
        # Check variations
        for term_dict in [self.vocabulary.technical_terms, 
                         self.vocabulary.frameworks_and_tools,
                         self.vocabulary.programming_languages]:
            for term_key, term_obj in term_dict.items():
                if word_lower in [v.lower() for v in term_obj.variations]:
                    return self._validate_term_context(term_key, context, term_obj.category)
        
        # Check company names
        if word_lower in self.vocabulary.company_names:
            return self._validate_company_context(word, context)
        
        # Check certifications
        if any(word_lower in cert.lower() for cert in self.vocabulary.certifications):
            return True
        
        return False
    
    def _validate_term_context(self, term: str, context: str, term_type: str) -> bool:
        """Validate term based on context patterns"""
        context_lower = context.lower()
        
        # Get term object
        term_obj = None
        if term_type == "technical":
            term_obj = self.vocabulary.technical_terms.get(term)
        elif term_type == "framework":
            term_obj = self.vocabulary.frameworks_and_tools.get(term)
        elif term_type == "language":
            term_obj = self.vocabulary.programming_languages.get(term)
        
        if not term_obj:
            return False
        
        # Check if context matches expected patterns
        for pattern in term_obj.context_patterns:
            if pattern.lower() in context_lower:
                return True
        
        # Check general technical context patterns
        technical_patterns = [
            "skill", "technology", "framework", "language", "tool",
            "experience", "knowledge", "proficient", "familiar"
        ]
        
        for pattern in technical_patterns:
            if pattern in context_lower:
                return True
        
        return False
    
    def _validate_company_context(self, word: str, context: str) -> bool:
        """Validate company name in context"""
        context_lower = context.lower()
        
        company_patterns = [
            "worked at", "employed by", "company", "organization",
            "employer", "client", "contractor", "consultant"
        ]
        
        for pattern in company_patterns:
            if pattern in context_lower:
                return True
        
        return False
    
    def get_context_appropriate_suggestions(self, word: str, context: str) -> List[str]:
        """
        Get context-appropriate suggestions for a word.
        
        Args:
            word: Original word
            context: Surrounding text context
            
        Returns:
            List of context-appropriate suggestions
        """
        suggestions = []
        word_lower = word.lower()
        context_lower = context.lower()
        
        # Analyze context to determine what type of suggestions to provide
        context_type = self._analyze_context_type(context)
        
        # Get suggestions based on context type
        if context_type == "technical_skills":
            suggestions.extend(self._get_technical_suggestions(word_lower))
        elif context_type == "programming":
            suggestions.extend(self._get_programming_suggestions(word_lower))
        elif context_type == "tools_frameworks":
            suggestions.extend(self._get_framework_suggestions(word_lower))
        elif context_type == "company":
            suggestions.extend(self._get_company_suggestions(word_lower))
        
        # Add fuzzy matches from all vocabularies
        suggestions.extend(self._get_fuzzy_matches(word_lower))
        
        # Remove duplicates and sort by relevance
        unique_suggestions = list(dict.fromkeys(suggestions))
        return self._rank_suggestions(unique_suggestions, word, context)
    
    def _analyze_context_type(self, context: str) -> str:
        """Analyze context to determine the type of suggestions needed"""
        context_lower = context.lower()
        
        # Technical skills context
        if any(pattern in context_lower for pattern in [
            "skill", "technology", "proficient", "experience with"
        ]):
            return "technical_skills"
        
        # Programming context
        if any(pattern in context_lower for pattern in [
            "programming", "language", "code", "development"
        ]):
            return "programming"
        
        # Tools and frameworks context
        if any(pattern in context_lower for pattern in [
            "framework", "tool", "library", "platform"
        ]):
            return "tools_frameworks"
        
        # Company context
        if any(pattern in context_lower for pattern in [
            "company", "worked at", "employed", "organization"
        ]):
            return "company"
        
        return "general"
    
    def _get_technical_suggestions(self, word: str) -> List[str]:
        """Get technical term suggestions"""
        suggestions = []
        
        for term_key, term_obj in self.vocabulary.technical_terms.items():
            if self._is_similar(word, term_key):
                suggestions.append(term_obj.term)
            
            for variation in term_obj.variations:
                if self._is_similar(word, variation.lower()):
                    suggestions.append(variation)
        
        return suggestions
    
    def _get_programming_suggestions(self, word: str) -> List[str]:
        """Get programming language suggestions"""
        suggestions = []
        
        for lang_key, lang_obj in self.vocabulary.programming_languages.items():
            if self._is_similar(word, lang_key):
                suggestions.append(lang_obj.term)
            
            for variation in lang_obj.variations:
                if self._is_similar(word, variation.lower()):
                    suggestions.append(variation)
        
        return suggestions
    
    def _get_framework_suggestions(self, word: str) -> List[str]:
        """Get framework and tool suggestions"""
        suggestions = []
        
        for tool_key, tool_obj in self.vocabulary.frameworks_and_tools.items():
            if self._is_similar(word, tool_key):
                suggestions.append(tool_obj.term)
            
            for variation in tool_obj.variations:
                if self._is_similar(word, variation.lower()):
                    suggestions.append(variation)
        
        return suggestions
    
    def _get_company_suggestions(self, word: str) -> List[str]:
        """Get company name suggestions"""
        suggestions = []
        
        for company in self.vocabulary.company_names:
            if self._is_similar(word, company.lower()):
                suggestions.append(company.title())
        
        return suggestions
    
    def _get_fuzzy_matches(self, word: str) -> List[str]:
        """Get fuzzy matches from all vocabularies"""
        suggestions = []
        
        # Check all vocabularies for fuzzy matches
        all_terms = []
        
        # Add technical terms
        for term_obj in self.vocabulary.technical_terms.values():
            all_terms.append(term_obj.term)
            all_terms.extend(term_obj.variations)
        
        # Add frameworks and tools
        for tool_obj in self.vocabulary.frameworks_and_tools.values():
            all_terms.append(tool_obj.term)
            all_terms.extend(tool_obj.variations)
        
        # Add programming languages
        for lang_obj in self.vocabulary.programming_languages.values():
            all_terms.append(lang_obj.term)
            all_terms.extend(lang_obj.variations)
        
        # Find fuzzy matches
        for term in all_terms:
            if self._is_similar(word, term.lower()):
                suggestions.append(term)
        
        return suggestions
    
    def _is_similar(self, word1: str, word2: str, threshold: float = 0.7) -> bool:
        """Check if two words are similar using edit distance"""
        if abs(len(word1) - len(word2)) > 3:
            return False
        
        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(word1, word2)
        max_len = max(len(word1), len(word2))
        
        if max_len == 0:
            return True
        
        similarity = 1 - (distance / max_len)
        return similarity >= threshold
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
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
    
    def _rank_suggestions(self, suggestions: List[str], original: str, context: str) -> List[str]:
        """Rank suggestions by relevance"""
        scored_suggestions = []
        
        for suggestion in suggestions:
            score = 0.0
            
            # Similarity to original word
            similarity = 1 - (self._levenshtein_distance(original.lower(), suggestion.lower()) / 
                             max(len(original), len(suggestion)))
            score += similarity * 0.4
            
            # Context relevance
            if self.is_valid_technical_term(suggestion, context):
                score += 0.3
            
            # Frequency/popularity (simplified)
            if suggestion.lower() in ["python", "javascript", "react", "aws", "docker"]:
                score += 0.2
            
            # Length similarity
            length_diff = abs(len(original) - len(suggestion))
            length_score = max(0, 1 - (length_diff / 10))
            score += length_score * 0.1
            
            scored_suggestions.append((suggestion, score))
        
        # Sort by score and return top suggestions
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        return [suggestion for suggestion, _ in scored_suggestions[:5]]
    
    def validate_suggestion(self, original: str, suggestion: str, context: str) -> ValidationResult:
        """
        Validate a suggestion against domain rules.
        
        Args:
            original: Original word
            suggestion: Suggested correction
            context: Surrounding text context
            
        Returns:
            ValidationResult with validation details
        """
        # Check cache first
        cache_key = f"{original}_{suggestion}_{hash(context)}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        is_valid = True
        confidence_adjustment = 0.0
        explanation = ""
        rules_applied = []
        
        # Apply validation rules
        for rule in self.validation_rules:
            if not rule.enabled:
                continue
            
            rule_result = self._apply_validation_rule(rule, original, suggestion, context)
            
            if not rule_result['passed']:
                is_valid = False
                confidence_adjustment -= rule.weight * 10
                explanation += f"{rule.rule_name}: {rule_result['reason']}. "
            else:
                confidence_adjustment += rule.weight * 5
            
            rules_applied.append(rule.rule_name)
        
        # Check if suggestion is a known technical term
        if self.is_valid_technical_term(suggestion, context):
            confidence_adjustment += 15
            explanation += "Suggestion is a recognized technical term. "
        
        # Create validation result
        result = ValidationResult(
            is_valid=is_valid,
            confidence_adjustment=confidence_adjustment,
            explanation=explanation.strip(),
            rules_applied=rules_applied
        )
        
        # Cache result
        self.validation_cache[cache_key] = result
        
        return result
    
    def _apply_validation_rule(self, rule: ValidationRule, original: str, 
                             suggestion: str, context: str) -> Dict[str, Any]:
        """Apply a specific validation rule"""
        if rule.rule_type == "context_validation":
            return self._validate_context_rule(rule, original, suggestion, context)
        elif rule.rule_type == "capitalization":
            return self._validate_capitalization_rule(rule, original, suggestion, context)
        elif rule.rule_type == "format_validation":
            return self._validate_format_rule(rule, original, suggestion, context)
        else:
            return {'passed': True, 'reason': 'Unknown rule type'}
    
    def _validate_context_rule(self, rule: ValidationRule, original: str,
                             suggestion: str, context: str) -> Dict[str, Any]:
        """Validate context-based rules"""
        if rule.rule_name == "technical_term_context":
            if self.is_valid_technical_term(suggestion, context):
                return {'passed': True, 'reason': 'Valid technical term in context'}
            else:
                return {'passed': False, 'reason': 'Not a valid technical term in this context'}
        
        return {'passed': True, 'reason': 'Context validation passed'}
    
    def _validate_capitalization_rule(self, rule: ValidationRule, original: str,
                                    suggestion: str, context: str) -> Dict[str, Any]:
        """Validate capitalization rules"""
        if rule.rule_name == "company_name_capitalization":
            if suggestion.lower() in self.vocabulary.company_names:
                # Check if properly capitalized
                if suggestion.istitle() or suggestion.isupper():
                    return {'passed': True, 'reason': 'Proper company name capitalization'}
                else:
                    return {'passed': False, 'reason': 'Company name should be properly capitalized'}
        
        return {'passed': True, 'reason': 'Capitalization validation passed'}
    
    def _validate_format_rule(self, rule: ValidationRule, original: str,
                            suggestion: str, context: str) -> Dict[str, Any]:
        """Validate format-based rules"""
        if rule.rule_name == "certification_format":
            if any(suggestion.lower() in cert.lower() for cert in self.vocabulary.certifications):
                return {'passed': True, 'reason': 'Valid certification format'}
        
        return {'passed': True, 'reason': 'Format validation passed'}
    
    def get_industry_terms(self, industry: str) -> List[str]:
        """Get terms specific to an industry"""
        return list(self.vocabulary.industry_terms.get(industry, set()))
    
    def update_vocabulary(self, new_terms: List[str]) -> None:
        """Update the domain vocabulary with new terms"""
        for term in new_terms:
            # Add to technical terms with default confidence
            self.vocabulary.technical_terms[term.lower()] = TechnicalTerm(
                term=term,
                variations=[],
                category="custom",
                confidence=0.7,
                context_patterns=["custom"]
            )
        
        logger.info(f"Added {len(new_terms)} new terms to vocabulary")

class ContextAnalyzer:
    """Analyzes context to provide better validation"""
    
    def __init__(self):
        self.section_patterns = {
            'skills': r'(?i)(skills?|technologies?|technical\s+skills?)',
            'experience': r'(?i)(experience|work\s+history|employment)',
            'education': r'(?i)(education|academic|degree|university|college)',
            'projects': r'(?i)(projects?|portfolio|work\s+samples?)',
            'certifications': r'(?i)(certifications?|licenses?|credentials?)'
        }
    
    def analyze_section_context(self, text: str, position: int) -> str:
        """Analyze what section of resume the text is in"""
        # Get text before the position to determine section
        before_text = text[:position]
        
        # Find the most recent section header
        for section, pattern in self.section_patterns.items():
            matches = list(re.finditer(pattern, before_text))
            if matches:
                # Return the section of the most recent match
                return section
        
        return "general"
    
    def get_context_window(self, text: str, position: int, window_size: int = 100) -> str:
        """Get context window around a position"""
        start = max(0, position - window_size)
        end = min(len(text), position + window_size)
        return text[start:end]