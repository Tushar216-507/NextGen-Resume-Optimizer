"""
Context Analyzer for Resume-Specific Understanding
Analyzes resume structure, formatting, and professional context
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from models import ResumeContext, TextSection, FormattingStyle, ProfessionalLevel

@dataclass
class SectionPattern:
    """Pattern for identifying resume sections"""
    keywords: List[str]
    patterns: List[str]
    section_type: str
    priority: int

class ContextAnalyzer:
    """Analyzes resume context and structure"""
    
    def __init__(self):
        self.section_patterns = self._load_section_patterns()
        self.formatting_indicators = self._load_formatting_indicators()
        self.professional_indicators = self._load_professional_indicators()
        self.industry_keywords = self._load_industry_keywords()
    
    def _load_section_patterns(self) -> List[SectionPattern]:
        """Load patterns for identifying resume sections"""
        patterns = [
            SectionPattern(
                keywords=['experience', 'employment', 'work history', 'professional experience', 'career'],
                patterns=[r'(?i)(work\s+)?experience', r'(?i)employment\s+history', r'(?i)professional\s+experience'],
                section_type='experience',
                priority=1
            ),
            SectionPattern(
                keywords=['education', 'academic', 'degree', 'university', 'college'],
                patterns=[r'(?i)education', r'(?i)academic\s+background', r'(?i)degrees?'],
                section_type='education',
                priority=2
            ),
            SectionPattern(
                keywords=['skills', 'technical skills', 'competencies', 'expertise', 'technologies'],
                patterns=[r'(?i)(technical\s+)?skills', r'(?i)competencies', r'(?i)expertise'],
                section_type='skills',
                priority=3
            ),
            SectionPattern(
                keywords=['projects', 'portfolio', 'work samples', 'key projects'],
                patterns=[r'(?i)projects?', r'(?i)portfolio', r'(?i)key\s+projects'],
                section_type='projects',
                priority=4
            ),
            SectionPattern(
                keywords=['certifications', 'certificates', 'credentials', 'licenses'],
                patterns=[r'(?i)certifications?', r'(?i)certificates?', r'(?i)credentials'],
                section_type='certifications',
                priority=5
            ),
            SectionPattern(
                keywords=['contact', 'personal information', 'details'],
                patterns=[r'(?i)contact', r'(?i)personal\s+information'],
                section_type='contact',
                priority=6
            ),
            SectionPattern(
                keywords=['summary', 'objective', 'profile', 'about'],
                patterns=[r'(?i)summary', r'(?i)objective', r'(?i)professional\s+summary'],
                section_type='summary',
                priority=7
            ),
            SectionPattern(
                keywords=['achievements', 'accomplishments', 'awards', 'honors'],
                patterns=[r'(?i)achievements?', r'(?i)accomplishments?', r'(?i)awards?'],
                section_type='achievements',
                priority=8
            )
        ]
        return patterns
    
    def _load_formatting_indicators(self) -> Dict[str, List[str]]:
        """Load indicators for different formatting styles"""
        return {
            'bullet_points': ['•', '◦', '▪', '▫', '‣', '*', '-', '→'],
            'date_patterns': [
                r'\b\d{4}\s*[-–—]\s*\d{4}\b',  # 2020-2023
                r'\b\d{4}\s*[-–—]\s*present\b',  # 2020-present
                r'\b\w+\s+\d{4}\s*[-–—]\s*\w+\s+\d{4}\b',  # Jan 2020 - Dec 2023
                r'\b\d{1,2}/\d{4}\s*[-–—]\s*\d{1,2}/\d{4}\b',  # 01/2020 - 12/2023
            ],
            'section_headers': [
                r'^[A-Z\s]+$',  # ALL CAPS headers
                r'^[A-Z][a-z\s]+:?$',  # Title case headers
                r'^[A-Z][A-Z\s]+[A-Z]$',  # Mixed caps
            ],
            'contact_patterns': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
                r'\b(?:linkedin\.com/in/|github\.com/)\S+',  # Social profiles
            ]
        }
    
    def _load_professional_indicators(self) -> Dict[str, List[str]]:
        """Load indicators for professional level assessment"""
        return {
            'entry_level': [
                'recent graduate', 'new graduate', 'entry level', 'junior', 'intern',
                'seeking opportunities', 'eager to learn', 'fresh graduate',
                '0-2 years', 'less than 2 years', 'starting career'
            ],
            'mid_level': [
                '3-5 years', '2-7 years', 'experienced', 'skilled', 'proficient',
                'solid background', 'proven track record', 'demonstrated ability',
                'mid-level', 'intermediate'
            ],
            'senior_level': [
                'senior', 'lead', 'principal', 'architect', '5+ years', '7+ years',
                'extensive experience', 'deep expertise', 'advanced knowledge',
                'mentoring', 'leadership', 'team lead'
            ],
            'executive': [
                'director', 'manager', 'vp', 'vice president', 'cto', 'ceo', 'cfo',
                'head of', 'chief', 'executive', 'strategic', 'organizational',
                '10+ years', 'executive leadership'
            ]
        }
    
    def _load_industry_keywords(self) -> Dict[str, List[str]]:
        """Load keywords for industry identification"""
        return {
            'software_development': [
                'software', 'development', 'programming', 'coding', 'application',
                'web development', 'mobile development', 'full stack', 'frontend', 'backend'
            ],
            'data_science': [
                'data science', 'machine learning', 'artificial intelligence', 'analytics',
                'big data', 'data analysis', 'statistics', 'modeling', 'algorithms'
            ],
            'devops': [
                'devops', 'infrastructure', 'deployment', 'automation', 'ci/cd',
                'cloud', 'containers', 'orchestration', 'monitoring', 'scalability'
            ],
            'cybersecurity': [
                'security', 'cybersecurity', 'information security', 'penetration testing',
                'vulnerability', 'compliance', 'risk assessment', 'incident response'
            ],
            'product_management': [
                'product management', 'product owner', 'roadmap', 'stakeholder',
                'requirements', 'user stories', 'agile', 'scrum', 'product strategy'
            ],
            'design': [
                'design', 'ui/ux', 'user interface', 'user experience', 'visual design',
                'graphic design', 'prototyping', 'wireframes', 'design systems'
            ]
        }
    
    def analyze_resume_context(self, text: str) -> ResumeContext:
        """Analyze complete resume context"""
        sections = self.identify_sections(text)
        formatting_style = self.detect_formatting_style(text)
        professional_level = self.assess_professional_level(text)
        industry_indicators = self.detect_industry_indicators(text)
        technologies = self.detect_technologies(text)
        
        return ResumeContext(
            sections=sections,
            formatting_style=formatting_style,
            professional_level=professional_level,
            industry_indicators=industry_indicators,
            detected_technologies=technologies
        )
    
    def identify_sections(self, text: str) -> Dict[str, TextSection]:
        """Identify and extract resume sections"""
        sections = {}
        lines = text.split('\n')
        current_section = None
        section_content = []
        section_start = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this line is a section header
            detected_section = self._detect_section_header(line_stripped)
            
            if detected_section:
                # Save previous section if exists
                if current_section and section_content:
                    content = '\n'.join(section_content)
                    sections[current_section] = TextSection(
                        section_type=current_section,
                        content=content,
                        start_position=section_start,
                        end_position=i,
                        formatting_indicators=self._analyze_section_formatting(content)
                    )
                
                # Start new section
                current_section = detected_section
                section_content = []
                section_start = i
            else:
                # Add to current section content
                if current_section:
                    section_content.append(line)
        
        # Save last section
        if current_section and section_content:
            content = '\n'.join(section_content)
            sections[current_section] = TextSection(
                section_type=current_section,
                content=content,
                start_position=section_start,
                end_position=len(lines),
                formatting_indicators=self._analyze_section_formatting(content)
            )
        
        return sections
    
    def _detect_section_header(self, line: str) -> Optional[str]:
        """Detect if a line is a section header"""
        line_lower = line.lower()
        
        # Check against known patterns
        for pattern in self.section_patterns:
            # Check keywords
            for keyword in pattern.keywords:
                if keyword in line_lower:
                    return pattern.section_type
            
            # Check regex patterns
            for regex_pattern in pattern.patterns:
                if re.search(regex_pattern, line):
                    return pattern.section_type
        
        # Check formatting-based detection
        if self._looks_like_header(line):
            return self._infer_section_type(line_lower)
        
        return None
    
    def _looks_like_header(self, line: str) -> bool:
        """Check if line looks like a section header based on formatting"""
        # All caps
        if line.isupper() and len(line.split()) <= 4:
            return True
        
        # Title case with colon
        if line.endswith(':') and line[:-1].istitle():
            return True
        
        # Short line with title case
        if line.istitle() and len(line.split()) <= 3 and len(line) < 30:
            return True
        
        return False
    
    def _infer_section_type(self, line_lower: str) -> str:
        """Infer section type from header text"""
        if any(word in line_lower for word in ['work', 'job', 'employ', 'career']):
            return 'experience'
        elif any(word in line_lower for word in ['school', 'university', 'degree', 'education']):
            return 'education'
        elif any(word in line_lower for word in ['skill', 'technical', 'competenc']):
            return 'skills'
        elif any(word in line_lower for word in ['project', 'portfolio']):
            return 'projects'
        elif any(word in line_lower for word in ['contact', 'info', 'detail']):
            return 'contact'
        else:
            return 'other'
    
    def _analyze_section_formatting(self, content: str) -> List[str]:
        """Analyze formatting within a section"""
        indicators = []
        
        # Check for bullet points
        for bullet in self.formatting_indicators['bullet_points']:
            if bullet in content:
                indicators.append(f'bullet_point_{bullet}')
        
        # Check for date patterns
        for date_pattern in self.formatting_indicators['date_patterns']:
            if re.search(date_pattern, content):
                indicators.append('date_range')
        
        # Check for contact patterns
        for contact_pattern in self.formatting_indicators['contact_patterns']:
            if re.search(contact_pattern, content):
                indicators.append('contact_info')
        
        return indicators
    
    def detect_formatting_style(self, text: str) -> FormattingStyle:
        """Detect overall formatting style of the resume"""
        # Count different formatting elements
        bullet_count = sum(1 for bullet in self.formatting_indicators['bullet_points'] if bullet in text)
        has_dates = any(re.search(pattern, text) for pattern in self.formatting_indicators['date_patterns'])
        has_contact = any(re.search(pattern, text) for pattern in self.formatting_indicators['contact_patterns'])
        
        # Analyze structure
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        short_lines = sum(1 for line in lines if len(line) < 50)
        long_lines = sum(1 for line in lines if len(line) > 100)
        
        # Determine style based on characteristics
        if bullet_count > 5 and has_dates:
            return FormattingStyle.TRADITIONAL
        elif short_lines > long_lines and has_contact:
            return FormattingStyle.MODERN
        elif bullet_count < 3 and long_lines > short_lines:
            return FormattingStyle.CREATIVE
        else:
            return FormattingStyle.TECHNICAL
    
    def assess_professional_level(self, text: str) -> ProfessionalLevel:
        """Assess professional level based on content"""
        text_lower = text.lower()
        
        # Count indicators for each level
        level_scores = {}
        for level, indicators in self.professional_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            level_scores[level] = score
        
        # Find the level with highest score
        if not level_scores or max(level_scores.values()) == 0:
            return ProfessionalLevel.MID_LEVEL  # Default
        
        max_level = max(level_scores, key=level_scores.get)
        
        # Map to enum
        level_mapping = {
            'entry_level': ProfessionalLevel.ENTRY_LEVEL,
            'mid_level': ProfessionalLevel.MID_LEVEL,
            'senior_level': ProfessionalLevel.SENIOR_LEVEL,
            'executive': ProfessionalLevel.EXECUTIVE
        }
        
        return level_mapping.get(max_level, ProfessionalLevel.MID_LEVEL)
    
    def detect_industry_indicators(self, text: str) -> List[str]:
        """Detect industry indicators in the text"""
        text_lower = text.lower()
        detected_industries = []
        
        for industry, keywords in self.industry_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score >= 2:  # Require at least 2 keyword matches
                detected_industries.append(industry)
        
        return detected_industries
    
    def detect_technologies(self, text: str) -> List[str]:
        """Detect mentioned technologies and tools"""
        # This would integrate with DomainVocabulary
        # For now, simple pattern matching
        tech_patterns = [
            r'\b(?:JavaScript|Python|Java|C\+\+|C#|Go|Rust|Swift|Kotlin)\b',
            r'\b(?:React|Angular|Vue|Node\.js|Django|Flask|Spring|Rails)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git)\b',
            r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch)\b'
        ]
        
        technologies = []
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technologies.extend(matches)
        
        return list(set(technologies))  # Remove duplicates
    
    def validate_section_consistency(self, sections: Dict[str, TextSection]) -> List[Dict]:
        """Validate consistency within and between sections"""
        issues = []
        
        # Check experience section for tense consistency
        if 'experience' in sections:
            tense_issues = self._check_tense_consistency(sections['experience'].content)
            issues.extend(tense_issues)
        
        # Check date consistency across sections
        date_issues = self._check_date_consistency(sections)
        issues.extend(date_issues)
        
        # Check formatting consistency
        format_issues = self._check_formatting_consistency(sections)
        issues.extend(format_issues)
        
        return issues
    
    def _check_tense_consistency(self, experience_text: str) -> List[Dict]:
        """Check for tense consistency in experience section"""
        issues = []
        
        # Look for mixed tenses in bullet points
        lines = experience_text.split('\n')
        past_tense_verbs = []
        present_tense_verbs = []
        
        for line in lines:
            if any(bullet in line for bullet in self.formatting_indicators['bullet_points']):
                # Extract first word after bullet point
                cleaned_line = re.sub(r'^[•◦▪▫‣*\-→\s]+', '', line.strip())
                if cleaned_line:
                    first_word = cleaned_line.split()[0].lower()
                    
                    # Simple past/present tense detection
                    if first_word.endswith('ed') or first_word in ['led', 'managed', 'developed', 'created']:
                        past_tense_verbs.append(first_word)
                    elif first_word.endswith('ing') or first_word in ['manage', 'develop', 'create', 'lead']:
                        present_tense_verbs.append(first_word)
        
        # Flag if mixed tenses found
        if past_tense_verbs and present_tense_verbs:
            issues.append({
                'type': 'tense_inconsistency',
                'message': 'Mixed past and present tense verbs found in experience section',
                'past_verbs': past_tense_verbs[:3],
                'present_verbs': present_tense_verbs[:3]
            })
        
        return issues
    
    def _check_date_consistency(self, sections: Dict[str, TextSection]) -> List[Dict]:
        """Check for date consistency across sections"""
        issues = []
        
        # Extract dates from different sections
        all_dates = {}
        for section_name, section in sections.items():
            dates = self._extract_dates(section.content)
            if dates:
                all_dates[section_name] = dates
        
        # Check for overlapping employment dates
        if 'experience' in all_dates and len(all_dates['experience']) > 1:
            # Simple overlap check (would need more sophisticated logic)
            issues.append({
                'type': 'date_overlap_check',
                'message': 'Multiple date ranges found in experience - check for overlaps',
                'dates': all_dates['experience']
            })
        
        return issues
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract date patterns from text"""
        dates = []
        for pattern in self.formatting_indicators['date_patterns']:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        return dates
    
    def _check_formatting_consistency(self, sections: Dict[str, TextSection]) -> List[Dict]:
        """Check for formatting consistency across sections"""
        issues = []
        
        # Check bullet point consistency
        bullet_types = set()
        for section in sections.values():
            for indicator in section.formatting_indicators:
                if indicator.startswith('bullet_point_'):
                    bullet_types.add(indicator)
        
        if len(bullet_types) > 2:
            issues.append({
                'type': 'inconsistent_bullets',
                'message': 'Multiple bullet point styles used',
                'bullet_types': list(bullet_types)
            })
        
        return issues