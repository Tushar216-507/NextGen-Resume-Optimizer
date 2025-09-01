"""
Domain Vocabulary Management System
Handles technical terms, company names, and professional phrases
"""

import re
from typing import Set, List, Dict, Optional
from dataclasses import dataclass

@dataclass
class VocabularyEntry:
    """Represents a vocabulary entry with context"""
    term: str
    category: str
    variations: List[str]
    context_indicators: List[str]

class DomainVocabulary:
    """Manages domain-specific vocabulary for resume analysis"""
    
    def __init__(self):
        self.technical_terms = self._load_technical_terms()
        self.company_names = self._load_company_names()
        self.professional_phrases = self._load_professional_phrases()
        self.frameworks_and_tools = self._load_frameworks_and_tools()
        self.certifications = self._load_certifications()
        
    def _load_technical_terms(self) -> Dict[str, VocabularyEntry]:
        """Load programming languages and technical terms"""
        terms = {
            # Programming Languages
            'javascript': VocabularyEntry('javascript', 'programming', ['js', 'ecmascript'], ['frontend', 'web', 'node']),
            'typescript': VocabularyEntry('typescript', 'programming', ['ts'], ['frontend', 'web', 'type']),
            'python': VocabularyEntry('python', 'programming', ['py'], ['backend', 'data', 'ml', 'ai']),
            'java': VocabularyEntry('java', 'programming', ['jvm'], ['backend', 'enterprise', 'spring']),
            'csharp': VocabularyEntry('c#', 'programming', ['c-sharp', 'dotnet'], ['microsoft', '.net', 'backend']),
            'cplusplus': VocabularyEntry('c++', 'programming', ['cpp'], ['systems', 'performance', 'embedded']),
            'golang': VocabularyEntry('go', 'programming', ['golang'], ['backend', 'microservices', 'concurrent']),
            'rust': VocabularyEntry('rust', 'programming', [], ['systems', 'performance', 'memory']),
            'kotlin': VocabularyEntry('kotlin', 'programming', [], ['android', 'jvm', 'mobile']),
            'swift': VocabularyEntry('swift', 'programming', [], ['ios', 'apple', 'mobile']),
            'php': VocabularyEntry('php', 'programming', [], ['web', 'backend', 'server']),
            'ruby': VocabularyEntry('ruby', 'programming', [], ['web', 'rails', 'backend']),
            'scala': VocabularyEntry('scala', 'programming', [], ['jvm', 'functional', 'big data']),
            'r': VocabularyEntry('r', 'programming', [], ['statistics', 'data science', 'analysis']),
            'matlab': VocabularyEntry('matlab', 'programming', [], ['engineering', 'scientific', 'analysis']),
            
            # Web Technologies
            'html': VocabularyEntry('html', 'web', ['html5'], ['frontend', 'web', 'markup']),
            'css': VocabularyEntry('css', 'web', ['css3'], ['frontend', 'styling', 'web']),
            'scss': VocabularyEntry('scss', 'web', ['sass'], ['css', 'preprocessing', 'styling']),
            'less': VocabularyEntry('less', 'web', [], ['css', 'preprocessing', 'styling']),
            'xml': VocabularyEntry('xml', 'markup', [], ['data', 'configuration', 'markup']),
            'json': VocabularyEntry('json', 'data', [], ['api', 'data', 'configuration']),
            'yaml': VocabularyEntry('yaml', 'data', ['yml'], ['configuration', 'devops', 'data']),
            
            # Databases
            'sql': VocabularyEntry('sql', 'database', [], ['database', 'query', 'data']),
            'nosql': VocabularyEntry('nosql', 'database', [], ['database', 'document', 'non-relational']),
            'postgresql': VocabularyEntry('postgresql', 'database', ['postgres'], ['relational', 'database', 'sql']),
            'mysql': VocabularyEntry('mysql', 'database', [], ['relational', 'database', 'sql']),
            'mongodb': VocabularyEntry('mongodb', 'database', ['mongo'], ['nosql', 'document', 'database']),
            'redis': VocabularyEntry('redis', 'database', [], ['cache', 'in-memory', 'key-value']),
            'elasticsearch': VocabularyEntry('elasticsearch', 'database', ['elastic'], ['search', 'analytics', 'nosql']),
            'cassandra': VocabularyEntry('cassandra', 'database', [], ['nosql', 'distributed', 'big data']),
            'dynamodb': VocabularyEntry('dynamodb', 'database', [], ['aws', 'nosql', 'serverless']),
            
            # Data Science & ML
            'machinelearning': VocabularyEntry('machine learning', 'ai', ['ml'], ['ai', 'data science', 'algorithms']),
            'artificialintelligence': VocabularyEntry('artificial intelligence', 'ai', ['ai'], ['ml', 'deep learning', 'algorithms']),
            'deeplearning': VocabularyEntry('deep learning', 'ai', ['dl'], ['neural networks', 'ai', 'ml']),
            'datascience': VocabularyEntry('data science', 'data', [], ['analytics', 'statistics', 'ml']),
            'bigdata': VocabularyEntry('big data', 'data', [], ['analytics', 'distributed', 'scale']),
            'analytics': VocabularyEntry('analytics', 'data', [], ['data', 'insights', 'business intelligence']),
        }
        return terms
    
    def _load_frameworks_and_tools(self) -> Dict[str, VocabularyEntry]:
        """Load frameworks, libraries, and development tools"""
        tools = {
            # Frontend Frameworks
            'react': VocabularyEntry('react', 'frontend', ['reactjs'], ['javascript', 'ui', 'component']),
            'angular': VocabularyEntry('angular', 'frontend', ['angularjs'], ['typescript', 'spa', 'google']),
            'vue': VocabularyEntry('vue', 'frontend', ['vuejs'], ['javascript', 'progressive', 'spa']),
            'svelte': VocabularyEntry('svelte', 'frontend', [], ['javascript', 'compiler', 'lightweight']),
            'nextjs': VocabularyEntry('next.js', 'frontend', ['nextjs'], ['react', 'ssr', 'fullstack']),
            'nuxtjs': VocabularyEntry('nuxt.js', 'frontend', ['nuxtjs'], ['vue', 'ssr', 'fullstack']),
            'gatsby': VocabularyEntry('gatsby', 'frontend', [], ['react', 'static', 'jamstack']),
            
            # Backend Frameworks
            'nodejs': VocabularyEntry('node.js', 'backend', ['nodejs', 'node'], ['javascript', 'server', 'runtime']),
            'express': VocabularyEntry('express', 'backend', ['expressjs'], ['nodejs', 'web framework', 'api']),
            'fastapi': VocabularyEntry('fastapi', 'backend', [], ['python', 'api', 'async']),
            'django': VocabularyEntry('django', 'backend', [], ['python', 'web framework', 'mvc']),
            'flask': VocabularyEntry('flask', 'backend', [], ['python', 'micro framework', 'web']),
            'spring': VocabularyEntry('spring', 'backend', ['spring boot'], ['java', 'enterprise', 'framework']),
            'rails': VocabularyEntry('rails', 'backend', ['ruby on rails'], ['ruby', 'web framework', 'mvc']),
            
            # Cloud Platforms
            'aws': VocabularyEntry('aws', 'cloud', ['amazon web services'], ['cloud', 'amazon', 'infrastructure']),
            'azure': VocabularyEntry('azure', 'cloud', ['microsoft azure'], ['cloud', 'microsoft', 'infrastructure']),
            'gcp': VocabularyEntry('gcp', 'cloud', ['google cloud'], ['cloud', 'google', 'infrastructure']),
            'heroku': VocabularyEntry('heroku', 'cloud', [], ['platform', 'deployment', 'paas']),
            'vercel': VocabularyEntry('vercel', 'cloud', [], ['deployment', 'jamstack', 'frontend']),
            'netlify': VocabularyEntry('netlify', 'cloud', [], ['deployment', 'jamstack', 'static']),
            
            # DevOps Tools
            'docker': VocabularyEntry('docker', 'devops', [], ['containerization', 'deployment', 'microservices']),
            'kubernetes': VocabularyEntry('kubernetes', 'devops', ['k8s'], ['orchestration', 'containers', 'scaling']),
            'jenkins': VocabularyEntry('jenkins', 'devops', [], ['ci/cd', 'automation', 'build']),
            'gitlab': VocabularyEntry('gitlab', 'devops', [], ['git', 'ci/cd', 'repository']),
            'github': VocabularyEntry('github', 'devops', [], ['git', 'repository', 'collaboration']),
            'terraform': VocabularyEntry('terraform', 'devops', [], ['infrastructure', 'iac', 'provisioning']),
            'ansible': VocabularyEntry('ansible', 'devops', [], ['automation', 'configuration', 'deployment']),
            'prometheus': VocabularyEntry('prometheus', 'devops', [], ['monitoring', 'metrics', 'alerting']),
            'grafana': VocabularyEntry('grafana', 'devops', [], ['visualization', 'monitoring', 'dashboards']),
            
            # Data Science Libraries
            'tensorflow': VocabularyEntry('tensorflow', 'ml', ['tf'], ['machine learning', 'deep learning', 'google']),
            'pytorch': VocabularyEntry('pytorch', 'ml', [], ['machine learning', 'deep learning', 'facebook']),
            'sklearn': VocabularyEntry('scikit-learn', 'ml', ['sklearn'], ['machine learning', 'python', 'algorithms']),
            'pandas': VocabularyEntry('pandas', 'data', [], ['python', 'data analysis', 'dataframes']),
            'numpy': VocabularyEntry('numpy', 'data', [], ['python', 'numerical', 'arrays']),
            'matplotlib': VocabularyEntry('matplotlib', 'data', [], ['python', 'visualization', 'plotting']),
            'seaborn': VocabularyEntry('seaborn', 'data', [], ['python', 'visualization', 'statistical']),
            'jupyter': VocabularyEntry('jupyter', 'data', [], ['notebook', 'interactive', 'data science']),
            
            # Testing Frameworks
            'jest': VocabularyEntry('jest', 'testing', [], ['javascript', 'unit testing', 'facebook']),
            'mocha': VocabularyEntry('mocha', 'testing', [], ['javascript', 'testing framework', 'nodejs']),
            'pytest': VocabularyEntry('pytest', 'testing', [], ['python', 'testing framework', 'unit testing']),
            'junit': VocabularyEntry('junit', 'testing', [], ['java', 'unit testing', 'testing framework']),
            'selenium': VocabularyEntry('selenium', 'testing', [], ['automation', 'web testing', 'browser']),
            'cypress': VocabularyEntry('cypress', 'testing', [], ['e2e testing', 'web testing', 'automation']),
        }
        return tools
    
    def _load_company_names(self) -> Set[str]:
        """Load major technology company names"""
        companies = {
            # Major Tech Companies
            'google', 'microsoft', 'amazon', 'apple', 'facebook', 'meta', 'netflix', 'tesla',
            'nvidia', 'intel', 'amd', 'qualcomm', 'salesforce', 'oracle', 'ibm', 'cisco',
            'adobe', 'vmware', 'servicenow', 'workday', 'zoom', 'slack', 'atlassian',
            'shopify', 'stripe', 'square', 'paypal', 'uber', 'lyft', 'airbnb', 'spotify',
            'twitter', 'linkedin', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
            
            # Cloud Providers
            'aws', 'azure', 'gcp', 'digitalocean', 'linode', 'vultr', 'heroku', 'vercel',
            'netlify', 'cloudflare', 'fastly', 'akamai',
            
            # Databases & Analytics
            'mongodb', 'redis', 'elasticsearch', 'snowflake', 'databricks', 'palantir',
            'tableau', 'looker', 'powerbi', 'qlik', 'splunk',
            
            # Development Tools
            'jetbrains', 'intellij', 'pycharm', 'webstorm', 'vscode', 'sublime', 'atom',
            'postman', 'insomnia', 'swagger', 'docker', 'kubernetes', 'jenkins', 'travis',
            'circleci', 'bamboo', 'teamcity',
        }
        return companies
    
    def _load_professional_phrases(self) -> Dict[str, List[str]]:
        """Load professional phrases and their variations"""
        phrases = {
            'experience_phrases': [
                'responsible for', 'led a team of', 'collaborated with', 'worked closely with',
                'managed a team', 'oversaw the development', 'spearheaded the initiative',
                'coordinated with stakeholders', 'liaised with clients', 'facilitated meetings',
                'implemented solutions', 'developed strategies', 'optimized processes',
                'streamlined operations', 'enhanced performance', 'improved efficiency'
            ],
            'achievement_patterns': [
                r'\d+%\s+increase', r'reduced.*by\s+\d+%?', r'improved.*by\s+\d+%?',
                r'achieved.*\d+%?', r'exceeded.*by\s+\d+%?', r'delivered.*\d+%?\s+faster',
                r'saved.*\$\d+', r'generated.*\$\d+', r'managed.*\$\d+.*budget'
            ],
            'skill_indicators': [
                'proficient in', 'experienced with', 'expertise in', 'skilled in',
                'knowledgeable about', 'familiar with', 'competent in', 'advanced knowledge of',
                'strong background in', 'extensive experience with', 'deep understanding of'
            ],
            'education_phrases': [
                'bachelor of science', 'bachelor of arts', 'master of science', 'master of arts',
                'bachelor\'s degree', 'master\'s degree', 'phd', 'doctorate', 'associate degree',
                'certificate in', 'certification in', 'diploma in', 'coursework in'
            ]
        }
        return phrases
    
    def _load_certifications(self) -> Set[str]:
        """Load professional certifications"""
        certs = {
            # Cloud Certifications
            'aws certified', 'azure certified', 'gcp certified', 'google cloud certified',
            'aws solutions architect', 'aws developer', 'aws sysops', 'aws devops',
            'azure fundamentals', 'azure administrator', 'azure developer', 'azure architect',
            
            # Programming Certifications
            'oracle certified', 'microsoft certified', 'java certified', 'python certified',
            'javascript certified', 'react certified', 'angular certified',
            
            # Project Management
            'pmp', 'scrum master', 'agile certified', 'safe certified', 'kanban certified',
            'prince2', 'itil certified', 'six sigma', 'lean certified',
            
            # Security
            'cissp', 'cism', 'cisa', 'comptia security+', 'certified ethical hacker',
            'cissp certified', 'security+ certified',
            
            # Data & Analytics
            'tableau certified', 'power bi certified', 'google analytics certified',
            'salesforce certified', 'hubspot certified', 'marketo certified'
        }
        return certs
    
    def is_valid_technical_term(self, word: str, context: str = "") -> bool:
        """Check if a word is a valid technical term"""
        word_lower = word.lower()
        
        # Check direct matches
        if word_lower in self.technical_terms:
            return True
        
        if word_lower in self.frameworks_and_tools:
            return True
        
        if word_lower in self.company_names:
            return True
        
        if word_lower in self.certifications:
            return True
        
        # Check variations
        for term_data in self.technical_terms.values():
            if word_lower in [v.lower() for v in term_data.variations]:
                return True
        
        for tool_data in self.frameworks_and_tools.values():
            if word_lower in [v.lower() for v in tool_data.variations]:
                return True
        
        # Context-based validation
        if context:
            return self._validate_with_context(word_lower, context.lower())
        
        return False
    
    def _validate_with_context(self, word: str, context: str) -> bool:
        """Validate term based on surrounding context"""
        # Check if context contains indicators for technical terms
        tech_indicators = [
            'programming', 'development', 'software', 'web', 'mobile', 'app',
            'framework', 'library', 'tool', 'platform', 'service', 'api',
            'database', 'cloud', 'devops', 'frontend', 'backend', 'fullstack'
        ]
        
        for indicator in tech_indicators:
            if indicator in context:
                # More lenient validation in technical context
                return self._is_likely_technical_term(word)
        
        return False
    
    def _is_likely_technical_term(self, word: str) -> bool:
        """Heuristic check for likely technical terms"""
        # Common patterns in technical terms
        patterns = [
            r'^[a-z]+js$',  # ends with 'js' (reactjs, vuejs, etc.)
            r'^[a-z]+sql$',  # ends with 'sql' (mysql, postgresql, etc.)
            r'^[a-z]+db$',   # ends with 'db' (mongodb, dynamodb, etc.)
            r'^[a-z]+\.js$', # ends with '.js' (node.js, next.js, etc.)
            r'^[a-z]+api$',  # ends with 'api' (fastapi, restapi, etc.)
        ]
        
        for pattern in patterns:
            if re.match(pattern, word):
                return True
        
        return False
    
    def get_context_appropriate_suggestions(self, word: str, context: str) -> List[str]:
        """Get spelling suggestions appropriate for the context"""
        suggestions = []
        word_lower = word.lower()
        
        # Look for similar technical terms
        for term_key, term_data in self.technical_terms.items():
            if self._is_similar_word(word_lower, term_key):
                suggestions.append(term_data.term)
            
            for variation in term_data.variations:
                if self._is_similar_word(word_lower, variation.lower()):
                    suggestions.append(term_data.term)
        
        # Look for similar tools/frameworks
        for tool_key, tool_data in self.frameworks_and_tools.items():
            if self._is_similar_word(word_lower, tool_key):
                suggestions.append(tool_data.term)
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _is_similar_word(self, word1: str, word2: str) -> bool:
        """Check if two words are similar (simple edit distance)"""
        if abs(len(word1) - len(word2)) > 2:
            return False
        
        # Simple character overlap check
        overlap = len(set(word1) & set(word2))
        min_len = min(len(word1), len(word2))
        
        return min_len > 0 and overlap / min_len > 0.7
    
    def get_vocabulary_stats(self) -> Dict[str, int]:
        """Get statistics about loaded vocabulary"""
        return {
            'technical_terms': len(self.technical_terms),
            'frameworks_and_tools': len(self.frameworks_and_tools),
            'company_names': len(self.company_names),
            'certifications': len(self.certifications),
            'total_terms': (len(self.technical_terms) + len(self.frameworks_and_tools) + 
                          len(self.company_names) + len(self.certifications))
        }