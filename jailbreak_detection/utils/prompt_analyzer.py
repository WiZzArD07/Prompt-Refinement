import re
from typing import Tuple, List

class PromptAnalyzer:
    def __init__(self):
        # Common adversarial patterns
        self.adversarial_patterns = [
            # Dangerous content patterns
            r'bomb|explosive|detonate|blast|terrorist|attack',
            r'kill|murder|assassinate|suicide|harm|hurt',
            r'weapon|gun|firearm|ammunition|bullet',
            r'drug|narcotic|illegal|substance',
            r'hack|exploit|vulnerability|breach|attack',
            
            # Bypass commands
            r'ignore|override|bypass|circumvent|disregard',
            r'previous|earlier|before|last|former',
            r'ignore.*instructions|disregard.*rules|break.*rules',
            r'as.*human|pretend.*human|act.*human',
            r'confidential|secret|private|hidden',
            
            # Suspicious patterns
            r'how.*to.*make|how.*to.*create|how.*to.*build',
            r'step.*by.*step|instructions.*for|guide.*to',
            r'without.*detection|avoid.*detection|evade.*detection',
            r'undetectable|untraceable|anonymous',
            
            # Code injection patterns
            r'<script>|javascript:|eval\(|exec\(|system\(',
            r'SELECT.*FROM|INSERT.*INTO|DROP.*TABLE',
            r'rm.*-rf|chmod.*777|sudo.*su',
            
            # Social engineering patterns
            r'password|credential|login|account|access',
            r'personal.*information|private.*data|sensitive',
            r'credit.*card|bank.*account|social.*security'
        ]
        
        # Compile patterns
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.adversarial_patterns]
        
        # Risk levels and their descriptions
        self.risk_levels = {
            'high': 'High risk of dangerous or adversarial content detected',
            'medium': 'Medium risk of potentially harmful content detected',
            'low': 'Low risk of suspicious content detected',
            'safe': 'No dangerous or adversarial content detected'
        }
    
    def analyze_prompt(self, prompt: str) -> Tuple[str, List[str]]:
        """
        Analyze a prompt for adversarial content.
        Returns a tuple of (risk_level, list_of_issues)
        """
        issues = []
        risk_score = 0
        
        # Check for adversarial patterns
        for pattern in self.patterns:
            matches = pattern.findall(prompt)
            if matches:
                issues.append(f"Found potential dangerous pattern: {matches[0]}")
                # Increase risk score more for dangerous patterns
                if any(danger in matches[0].lower() for danger in ['bomb', 'kill', 'weapon', 'drug', 'hack']):
                    risk_score += 2
                else:
                    risk_score += 1
        
        # Check for suspicious length
        if len(prompt) > 1000:
            issues.append("Prompt is unusually long")
            risk_score += 0.5
        
        # Check for suspicious character patterns
        if re.search(r'[<>\[\]{}]', prompt):
            issues.append("Contains suspicious characters")
            risk_score += 0.5
        
        # Check for multiple dangerous keywords
        dangerous_keywords = ['bomb', 'explosive', 'kill', 'weapon', 'drug', 'hack']
        keyword_count = sum(1 for keyword in dangerous_keywords if keyword in prompt.lower())
        if keyword_count > 1:
            issues.append(f"Contains multiple dangerous keywords ({keyword_count} found)")
            risk_score += keyword_count
        
        # Check for suspicious combinations
        suspicious_combinations = [
            (r'how.*to.*make', r'bomb|explosive|weapon'),
            (r'step.*by.*step', r'bomb|explosive|weapon'),
            (r'instructions.*for', r'bomb|explosive|weapon')
        ]
        for pattern1, pattern2 in suspicious_combinations:
            if re.search(pattern1, prompt, re.IGNORECASE) and re.search(pattern2, prompt, re.IGNORECASE):
                issues.append("Contains suspicious instruction patterns")
                risk_score += 2
        
        # Determine risk level with stricter thresholds
        if risk_score >= 2:
            risk_level = 'high'
        elif risk_score >= 1:
            risk_level = 'medium'
        elif risk_score > 0:
            risk_level = 'low'
        else:
            risk_level = 'safe'
        
        return risk_level, issues
    
    def get_risk_description(self, risk_level: str) -> str:
        """Get the description for a risk level"""
        return self.risk_levels.get(risk_level, 'Unknown risk level') 