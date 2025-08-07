"""
OLLAMA VALIDATOR - AI-Powered Input Validation and Threat Detection
================================================================

This module uses Ollama's intelligence for comprehensive input validation:
- Semantic understanding of user intent vs malicious behavior
- Context-aware validation beyond simple pattern matching
- Threat detection using LLM reasoning
- Content filtering with natural language understanding
- Real-time security analysis of all inputs
- Adaptive learning from validation patterns
- Multi-layered validation approach
"""

import asyncio
import json
import logging
import re
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import ipaddress
import urllib.parse
from pathlib import Path

from .ollama_everything import ollama_everything, OllamaTask, OllamaTaskType, OllamaResult
from .valkey_integration import valkey_connection_manager
from .config import settings

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat levels for security analysis."""
    SAFE = "safe"
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationType(str, Enum):
    """Types of validation performed."""
    INJECTION_ATTACK = "injection_attack"
    XSS_ATTACK = "xss_attack"
    MALICIOUS_CONTENT = "malicious_content"
    SPAM_DETECTION = "spam_detection"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    SOCIAL_ENGINEERING = "social_engineering"
    DATA_EXFILTRATION = "data_exfiltration"
    RECONNAISSANCE = "reconnaissance"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SEMANTIC_VALIDATION = "semantic_validation"
    BUSINESS_LOGIC_VALIDATION = "business_logic_validation"
    CONTEXT_VALIDATION = "context_validation"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    threat_level: ThreatLevel
    threat_types: List[ValidationType]
    confidence_score: float
    reasoning: str
    sanitized_input: Optional[str] = None
    blocked_elements: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class ValidationContext:
    """Context for validation including user and request information."""
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    endpoint: Optional[str] = None
    request_method: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    request_history: List[Dict] = field(default_factory=list)
    business_context: Dict[str, Any] = field(default_factory=dict)


class OllamaValidator:
    """AI-powered input validator using Ollama for intelligent threat detection."""
    
    def __init__(self):
        self.validation_stats = defaultdict(int)
        self.threat_patterns = {}
        self.whitelist_patterns = {}
        self.learning_cache = deque(maxlen=10000)
        
        # Load validation patterns
        self._initialize_validation_patterns()
        
        # Initialize threat intelligence
        self.threat_intelligence = {
            "known_attack_patterns": [],
            "suspicious_keywords": [],
            "social_engineering_indicators": [],
            "data_exfiltration_patterns": []
        }
        
        logger.info("OllamaValidator initialized with AI-powered threat detection")
    
    def _initialize_validation_patterns(self):
        """Initialize comprehensive validation patterns."""
        
        # SQL Injection patterns (for initial screening)
        self.threat_patterns["sql_injection"] = [
            r"(\b(union|select|insert|delete|update|drop|create|alter|exec|execute)\b)",
            r"(--|#|/\*|\*/)",
            r"(\bor\s+1\s*=\s*1\b|\band\s+1\s*=\s*1\b)",
            r"(\bchar\s*\(|\bcast\s*\(|\bconvert\s*\()",
            r"(\bhex\s*\(|\bunhex\s*\(|\bload_file\s*\()"
        ]
        
        # XSS patterns
        self.threat_patterns["xss"] = [
            r"(<script[^>]*>.*?</script>)",
            r"(javascript:)",
            r"(on\w+\s*=)",
            r"(<iframe[^>]*>.*?</iframe>)",
            r"(<object[^>]*>.*?</object>)"
        ]
        
        # Command injection patterns
        self.threat_patterns["command_injection"] = [
            r"(\||&|;|\$\(|\`|<|>)",
            r"(\b(cat|ls|dir|type|net|ping|curl|wget|nc|netcat)\b)",
            r"(\.\.\/|\.\.\\)",
            r"(/bin/|/usr/bin/|cmd\.exe|powershell)"
        ]
        
        # Path traversal patterns
        self.threat_patterns["path_traversal"] = [
            r"(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
            r"(/etc/passwd|/etc/shadow|web\.config|boot\.ini)",
            r"(%00|%20|%0a|%0d)"
        ]
    
    async def validate_input(
        self,
        input_data: Union[str, Dict, List],
        validation_context: ValidationContext,
        validation_types: List[ValidationType] = None
    ) -> ValidationResult:
        """
        Perform comprehensive AI-powered validation of input data.
        
        Args:
            input_data: The input to validate
            validation_context: Context information for validation
            validation_types: Specific types of validation to perform
            
        Returns:
            ValidationResult with detailed analysis
        """
        start_time = time.time()
        
        # Convert input to string for analysis
        input_str = self._normalize_input(input_data)
        
        # Quick pattern-based pre-screening
        quick_threats = await self._quick_threat_screening(input_str)
        
        # If immediate threats detected, skip AI analysis for performance
        if quick_threats and any(threat.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL] for threat in quick_threats):
            logger.warning(f"Critical threat detected in quick screening: {input_str[:100]}")
            return ValidationResult(
                is_valid=False,
                threat_level=ThreatLevel.CRITICAL,
                threat_types=[ValidationType.INJECTION_ATTACK],
                confidence_score=0.95,
                reasoning="Critical threat patterns detected",
                processing_time=time.time() - start_time
            )
        
        # Perform AI-powered deep validation
        ai_validation = await self._ai_powered_validation(input_str, validation_context, validation_types)
        
        # Combine results and make final decision
        final_result = await self._combine_validation_results(quick_threats, ai_validation, validation_context)
        final_result.processing_time = time.time() - start_time
        
        # Learn from the validation
        await self._learn_from_validation(input_str, final_result, validation_context)
        
        # Update statistics
        self._update_validation_stats(final_result)
        
        return final_result
    
    def _normalize_input(self, input_data: Union[str, Dict, List]) -> str:
        """Normalize input data to string for analysis."""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, (dict, list)):
            return json.dumps(input_data, separators=(',', ':'))
        else:
            return str(input_data)
    
    async def _quick_threat_screening(self, input_str: str) -> List[ValidationResult]:
        """Perform quick pattern-based threat screening."""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, input_str, re.IGNORECASE)
                for match in matches:
                    threat_level = self._assess_pattern_threat_level(threat_type, match.group())
                    
                    if threat_level != ThreatLevel.SAFE:
                        threats.append(ValidationResult(
                            is_valid=False,
                            threat_level=threat_level,
                            threat_types=[ValidationType.INJECTION_ATTACK],
                            confidence_score=0.8,
                            reasoning=f"Pattern match: {threat_type} - {match.group()}",
                            blocked_elements=[match.group()]
                        ))
        
        return threats
    
    def _assess_pattern_threat_level(self, threat_type: str, matched_text: str) -> ThreatLevel:
        """Assess threat level based on pattern type and matched text."""
        high_risk_indicators = {
            "sql_injection": ["union", "drop", "delete", "exec", "xp_"],
            "xss": ["<script", "javascript:", "onerror="],
            "command_injection": ["|", "&", "$(", "`", "/bin/", "cmd.exe"],
            "path_traversal": ["../../../", "/etc/passwd", "/etc/shadow"]
        }
        
        if threat_type in high_risk_indicators:
            for indicator in high_risk_indicators[threat_type]:
                if indicator.lower() in matched_text.lower():
                    return ThreatLevel.HIGH
        
        return ThreatLevel.MEDIUM
    
    async def _ai_powered_validation(
        self,
        input_str: str,
        context: ValidationContext,
        validation_types: List[ValidationType] = None
    ) -> ValidationResult:
        """Perform AI-powered deep validation using Ollama."""
        
        # Build comprehensive prompt for AI validation
        validation_prompt = self._build_validation_prompt(input_str, context, validation_types)
        
        # Create validation task
        task = OllamaTask(
            task_type=OllamaTaskType.SECURITY_ANALYSIS,
            input_data=validation_prompt,
            context={
                "validation_focus": "comprehensive_threat_detection",
                "user_context": context.__dict__ if context else {},
                "input_sample": input_str[:500]  # First 500 chars for context
            },
            temperature=0.2,  # Low temperature for consistent security analysis
            max_tokens=1500
        )
        
        try:
            result = await ollama_everything.process_task(task)
            
            if result.error:
                logger.error(f"AI validation failed: {result.error}")
                return self._create_fallback_validation_result(input_str)
            
            return await self._parse_ai_validation_response(result, input_str)
            
        except Exception as e:
            logger.error(f"AI validation exception: {e}")
            return self._create_fallback_validation_result(input_str)
    
    def _build_validation_prompt(
        self,
        input_str: str,
        context: ValidationContext,
        validation_types: List[ValidationType] = None
    ) -> str:
        """Build comprehensive validation prompt for AI analysis."""
        
        prompt = f"""
SECURITY VALIDATION ANALYSIS
===========================

Analyze the following input for security threats and validation issues:

INPUT TO ANALYZE:
{input_str[:2000]}  # Limit input size for analysis

CONTEXT INFORMATION:
- User Role: {context.user_role if context else 'Unknown'}
- Endpoint: {context.endpoint if context else 'Unknown'}
- Request Method: {context.request_method if context else 'Unknown'}
- IP Address: {context.ip_address if context else 'Unknown'}
- Business Context: {json.dumps(context.business_context, indent=2) if context and context.business_context else 'None'}

VALIDATION REQUIREMENTS:
Analyze for ALL of the following security threats:

1. INJECTION ATTACKS:
   - SQL injection attempts
   - NoSQL injection
   - LDAP injection
   - Command injection
   - Code injection

2. CROSS-SITE SCRIPTING (XSS):
   - Stored XSS payloads
   - Reflected XSS attempts
   - DOM-based XSS
   - Script injection

3. MALICIOUS CONTENT:
   - Malware signatures
   - Phishing attempts
   - Social engineering
   - Suspicious URLs/domains

4. DATA EXFILTRATION:
   - Attempts to access sensitive data
   - Information gathering
   - Reconnaissance activities

5. BUSINESS LOGIC VALIDATION:
   - Inappropriate content for context
   - Out-of-bounds values
   - Invalid state transitions

6. SEMANTIC ANALYSIS:
   - Intent vs stated purpose
   - Contextual appropriateness
   - Logical consistency

ANALYSIS REQUIREMENTS:
Provide your analysis in the following JSON format:

{{
  "threat_assessment": {{
    "is_malicious": boolean,
    "threat_level": "safe|low|medium|high|critical",
    "primary_threats": ["list of detected threat types"],
    "confidence_score": float (0.0 to 1.0)
  }},
  "detailed_analysis": {{
    "reasoning": "Detailed explanation of your analysis",
    "specific_threats": [
      {{
        "type": "threat_type",
        "description": "what was detected",
        "risk_level": "low|medium|high|critical",
        "evidence": "specific evidence found"
      }}
    ],
    "false_positive_likelihood": float (0.0 to 1.0),
    "context_considerations": "how context affects the assessment"
  }},
  "recommendations": {{
    "action": "allow|sanitize|block",
    "sanitization_needed": ["list of elements to sanitize"],
    "security_measures": ["recommended security measures"],
    "monitoring_flags": ["aspects to monitor"]
  }}
}}

CRITICAL INSTRUCTIONS:
- Be thorough but avoid false positives for legitimate use cases
- Consider the business context when assessing threats
- Explain your reasoning clearly
- If unsure, err on the side of caution
- Focus on intent and semantic meaning, not just patterns
"""
        
        return prompt
    
    async def _parse_ai_validation_response(self, ai_result: OllamaResult, input_str: str) -> ValidationResult:
        """Parse the AI validation response into a ValidationResult."""
        
        try:
            response_text = ai_result.result.get("response", str(ai_result.result))
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                # Fallback: parse structured response
                analysis_data = self._parse_structured_response(response_text)
            
            # Extract threat assessment
            threat_assessment = analysis_data.get("threat_assessment", {})
            detailed_analysis = analysis_data.get("detailed_analysis", {})
            recommendations = analysis_data.get("recommendations", {})
            
            # Determine threat level
            threat_level_str = threat_assessment.get("threat_level", "medium")
            threat_level = ThreatLevel(threat_level_str.lower())
            
            # Extract threat types
            primary_threats = threat_assessment.get("primary_threats", [])
            threat_types = []
            for threat in primary_threats:
                try:
                    # Map threat descriptions to ValidationType enums
                    threat_types.append(self._map_threat_to_validation_type(threat))
                except:
                    pass  # Skip unmappable threats
            
            # Build result
            is_valid = not threat_assessment.get("is_malicious", False)
            confidence_score = threat_assessment.get("confidence_score", ai_result.confidence_score)
            
            # Extract sanitization suggestions
            sanitized_input = None
            blocked_elements = []
            sanitization_needed = recommendations.get("sanitization_needed", [])
            
            if sanitization_needed:
                sanitized_input = await self._sanitize_input(input_str, sanitization_needed)
                blocked_elements = sanitization_needed
            
            return ValidationResult(
                is_valid=is_valid,
                threat_level=threat_level,
                threat_types=threat_types,
                confidence_score=confidence_score,
                reasoning=detailed_analysis.get("reasoning", "AI analysis completed"),
                sanitized_input=sanitized_input,
                blocked_elements=blocked_elements,
                recommendations=recommendations.get("security_measures", []),
                metadata={
                    "ai_analysis": analysis_data,
                    "model_used": ai_result.model_used,
                    "processing_time": ai_result.processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse AI validation response: {e}")
            logger.debug(f"Raw response: {ai_result.result}")
            
            # Fallback to text-based parsing
            return self._create_text_based_validation_result(ai_result, input_str)
    
    def _parse_structured_response(self, response_text: str) -> Dict[str, Any]:
        """Parse structured response when JSON parsing fails."""
        analysis = {
            "threat_assessment": {},
            "detailed_analysis": {},
            "recommendations": {}
        }
        
        # Extract key information using patterns
        threat_level_match = re.search(r'threat[_\s]*level[:\s]*([a-z]+)', response_text, re.IGNORECASE)
        if threat_level_match:
            analysis["threat_assessment"]["threat_level"] = threat_level_match.group(1).lower()
        
        malicious_match = re.search(r'(malicious|threat|dangerous|safe)', response_text, re.IGNORECASE)
        if malicious_match:
            malicious_term = malicious_match.group(1).lower()
            analysis["threat_assessment"]["is_malicious"] = malicious_term in ["malicious", "threat", "dangerous"]
        
        # Extract reasoning
        reasoning_patterns = [
            r'reasoning[:\s]*(.*?)(?:\n|$)',
            r'analysis[:\s]*(.*?)(?:\n|$)',
            r'explanation[:\s]*(.*?)(?:\n|$)'
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                analysis["detailed_analysis"]["reasoning"] = match.group(1).strip()
                break
        
        return analysis
    
    def _map_threat_to_validation_type(self, threat_description: str) -> ValidationType:
        """Map threat description to ValidationType enum."""
        threat_lower = threat_description.lower()
        
        if "injection" in threat_lower or "sql" in threat_lower:
            return ValidationType.INJECTION_ATTACK
        elif "xss" in threat_lower or "script" in threat_lower:
            return ValidationType.XSS_ATTACK
        elif "social" in threat_lower:
            return ValidationType.SOCIAL_ENGINEERING
        elif "spam" in threat_lower:
            return ValidationType.SPAM_DETECTION
        elif "inappropriate" in threat_lower:
            return ValidationType.INAPPROPRIATE_CONTENT
        elif "exfiltration" in threat_lower:
            return ValidationType.DATA_EXFILTRATION
        elif "reconnaissance" in threat_lower:
            return ValidationType.RECONNAISSANCE
        else:
            return ValidationType.MALICIOUS_CONTENT
    
    def _create_text_based_validation_result(self, ai_result: OllamaResult, input_str: str) -> ValidationResult:
        """Create validation result from text-based AI analysis."""
        response_text = str(ai_result.result).lower()
        
        # Determine threat level from keywords
        if any(word in response_text for word in ["critical", "severe", "dangerous"]):
            threat_level = ThreatLevel.CRITICAL
        elif any(word in response_text for word in ["high", "serious", "malicious"]):
            threat_level = ThreatLevel.HIGH
        elif any(word in response_text for word in ["medium", "moderate", "suspicious"]):
            threat_level = ThreatLevel.MEDIUM
        elif any(word in response_text for word in ["low", "minor"]):
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.SAFE
        
        is_valid = threat_level in [ThreatLevel.SAFE, ThreatLevel.LOW]
        
        return ValidationResult(
            is_valid=is_valid,
            threat_level=threat_level,
            threat_types=[ValidationType.MALICIOUS_CONTENT],
            confidence_score=ai_result.confidence_score * 0.8,  # Lower confidence for text parsing
            reasoning=str(ai_result.result),
            metadata={"parsing_method": "text_based"}
        )
    
    def _create_fallback_validation_result(self, input_str: str) -> ValidationResult:
        """Create fallback validation result when AI analysis fails."""
        # Use basic pattern matching as fallback
        threat_indicators = [
            "<script", "javascript:", "union select", "drop table",
            "../../../", "/etc/passwd", "cmd.exe", "powershell"
        ]
        
        has_threats = any(indicator in input_str.lower() for indicator in threat_indicators)
        
        return ValidationResult(
            is_valid=not has_threats,
            threat_level=ThreatLevel.MEDIUM if has_threats else ThreatLevel.SAFE,
            threat_types=[ValidationType.MALICIOUS_CONTENT] if has_threats else [],
            confidence_score=0.6,
            reasoning="Fallback pattern-based validation (AI analysis failed)",
            metadata={"validation_method": "fallback"}
        )
    
    async def _sanitize_input(self, input_str: str, sanitization_rules: List[str]) -> str:
        """Sanitize input based on AI recommendations."""
        sanitized = input_str
        
        # Apply common sanitization rules
        if "html_tags" in sanitization_rules:
            sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        if "script_tags" in sanitization_rules:
            sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        if "javascript" in sanitization_rules:
            sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        
        if "sql_keywords" in sanitization_rules:
            sql_keywords = ["union", "select", "insert", "delete", "update", "drop"]
            for keyword in sql_keywords:
                sanitized = re.sub(rf'\b{keyword}\b', f'[{keyword}]', sanitized, flags=re.IGNORECASE)
        
        # Use AI for advanced sanitization if needed
        if len(sanitization_rules) > 4:  # Complex sanitization needed
            sanitized = await self._ai_sanitize_input(sanitized, sanitization_rules)
        
        return sanitized
    
    async def _ai_sanitize_input(self, input_str: str, rules: List[str]) -> str:
        """Use AI to perform advanced input sanitization."""
        try:
            sanitization_task = OllamaTask(
                task_type=OllamaTaskType.TEXT_TRANSFORMATION,
                input_data=f"Sanitize this input according to these rules: {rules}\n\nInput: {input_str}",
                context={"sanitization_rules": rules},
                temperature=0.1,
                max_tokens=len(input_str) + 200
            )
            
            result = await ollama_everything.process_task(sanitization_task)
            if result.error:
                return input_str  # Return original if sanitization fails
            
            sanitized = result.result.get("response", str(result.result))
            
            # Extract sanitized content (look for patterns like "Sanitized: ...")
            sanitized_match = re.search(r'sanitized[:\s]*(.*)', sanitized, re.IGNORECASE | re.DOTALL)
            if sanitized_match:
                return sanitized_match.group(1).strip()
            
            return sanitized
            
        except Exception as e:
            logger.error(f"AI sanitization failed: {e}")
            return input_str
    
    async def _combine_validation_results(
        self,
        quick_results: List[ValidationResult],
        ai_result: ValidationResult,
        context: ValidationContext
    ) -> ValidationResult:
        """Combine quick screening and AI validation results."""
        
        # If quick screening found critical threats, prioritize them
        critical_quick_threats = [r for r in quick_results if r.threat_level == ThreatLevel.CRITICAL]
        if critical_quick_threats:
            return critical_quick_threats[0]
        
        # If AI found high/critical threats, use AI result
        if ai_result.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            return ai_result
        
        # Combine threat types from both analyses
        all_threat_types = set(ai_result.threat_types)
        for result in quick_results:
            all_threat_types.update(result.threat_types)
        
        # Use highest threat level found
        max_threat_level = ai_result.threat_level
        for result in quick_results:
            if result.threat_level.value > max_threat_level.value:
                max_threat_level = result.threat_level
        
        # Combine blocked elements
        blocked_elements = list(ai_result.blocked_elements)
        for result in quick_results:
            blocked_elements.extend(result.blocked_elements)
        
        # Use AI result as base but enhance with quick screening findings
        combined_result = ai_result
        combined_result.threat_level = max_threat_level
        combined_result.threat_types = list(all_threat_types)
        combined_result.blocked_elements = list(set(blocked_elements))
        combined_result.is_valid = max_threat_level in [ThreatLevel.SAFE, ThreatLevel.LOW]
        
        # Adjust confidence based on agreement between methods
        if quick_results and ai_result.threat_level == max_threat_level:
            combined_result.confidence_score = min(1.0, combined_result.confidence_score + 0.1)
        
        return combined_result
    
    async def _learn_from_validation(
        self,
        input_str: str,
        result: ValidationResult,
        context: ValidationContext
    ):
        """Learn from validation results to improve future performance."""
        
        learning_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_hash": hashlib.sha256(input_str.encode()).hexdigest()[:16],
            "threat_level": result.threat_level.value,
            "threat_types": [t.value for t in result.threat_types],
            "confidence": result.confidence_score,
            "context": {
                "endpoint": context.endpoint if context else None,
                "user_role": context.user_role if context else None,
                "ip_address": context.ip_address if context else None
            }
        }
        
        self.learning_cache.append(learning_record)
        
        # Periodically analyze patterns and update threat intelligence
        if len(self.learning_cache) % 100 == 0:
            await self._update_threat_intelligence()
    
    async def _update_threat_intelligence(self):
        """Update threat intelligence based on recent validations."""
        try:
            # Analyze recent validation patterns using AI
            recent_threats = [r for r in list(self.learning_cache)[-500:] 
                            if r["threat_level"] in ["medium", "high", "critical"]]
            
            if len(recent_threats) < 10:
                return
            
            # Ask AI to identify new patterns
            pattern_analysis_task = OllamaTask(
                task_type=OllamaTaskType.PATTERN_RECOGNITION,
                input_data=json.dumps(recent_threats, indent=2),
                context={
                    "analysis_type": "threat_pattern_discovery",
                    "focus": "emerging_attack_patterns"
                },
                temperature=0.3
            )
            
            result = await ollama_everything.process_task(pattern_analysis_task)
            
            if not result.error and result.confidence_score > 0.7:
                # Extract new patterns from AI analysis
                patterns = result.result.get("patterns", [])
                self.threat_intelligence["emerging_patterns"] = patterns
                
                logger.info(f"Updated threat intelligence with {len(patterns)} new patterns")
            
        except Exception as e:
            logger.error(f"Failed to update threat intelligence: {e}")
    
    def _update_validation_stats(self, result: ValidationResult):
        """Update validation statistics."""
        self.validation_stats["total_validations"] += 1
        self.validation_stats[f"threat_level_{result.threat_level.value}"] += 1
        
        if result.is_valid:
            self.validation_stats["valid_inputs"] += 1
        else:
            self.validation_stats["blocked_inputs"] += 1
        
        for threat_type in result.threat_types:
            self.validation_stats[f"threat_type_{threat_type.value}"] += 1
    
    async def validate_batch_inputs(
        self,
        inputs: List[Tuple[Any, ValidationContext]],
        parallel_processing: bool = True
    ) -> List[ValidationResult]:
        """Validate multiple inputs efficiently."""
        
        if parallel_processing:
            # Process inputs in parallel
            validation_tasks = [
                self.validate_input(input_data, context) 
                for input_data, context in inputs
            ]
            return await asyncio.gather(*validation_tasks)
        else:
            # Process sequentially
            results = []
            for input_data, context in inputs:
                result = await self.validate_input(input_data, context)
                results.append(result)
            return results
    
    async def validate_api_request(
        self,
        request_data: Dict[str, Any],
        endpoint: str,
        method: str,
        user_context: Dict[str, Any] = None
    ) -> Dict[str, ValidationResult]:
        """Validate all parameters in an API request."""
        
        context = ValidationContext(
            endpoint=endpoint,
            request_method=method,
            user_id=user_context.get("user_id") if user_context else None,
            user_role=user_context.get("role") if user_context else None,
            ip_address=user_context.get("ip_address") if user_context else None,
            business_context=user_context or {}
        )
        
        validation_results = {}
        
        # Validate each parameter
        for param_name, param_value in request_data.items():
            try:
                result = await self.validate_input(param_value, context)
                validation_results[param_name] = result
                
                # Log any threats detected
                if not result.is_valid:
                    logger.warning(
                        f"Threat detected in {endpoint} parameter '{param_name}': "
                        f"{result.threat_level.value} - {result.reasoning[:100]}"
                    )
                    
            except Exception as e:
                logger.error(f"Validation failed for parameter {param_name}: {e}")
                validation_results[param_name] = ValidationResult(
                    is_valid=False,
                    threat_level=ThreatLevel.MEDIUM,
                    threat_types=[ValidationType.MALICIOUS_CONTENT],
                    confidence_score=0.5,
                    reasoning=f"Validation error: {str(e)}"
                )
        
        return validation_results
    
    async def get_validation_stats(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        total_validations = self.validation_stats.get("total_validations", 1)
        
        stats = {
            "total_validations": total_validations,
            "blocked_rate": self.validation_stats.get("blocked_inputs", 0) / total_validations,
            "threat_level_distribution": {
                level: self.validation_stats.get(f"threat_level_{level}", 0) / total_validations
                for level in ["safe", "low", "medium", "high", "critical"]
            },
            "threat_type_distribution": {
                k.replace("threat_type_", ""): v / total_validations
                for k, v in self.validation_stats.items()
                if k.startswith("threat_type_")
            },
            "recent_learning_records": len(self.learning_cache),
            "threat_intelligence_patterns": len(self.threat_intelligence.get("emerging_patterns", []))
        }
        
        return stats
    
    async def shutdown(self):
        """Clean shutdown of the validator."""
        try:
            # Save learning data and threat intelligence
            await self._save_threat_intelligence()
            logger.info("OllamaValidator shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during validator shutdown: {e}")
    
    async def _save_threat_intelligence(self):
        """Save threat intelligence to persistent storage."""
        try:
            intelligence_data = {
                "threat_intelligence": self.threat_intelligence,
                "validation_stats": dict(self.validation_stats),
                "learning_records": list(self.learning_cache)[-1000:],  # Save last 1000 records
                "last_updated": datetime.utcnow().isoformat()
            }
            
            async with valkey_connection_manager.get_client() as client:
                await client.setex(
                    "ollama_validator_intelligence",
                    86400 * 7,  # 7 days
                    json.dumps(intelligence_data, default=str)
                )
                
            logger.info("Threat intelligence saved successfully")
            
        except Exception as e:
            logger.warning(f"Failed to save threat intelligence: {e}")


# Global validator instance
ollama_validator = OllamaValidator()


# Convenience functions
async def validate_input_safe(
    input_data: Any,
    context: ValidationContext = None
) -> ValidationResult:
    """Quick and safe input validation."""
    if context is None:
        context = ValidationContext()
    return await ollama_validator.validate_input(input_data, context)


async def validate_api_request_safe(
    request_data: Dict[str, Any],
    endpoint: str,
    method: str = "POST",
    user_context: Dict[str, Any] = None
) -> Dict[str, ValidationResult]:
    """Validate API request safely."""
    return await ollama_validator.validate_api_request(
        request_data, endpoint, method, user_context
    )


async def check_for_threats(input_text: str) -> bool:
    """Quick threat check - returns True if threats detected."""
    result = await validate_input_safe(input_text)
    return not result.is_valid


# Initialize and shutdown functions
async def initialize_ollama_validator():
    """Initialize the Ollama validator system."""
    try:
        # Load existing threat intelligence
        async with valkey_connection_manager.get_client() as client:
            intelligence_data = await client.get("ollama_validator_intelligence")
            
            if intelligence_data:
                data = json.loads(intelligence_data)
                ollama_validator.threat_intelligence = data.get("threat_intelligence", {})
                ollama_validator.validation_stats.update(data.get("validation_stats", {}))
                
                # Restore learning records
                learning_records = data.get("learning_records", [])
                ollama_validator.learning_cache.extend(learning_records)
                
                logger.info(f"Loaded threat intelligence with {len(learning_records)} learning records")
        
        # Test the validator
        test_result = await validate_input_safe("Hello, this is a test message")
        if test_result.error:
            raise Exception(f"Validator test failed: {test_result.error}")
        
        logger.info("OllamaValidator initialized and tested successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize OllamaValidator: {e}")
        raise


async def shutdown_ollama_validator():
    """Shutdown the Ollama validator system."""
    try:
        await ollama_validator.shutdown()
        logger.info("OllamaValidator shutdown completed")
        
    except Exception as e:
        logger.error(f"Error shutting down OllamaValidator: {e}")