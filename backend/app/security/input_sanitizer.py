"""
Comprehensive Input Sanitization and Validation System.
Prevents injection attacks, validates data types, and ensures secure data processing
for the automata-repo application with focus on educational content security.
"""

import re
import html
import json
import logging
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import bleach
from markupsafe import Markup
import unicodedata
import base64
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class SanitizationLevel(Enum):
    """Levels of sanitization strictness."""
    BASIC = "basic"          # Basic XSS prevention
    STRICT = "strict"        # Strict filtering for educational content
    PARANOID = "paranoid"    # Maximum security for system operations


class InputType(Enum):
    """Types of input content for specialized handling."""
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    SQL = "sql"
    JFLAP_XML = "jflap_xml"
    MATHEMATICAL = "mathematical"
    CODE = "code"
    FILENAME = "filename"
    URL = "url"
    EMAIL = "email"


@dataclass
class SanitizationResult:
    """Result of input sanitization."""
    original: str
    sanitized: str
    is_safe: bool
    threats_detected: List[str]
    sanitization_applied: List[str]
    confidence_score: float
    metadata: Dict[str, Any]


class InputSanitizer:
    """Comprehensive input sanitization with educational content focus."""
    
    def __init__(self, default_level: SanitizationLevel = SanitizationLevel.STRICT):
        self.default_level = default_level
        
        # XSS patterns - comprehensive list for educational platforms
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>',
            r'<embed[^>]*>.*?</embed>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
            r'<style[^>]*>.*?</style>',
            r'vbscript:',
            r'data:text/html',
            r'<svg[^>]*>.*?</svg>',
            r'<math[^>]*>.*?</math>',
            r'expression\s*\(',
            r'@import',
            r'binding\s*:',
            r'-moz-binding',
            r'behavior\s*:',
        ]
        
        # SQL injection patterns
        self.sql_patterns = [
            r'(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)',
            r'(\b(or|and)\s+[\'"]?\w+[\'"]?\s*=\s*[\'"]?\w+[\'"]?)',
            r'(\b(or|and)\s+\d+\s*=\s*\d+)',
            r'([\'"]\s*;\s*)',
            r'(\b(declare|cast|char|nchar)\b)',
            r'(--|\#|/\*|\*/)',
            r'(xp_|sp_cmdshell)',
            r'(\bwaitfor\s+delay\b)'
        ]
        
        # NoSQL injection patterns
        self.nosql_patterns = [
            r'\$where\s*:',
            r'\$ne\s*:',
            r'\$gt\s*:',
            r'\$lt\s*:',
            r'\$regex\s*:',
            r'\$in\s*:',
            r'\$nin\s*:',
            r'this\.',
            r'function\s*\('
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'%2e%2e%2f',
            r'%2e%2e%5c',
            r'..%2f',
            r'..%5c',
            r'%252e%252e%252f'
        ]
        
        # JFLAP-specific potentially dangerous patterns
        self.jflap_dangerous_patterns = [
            r'<!ENTITY[^>]*>',  # XML entities
            r'<!DOCTYPE[^>]*>',  # DOCTYPE declarations
            r'SYSTEM\s+["\']',   # External system references
            r'PUBLIC\s+["\']',   # Public DTD references
            r'<\?xml[^>]*\?>.*<!',  # XML with DTD
        ]
        
        # Mathematical expression dangerous patterns
        self.math_dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'subprocess',
            r'os\.',
            r'system\(',
            r'popen\(',
            r'file\(',
            r'open\('
        ]
        
        # Allowed HTML tags for educational content
        self.allowed_html_tags = {
            'p', 'br', 'strong', 'em', 'u', 'i', 'b',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'ul', 'ol', 'li', 'dl', 'dt', 'dd',
            'table', 'thead', 'tbody', 'tr', 'td', 'th',
            'code', 'pre', 'kbd', 'samp', 'var',
            'sub', 'sup', 'small', 'big',
            'blockquote', 'cite', 'q',
            'div', 'span', 'section', 'article',
            'mark', 'del', 'ins'
        }
        
        # Allowed HTML attributes
        self.allowed_html_attributes = {
            '*': ['class', 'id'],
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title', 'width', 'height'],
            'table': ['cellpadding', 'cellspacing', 'border'],
            'td': ['colspan', 'rowspan'],
            'th': ['colspan', 'rowspan', 'scope'],
        }
        
        # URL schemes whitelist
        self.allowed_url_schemes = {'http', 'https', 'ftp', 'mailto', 'tel'}
        
        # File extension whitelist for educational content
        self.allowed_file_extensions = {
            # Documents
            '.txt', '.pdf', '.doc', '.docx', '.odt',
            # Images
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp', '.webp',
            # JFLAP and automata files
            '.jff', '.jflap', '.xml',
            # Code files
            '.py', '.js', '.html', '.css', '.json', '.md',
            # Archive files
            '.zip', '.tar', '.gz'
        }
    
    def sanitize(
        self,
        input_data: Any,
        input_type: InputType = InputType.TEXT,
        level: Optional[SanitizationLevel] = None,
        context: Dict[str, Any] = None
    ) -> SanitizationResult:
        """
        Main sanitization method with comprehensive threat detection.
        """
        level = level or self.default_level
        context = context or {}
        
        if input_data is None:
            return SanitizationResult(
                original="",
                sanitized="",
                is_safe=True,
                threats_detected=[],
                sanitization_applied=[],
                confidence_score=1.0,
                metadata={"type": input_type.value}
            )
        
        # Convert to string if needed
        original_input = str(input_data)
        
        # Track sanitization process
        threats_detected = []
        sanitization_applied = []
        
        try:
            # Unicode normalization to prevent bypass attempts
            normalized_input = unicodedata.normalize('NFKC', original_input)
            if normalized_input != original_input:
                sanitization_applied.append("unicode_normalization")
            
            # Decode common encodings that might hide malicious content
            decoded_input = self._decode_common_encodings(normalized_input)
            if decoded_input != normalized_input:
                sanitization_applied.append("encoding_decode")
                # Check for threats in decoded content
                decoded_threats = self._detect_threats(decoded_input, input_type)
                threats_detected.extend(decoded_threats)
            
            # Primary threat detection
            primary_threats = self._detect_threats(original_input, input_type)
            threats_detected.extend(primary_threats)
            
            # Apply sanitization based on input type
            sanitized_input = self._sanitize_by_type(
                decoded_input, input_type, level, threats_detected, sanitization_applied
            )
            
            # Final safety check
            final_threats = self._detect_threats(sanitized_input, input_type)
            if final_threats:
                threats_detected.extend(final_threats)
                # If threats still exist, apply most restrictive sanitization
                sanitized_input = self._apply_emergency_sanitization(sanitized_input)
                sanitization_applied.append("emergency_sanitization")
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                threats_detected, sanitization_applied, input_type
            )
            
            is_safe = len(threats_detected) == 0 and confidence_score >= 0.8
            
            return SanitizationResult(
                original=original_input,
                sanitized=sanitized_input,
                is_safe=is_safe,
                threats_detected=list(set(threats_detected)),  # Remove duplicates
                sanitization_applied=sanitization_applied,
                confidence_score=confidence_score,
                metadata={
                    "type": input_type.value,
                    "level": level.value,
                    "context": context,
                    "timestamp": datetime.now().isoformat(),
                    "input_length": len(original_input),
                    "output_length": len(sanitized_input)
                }
            )
            
        except Exception as e:
            logger.error(f"Error during sanitization: {e}")
            return SanitizationResult(
                original=original_input,
                sanitized="",
                is_safe=False,
                threats_detected=["sanitization_error"],
                sanitization_applied=["error_handling"],
                confidence_score=0.0,
                metadata={"error": str(e), "type": input_type.value}
            )
    
    def _decode_common_encodings(self, input_str: str) -> str:
        """Decode common encodings to reveal hidden content."""
        decoded = input_str
        
        try:
            # URL decoding (multiple passes to catch double encoding)
            for _ in range(3):
                url_decoded = urllib.parse.unquote_plus(decoded)
                if url_decoded == decoded:
                    break
                decoded = url_decoded
            
            # HTML entity decoding
            html_decoded = html.unescape(decoded)
            if html_decoded != decoded:
                decoded = html_decoded
            
            # Base64 decoding (if it looks like base64)
            if self._looks_like_base64(decoded):
                try:
                    base64_decoded = base64.b64decode(decoded).decode('utf-8', errors='ignore')
                    if base64_decoded and len(base64_decoded) < len(decoded) * 2:
                        decoded = base64_decoded
                except Exception:
                    pass  # Not valid base64, continue
            
        except Exception as e:
            logger.warning(f"Error during encoding detection: {e}")
        
        return decoded
    
    def _looks_like_base64(self, s: str) -> bool:
        """Check if string looks like base64."""
        if len(s) < 4 or len(s) % 4 != 0:
            return False
        
        base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
        return bool(base64_pattern.match(s))
    
    def _detect_threats(self, input_str: str, input_type: InputType) -> List[str]:
        """Comprehensive threat detection."""
        threats = []
        input_lower = input_str.lower()
        
        # XSS detection
        for pattern in self.xss_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                threats.append("xss_attempt")
                break
        
        # SQL injection detection
        for pattern in self.sql_patterns:
            if re.search(pattern, input_lower, re.IGNORECASE):
                threats.append("sql_injection")
                break
        
        # NoSQL injection detection
        for pattern in self.nosql_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                threats.append("nosql_injection")
                break
        
        # Path traversal detection
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                threats.append("path_traversal")
                break
        
        # JFLAP-specific threats
        if input_type == InputType.JFLAP_XML:
            for pattern in self.jflap_dangerous_patterns:
                if re.search(pattern, input_str, re.IGNORECASE):
                    threats.append("xml_entity_injection")
                    break
        
        # Mathematical expression threats
        if input_type == InputType.MATHEMATICAL:
            for pattern in self.math_dangerous_patterns:
                if re.search(pattern, input_str, re.IGNORECASE):
                    threats.append("code_execution_attempt")
                    break
        
        # Generic dangerous patterns
        dangerous_patterns = [
            r'<\?php',  # PHP code
            r'<%.*%>',  # ASP code
            r'\${.*}',  # Template injection
            r'{{.*}}',  # Template injection
            r'eval\s*\(',  # Code evaluation
            r'setTimeout\s*\(',  # JavaScript timing
            r'setInterval\s*\(',  # JavaScript timing
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                threats.append("code_injection")
                break
        
        # Check for suspicious Unicode characters
        if self._contains_suspicious_unicode(input_str):
            threats.append("suspicious_unicode")
        
        # Check for excessive length (potential DoS)
        if len(input_str) > 1000000:  # 1MB
            threats.append("excessive_length")
        
        return threats
    
    def _contains_suspicious_unicode(self, input_str: str) -> bool:
        """Check for suspicious Unicode characters that could be used for bypasses."""
        suspicious_ranges = [
            (0x200B, 0x200D),  # Zero-width characters
            (0x2060, 0x2064),  # Word joiner and invisible characters
            (0xFEFF, 0xFEFF),  # Byte order mark
            (0x202A, 0x202E),  # Directional formatting characters
        ]
        
        for char in input_str:
            code_point = ord(char)
            for start, end in suspicious_ranges:
                if start <= code_point <= end:
                    return True
        
        return False
    
    def _sanitize_by_type(
        self,
        input_str: str,
        input_type: InputType,
        level: SanitizationLevel,
        threats: List[str],
        applied: List[str]
    ) -> str:
        """Apply type-specific sanitization."""
        
        if input_type == InputType.HTML:
            return self._sanitize_html(input_str, level, applied)
        elif input_type == InputType.MARKDOWN:
            return self._sanitize_markdown(input_str, level, applied)
        elif input_type == InputType.JSON:
            return self._sanitize_json(input_str, applied)
        elif input_type == InputType.JFLAP_XML:
            return self._sanitize_jflap_xml(input_str, applied)
        elif input_type == InputType.MATHEMATICAL:
            return self._sanitize_mathematical(input_str, applied)
        elif input_type == InputType.CODE:
            return self._sanitize_code(input_str, level, applied)
        elif input_type == InputType.FILENAME:
            return self._sanitize_filename(input_str, applied)
        elif input_type == InputType.URL:
            return self._sanitize_url(input_str, applied)
        elif input_type == InputType.EMAIL:
            return self._sanitize_email(input_str, applied)
        else:  # TEXT and default
            return self._sanitize_text(input_str, level, applied)
    
    def _sanitize_html(
        self, input_str: str, level: SanitizationLevel, applied: List[str]
    ) -> str:
        """Sanitize HTML content for educational use."""
        if level == SanitizationLevel.PARANOID:
            # Strip all HTML
            sanitized = bleach.clean(input_str, tags=[], attributes={}, strip=True)
            applied.append("html_strip_all")
        else:
            # Allow safe HTML tags for educational content
            sanitized = bleach.clean(
                input_str,
                tags=self.allowed_html_tags,
                attributes=self.allowed_html_attributes,
                strip=True
            )
            applied.append("html_whitelist")
        
        return sanitized
    
    def _sanitize_markdown(
        self, input_str: str, level: SanitizationLevel, applied: List[str]
    ) -> str:
        """Sanitize Markdown content."""
        # First convert markdown to HTML, then sanitize HTML
        try:
            import markdown
            html_content = markdown.markdown(input_str)
            sanitized = self._sanitize_html(html_content, level, applied)
            applied.append("markdown_to_html")
            return sanitized
        except ImportError:
            # If markdown library not available, treat as text
            return self._sanitize_text(input_str, level, applied)
    
    def _sanitize_json(self, input_str: str, applied: List[str]) -> str:
        """Sanitize and validate JSON input."""
        try:
            # Parse and re-serialize to ensure valid JSON
            parsed = json.loads(input_str)
            sanitized = json.dumps(parsed, ensure_ascii=True, separators=(',', ':'))
            applied.append("json_reserialize")
            return sanitized
        except json.JSONDecodeError:
            # If invalid JSON, return empty object
            applied.append("json_invalid_replaced")
            return "{}"
    
    def _sanitize_jflap_xml(self, input_str: str, applied: List[str]) -> str:
        """Sanitize JFLAP XML files with educational focus."""
        # Remove XML declarations and DTDs
        sanitized = re.sub(r'<\?xml[^>]*\?>', '', input_str)
        sanitized = re.sub(r'<!DOCTYPE[^>]*>', '', sanitized)
        sanitized = re.sub(r'<!ENTITY[^>]*>', '', sanitized)
        
        # Remove SYSTEM and PUBLIC references
        sanitized = re.sub(r'SYSTEM\s+["\'][^"\']*["\']', '', sanitized)
        sanitized = re.sub(r'PUBLIC\s+["\'][^"\']*["\']', '', sanitized)
        
        # Validate basic XML structure
        if '<structure>' not in sanitized or '</structure>' not in sanitized:
            applied.append("jflap_structure_validation_failed")
            return "<structure></structure>"  # Return minimal valid structure
        
        applied.append("jflap_xml_sanitized")
        return sanitized
    
    def _sanitize_mathematical(self, input_str: str, applied: List[str]) -> str:
        """Sanitize mathematical expressions."""
        # Allow only mathematical characters and basic syntax
        allowed_chars = set('0123456789+-*/^()[]{}.,= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')
        mathematical_functions = {'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'abs', 'floor', 'ceil'}
        
        # Remove any function calls except allowed mathematical ones
        words = re.findall(r'\b\w+\b', input_str)
        for word in words:
            if word not in mathematical_functions and not word.isdigit():
                if len(word) > 10 or any(c not in allowed_chars for c in word):
                    input_str = input_str.replace(word, '')
                    applied.append("math_function_removed")
        
        # Filter characters
        sanitized = ''.join(c for c in input_str if c in allowed_chars)
        if sanitized != input_str:
            applied.append("math_char_filter")
        
        return sanitized
    
    def _sanitize_code(
        self, input_str: str, level: SanitizationLevel, applied: List[str]
    ) -> str:
        """Sanitize code input for educational purposes."""
        if level == SanitizationLevel.PARANOID:
            # Remove potentially dangerous keywords
            dangerous_keywords = [
                'eval', 'exec', 'import', 'subprocess', 'os.system',
                'open', 'file', 'input', 'raw_input', '__import__'
            ]
            
            sanitized = input_str
            for keyword in dangerous_keywords:
                if keyword in sanitized:
                    sanitized = sanitized.replace(keyword, f"REMOVED_{keyword.upper()}")
                    applied.append("code_keyword_removed")
            
            return sanitized
        else:
            # Basic sanitization - just escape HTML
            return html.escape(input_str)
    
    def _sanitize_filename(self, input_str: str, applied: List[str]) -> str:
        """Sanitize filename input."""
        # Remove path separators and dangerous characters
        dangerous_chars = '<>:"|?*\\/'
        sanitized = ''.join(c for c in input_str if c not in dangerous_chars)
        
        # Check file extension
        if '.' in sanitized:
            name, ext = sanitized.rsplit('.', 1)
            if f'.{ext.lower()}' not in self.allowed_file_extensions:
                sanitized = name  # Remove extension if not allowed
                applied.append("filename_extension_removed")
        
        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
            applied.append("filename_length_limited")
        
        applied.append("filename_sanitized")
        return sanitized
    
    def _sanitize_url(self, input_str: str, applied: List[str]) -> str:
        """Sanitize URL input."""
        try:
            from urllib.parse import urlparse, urlunparse
            
            parsed = urlparse(input_str)
            
            # Check scheme
            if parsed.scheme.lower() not in self.allowed_url_schemes:
                applied.append("url_scheme_rejected")
                return ""
            
            # Reconstruct clean URL
            clean_url = urlunparse((
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
            
            applied.append("url_sanitized")
            return clean_url
            
        except Exception:
            applied.append("url_parse_error")
            return ""
    
    def _sanitize_email(self, input_str: str, applied: List[str]) -> str:
        """Sanitize email address."""
        # Basic email pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if re.match(email_pattern, input_str):
            applied.append("email_validated")
            return input_str.lower()
        else:
            applied.append("email_invalid")
            return ""
    
    def _sanitize_text(
        self, input_str: str, level: SanitizationLevel, applied: List[str]
    ) -> str:
        """Sanitize plain text."""
        if level == SanitizationLevel.PARANOID:
            # Very strict - only allow alphanumeric and basic punctuation
            allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-_()[]{}:;\'"')
            sanitized = ''.join(c for c in input_str if c in allowed_chars)
            applied.append("text_char_whitelist")
        else:
            # Escape HTML and remove null bytes
            sanitized = html.escape(input_str.replace('\x00', ''))
            applied.append("text_html_escape")
        
        return sanitized
    
    def _apply_emergency_sanitization(self, input_str: str) -> str:
        """Apply emergency sanitization when threats are still detected."""
        # Most restrictive sanitization
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-')
        return ''.join(c for c in input_str if c in allowed_chars)[:1000]  # Limit length
    
    def _calculate_confidence_score(
        self,
        threats_detected: List[str],
        sanitization_applied: List[str],
        input_type: InputType
    ) -> float:
        """Calculate confidence score for sanitization result."""
        base_score = 1.0
        
        # Reduce score for each threat detected
        threat_penalty = {
            'xss_attempt': 0.4,
            'sql_injection': 0.5,
            'nosql_injection': 0.3,
            'path_traversal': 0.3,
            'xml_entity_injection': 0.4,
            'code_execution_attempt': 0.6,
            'code_injection': 0.5,
            'suspicious_unicode': 0.1,
            'excessive_length': 0.2
        }
        
        for threat in threats_detected:
            penalty = threat_penalty.get(threat, 0.2)
            base_score -= penalty
        
        # Bonus for successful sanitization
        if sanitization_applied:
            base_score += 0.1  # Bonus for applying sanitization
        
        return max(0.0, min(1.0, base_score))
    
    def batch_sanitize(
        self,
        inputs: List[Tuple[Any, InputType]],
        level: Optional[SanitizationLevel] = None
    ) -> List[SanitizationResult]:
        """Sanitize multiple inputs efficiently."""
        results = []
        
        for input_data, input_type in inputs:
            result = self.sanitize(input_data, input_type, level)
            results.append(result)
        
        return results
    
    def is_safe_for_storage(self, sanitization_result: SanitizationResult) -> bool:
        """Check if sanitization result is safe for database storage."""
        return (
            sanitization_result.is_safe and
            sanitization_result.confidence_score >= 0.8 and
            len(sanitization_result.threats_detected) == 0
        )
    
    def get_sanitization_report(
        self, results: List[SanitizationResult]
    ) -> Dict[str, Any]:
        """Generate a comprehensive sanitization report."""
        total_inputs = len(results)
        safe_inputs = sum(1 for r in results if r.is_safe)
        
        threat_counts = {}
        for result in results:
            for threat in result.threats_detected:
                threat_counts[threat] = threat_counts.get(threat, 0) + 1
        
        avg_confidence = sum(r.confidence_score for r in results) / total_inputs if total_inputs > 0 else 0
        
        return {
            "total_inputs": total_inputs,
            "safe_inputs": safe_inputs,
            "unsafe_inputs": total_inputs - safe_inputs,
            "safety_rate": safe_inputs / total_inputs if total_inputs > 0 else 0,
            "average_confidence": avg_confidence,
            "threat_breakdown": threat_counts,
            "timestamp": datetime.now().isoformat()
        }


# Global sanitizer instance
default_sanitizer = InputSanitizer()


# Convenience functions
def sanitize_text(text: str, level: SanitizationLevel = SanitizationLevel.STRICT) -> str:
    """Quick text sanitization."""
    result = default_sanitizer.sanitize(text, InputType.TEXT, level)
    return result.sanitized


def sanitize_html(html_content: str, level: SanitizationLevel = SanitizationLevel.STRICT) -> str:
    """Quick HTML sanitization."""
    result = default_sanitizer.sanitize(html_content, InputType.HTML, level)
    return result.sanitized


def sanitize_jflap_xml(xml_content: str) -> str:
    """Quick JFLAP XML sanitization."""
    result = default_sanitizer.sanitize(xml_content, InputType.JFLAP_XML)
    return result.sanitized


def is_safe_input(
    input_data: Any,
    input_type: InputType = InputType.TEXT,
    level: SanitizationLevel = SanitizationLevel.STRICT
) -> bool:
    """Quick safety check for input."""
    result = default_sanitizer.sanitize(input_data, input_type, level)
    return result.is_safe


def validate_user_input(
    input_data: Any,
    input_type: InputType = InputType.TEXT,
    max_length: int = 10000
) -> Tuple[bool, str, List[str]]:
    """Validate user input and return validation result."""
    if len(str(input_data)) > max_length:
        return False, "", ["input_too_long"]
    
    result = default_sanitizer.sanitize(input_data, input_type)
    return result.is_safe, result.sanitized, result.threats_detected