#!/usr/bin/env python3
"""
Test script for the new LaTeX export, API platform, and automated grading features.
"""

import asyncio
import json
from datetime import datetime, timedelta

# Mock test data
sample_automaton = {
    "states": [
        {"id": "q0", "label": "q0"},
        {"id": "q1", "label": "q1"},
        {"id": "q2", "label": "q2"}
    ],
    "transitions": [
        {"from": "q0", "to": "q1", "symbol": "a"},
        {"from": "q1", "to": "q2", "symbol": "b"},
        {"from": "q2", "to": "q2", "symbol": "a,b"}
    ],
    "alphabet": ["a", "b"],
    "start_state": "q0",
    "accept_states": ["q2"]
}

sample_grammar = {
    "variables": ["S", "A", "B"],
    "terminals": ["a", "b"],
    "productions": [
        {"left": "S", "right": "aA | bB"},
        {"left": "A", "right": "aS | b"},
        {"left": "B", "right": "bS | a"}
    ],
    "start_variable": "S"
}

sample_proof = {
    "type": "theorem",
    "statement": "The language L = {a^n b^n | n ≥ 0} is context-free.",
    "steps": [
        {
            "text": "We construct a PDA that recognizes L.",
            "justification": "Construction"
        },
        {
            "text": "The PDA uses a stack to count a's and match them with b's.",
            "justification": "Stack operation"
        },
        {
            "text": "Therefore, L is context-free.",
            "justification": "Definition of context-free languages"
        }
    ]
}

async def test_latex_export():
    """Test LaTeX export functionality."""
    print("Testing LaTeX Export...")
    
    try:
        from app.latex_export import latex_exporter
        
        # Test automaton export
        print("  Testing automaton export...")
        automaton_latex = await latex_exporter.export_automaton(sample_automaton)
        assert "tikzpicture" in automaton_latex
        assert "q0" in automaton_latex
        print("    ✓ Automaton export works")
        
        # Test grammar export
        print("  Testing grammar export...")
        grammar_latex = await latex_exporter.export_grammar(sample_grammar)
        assert "align" in grammar_latex
        assert "S" in grammar_latex
        print("    ✓ Grammar export works")
        
        # Test proof export
        print("  Testing proof export...")
        proof_latex = await latex_exporter.export_proof(sample_proof)
        assert "theorem" in proof_latex
        assert "proof" in proof_latex
        print("    ✓ Proof export works")
        
        print("✓ LaTeX Export tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ LaTeX Export test failed: {e}")
        return False

def test_api_platform():
    """Test API Platform functionality."""
    print("Testing API Platform...")
    
    try:
        from app.api_platform import api_platform, ClientRegistrationRequest, APIKeyRequest
        
        # Test client registration request validation
        print("  Testing client registration...")
        client_request = ClientRegistrationRequest(
            name="Test Client",
            description="Test client for API platform",
            owner_email="test@example.com",
            scopes=["read:problems", "write:solutions"]
        )
        assert client_request.name == "Test Client"
        print("    ✓ Client registration request validation works")
        
        # Test API key request validation
        print("  Testing API key request...")
        key_request = APIKeyRequest(
            name="Test Key",
            scopes=["read:problems"],
            expires_in_days=30
        )
        assert key_request.name == "Test Key"
        print("    ✓ API key request validation works")
        
        print("✓ API Platform tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ API Platform test failed: {e}")
        return False

def test_automated_grading():
    """Test Automated Grading functionality."""
    print("Testing Automated Grading...")
    
    try:
        from app.automated_grading import (
            automated_grader, AssignmentRequest, SubmissionRequest,
            AssignmentType, GradingResult
        )
        
        # Test assignment request validation
        print("  Testing assignment request...")
        assignment_request = AssignmentRequest(
            title="Test Assignment",
            description="Test assignment for grading system",
            type=AssignmentType.HOMEWORK,
            problem_ids=["problem1", "problem2"],
            total_points=100.0,
            due_time=datetime.now() + timedelta(days=7)
        )
        assert assignment_request.title == "Test Assignment"
        print("    ✓ Assignment request validation works")
        
        # Test submission request validation
        print("  Testing submission request...")
        submission_request = SubmissionRequest(
            solutions={
                "problem1": {"automaton": sample_automaton},
                "problem2": {"automaton": sample_automaton}
            }
        )
        assert "problem1" in submission_request.solutions
        print("    ✓ Submission request validation works")
        
        # Test grading result
        print("  Testing grading result...")
        result = GradingResult(
            problem_id="test_problem",
            max_score=10.0,
            earned_score=8.5,
            percentage=85.0,
            correctness_score=0.9,
            efficiency_score=0.8,
            style_score=0.7,
            feedback="Good solution with minor efficiency issues",
            test_results=[]
        )
        assert result.percentage == 85.0
        print("    ✓ Grading result creation works")
        
        print("✓ Automated Grading tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Automated Grading test failed: {e}")
        return False

def test_data_models():
    """Test data model validation."""
    print("Testing Data Models...")
    
    try:
        # Test that imports work correctly
        from app.latex_export import ExportRequest, ExportFormat
        from app.api_platform import APIScope, RateLimitTier
        from app.automated_grading import AssignmentType, SubmissionStatus
        
        print("    ✓ All imports successful")
        
        # Test enum values
        assert ExportFormat.TIKZ == "tikz"
        assert APIScope.READ_PROBLEMS == "read:problems"
        assert AssignmentType.HOMEWORK == "homework"
        
        print("    ✓ Enum values correct")
        
        print("✓ Data Models tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Data Models test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running Tests for New Features")
    print("=" * 50)
    
    results = []
    
    # Test each component
    results.append(await test_latex_export())
    results.append(test_api_platform())
    results.append(test_automated_grading())
    results.append(test_data_models())
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    import sys
    sys.path.append("/Users/sharminsirajudeen/Projects/automata-repo/backend")
    
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test runner failed: {e}")
        sys.exit(1)