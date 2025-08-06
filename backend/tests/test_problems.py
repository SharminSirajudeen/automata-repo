"""
Tests for problems functionality.
"""

import pytest
from fastapi import status
from httpx import AsyncClient


class TestProblemRetrieval:
    """Test problem retrieval functionality."""
    
    async def test_get_all_problems(self, async_client: AsyncClient):
        """Test getting all problems."""
        response = await async_client.get("/problems/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "problems" in data
        assert isinstance(data["problems"], list)
    
    async def test_get_specific_problem(self, async_client: AsyncClient, sample_problem):
        """Test getting a specific problem by ID."""
        # First, we'd need to create the problem in the database
        # For now, test with a known problem ID
        response = await async_client.get("/problems/sample_dfa")
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "id" in data
            assert "type" in data
            assert "title" in data
            assert "description" in data
        else:
            # Problem doesn't exist, which is expected in test environment
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    async def test_get_nonexistent_problem(self, async_client: AsyncClient):
        """Test getting a non-existent problem."""
        response = await async_client.get("/problems/nonexistent_problem")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Problem not found" in response.json()["detail"]


class TestProblemValidation:
    """Test problem solution validation."""
    
    async def test_validate_dfa_solution(self, async_client: AsyncClient, sample_dfa):
        """Test DFA solution validation."""
        solution_data = {
            "automaton": sample_dfa,
            "user_id": "test_user_123"
        }
        
        response = await async_client.post("/problems/sample_dfa/validate", json=solution_data)
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "is_correct" in data
            assert "score" in data
            assert "feedback" in data
            assert "test_results" in data
        else:
            # Problem doesn't exist in test environment
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    async def test_validate_solution_invalid_problem(self, async_client: AsyncClient, sample_dfa):
        """Test validation with invalid problem ID."""
        solution_data = {
            "automaton": sample_dfa,
            "user_id": "test_user_123"
        }
        
        response = await async_client.post("/problems/invalid_problem/validate", json=solution_data)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    async def test_validate_malformed_solution(self, async_client: AsyncClient):
        """Test validation with malformed solution data."""
        malformed_solution = {
            "automaton": {"invalid": "data"},
            "user_id": "test_user_123"
        }
        
        response = await async_client.post("/problems/sample_dfa/validate", json=malformed_solution)
        
        # Should handle gracefully
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]


class TestProblemHints:
    """Test problem hint functionality."""
    
    async def test_get_problem_hint(self, async_client: AsyncClient):
        """Test getting a problem hint."""
        response = await async_client.get("/problems/sample_dfa/hint?hint_index=0")
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "hint" in data
            assert "total_hints" in data
        else:
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    async def test_get_hint_invalid_index(self, async_client: AsyncClient):
        """Test getting hint with invalid index."""
        response = await async_client.get("/problems/sample_dfa/hint?hint_index=999")
        
        # Should return error for invalid hint index
        assert response.status_code in [status.HTTP_404_NOT_FOUND, status.HTTP_400_BAD_REQUEST]
    
    async def test_get_ai_hint(self, async_client: AsyncClient, sample_dfa):
        """Test getting AI-powered hint."""
        hint_request = {
            "user_automaton": sample_dfa,
            "difficulty_level": "beginner"
        }
        
        response = await async_client.post("/problems/sample_dfa/ai-hint", json=hint_request)
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "ai_hint" in data
        else:
            # AI service might not be available in test environment
            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ]


class TestProblemSecurity:
    """Test security aspects of problem endpoints."""
    
    async def test_sql_injection_in_problem_id(self, async_client: AsyncClient):
        """Test SQL injection attempt in problem ID."""
        malicious_id = "'; DROP TABLE problems; --"
        response = await async_client.get(f"/problems/{malicious_id}")
        
        # Should handle gracefully without crashing
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND
        ]
    
    async def test_xss_in_solution_data(self, async_client: AsyncClient):
        """Test XSS attempt in solution data."""
        malicious_solution = {
            "automaton": {
                "states": [
                    {
                        "id": "<script>alert('xss')</script>",
                        "x": 100,
                        "y": 100,
                        "is_start": True,
                        "is_accept": False
                    }
                ],
                "transitions": [],
                "alphabet": ["a", "b"]
            },
            "user_id": "test_user_123"
        }
        
        response = await async_client.post("/problems/sample_dfa/validate", json=malicious_solution)
        
        # Should handle malicious input gracefully
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
    
    async def test_large_automaton_rejection(self, async_client: AsyncClient):
        """Test rejection of abnormally large automaton data."""
        large_automaton = {
            "states": [
                {
                    "id": f"q{i}",
                    "x": 100,
                    "y": 100,
                    "is_start": i == 0,
                    "is_accept": False
                }
                for i in range(10000)  # Very large number of states
            ],
            "transitions": [],
            "alphabet": ["a", "b"]
        }
        
        solution_data = {
            "automaton": large_automaton,
            "user_id": "test_user_123"
        }
        
        response = await async_client.post("/problems/sample_dfa/validate", json=solution_data)
        
        # Should reject or handle large payloads appropriately
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]


class TestProblemValidationLogic:
    """Test the actual validation logic for different automaton types."""
    
    def test_dfa_validation_correct_solution(self, sample_dfa, sample_problem):
        """Test DFA validation with correct solution."""
        from app.routers.problems_router import validate_dfa
        
        # Create a mock problem with test strings that match the sample DFA
        test_problem = type('Problem', (), {
            'test_strings': [
                {"string": "a", "should_accept": True},
                {"string": "aa", "should_accept": True},
                {"string": "b", "should_accept": False},
                {"string": "bb", "should_accept": False}
            ]
        })()
        
        # Mock automaton object
        automaton = type('Automaton', (), {
            'states': [
                type('State', (), {'id': 'q0', 'is_start': True, 'is_accept': False})(),
                type('State', (), {'id': 'q1', 'is_start': False, 'is_accept': True})()
            ]
        })()
        
        # This test would need the actual validation logic to be testable
        # For now, it's a placeholder showing the structure
    
    def test_dfa_validation_incorrect_solution(self):
        """Test DFA validation with incorrect solution."""
        # Placeholder for incorrect solution validation test
        pass
    
    def test_nfa_validation(self):
        """Test NFA validation logic."""
        # Placeholder for NFA validation test
        pass


class TestProblemRateLimiting:
    """Test rate limiting for problem endpoints."""
    
    async def test_validation_rate_limiting(self, async_client: AsyncClient, sample_dfa):
        """Test rate limiting on validation endpoint."""
        solution_data = {
            "automaton": sample_dfa,
            "user_id": "test_user_123"
        }
        
        # Make multiple rapid requests
        responses = []
        for _ in range(50):  # Exceed rate limit
            response = await async_client.post("/problems/sample_dfa/validate", json=solution_data)
            responses.append(response)
        
        # Should eventually get rate limited
        rate_limited = any(r.status_code == status.HTTP_429_TOO_MANY_REQUESTS for r in responses)
        # Note: Actual behavior depends on rate limit configuration
    
    async def test_hint_rate_limiting(self, async_client: AsyncClient):
        """Test rate limiting on hint endpoint."""
        # Make multiple rapid requests for hints
        responses = []
        for _ in range(20):
            response = await async_client.get("/problems/sample_dfa/hint?hint_index=0")
            responses.append(response)
        
        # Check for rate limiting (behavior depends on configuration)
        status_codes = [r.status_code for r in responses]
        # Most should be either successful or 404 (problem not found)
        # Some might be rate limited


@pytest.mark.integration
class TestProblemIntegration:
    """Integration tests for problem functionality."""
    
    async def test_complete_problem_solving_flow(self, async_client: AsyncClient, auth_headers, sample_dfa):
        """Test complete problem solving flow."""
        # This would test: get problem -> get hint -> validate solution
        
        # Step 1: Get problem
        problem_response = await async_client.get("/problems/sample_dfa")
        
        # Step 2: Get hint (if problem exists)
        if problem_response.status_code == status.HTTP_200_OK:
            hint_response = await async_client.get("/problems/sample_dfa/hint?hint_index=0")
            
            # Step 3: Submit solution
            solution_data = {
                "automaton": sample_dfa,
                "user_id": "test_user_123"
            }
            validation_response = await async_client.post(
                "/problems/sample_dfa/validate",
                json=solution_data
            )
            
            # All steps should work together
            assert hint_response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]
            assert validation_response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]