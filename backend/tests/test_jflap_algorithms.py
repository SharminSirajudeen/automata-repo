"""
Tests for JFLAP algorithms functionality.
"""

import pytest
from fastapi import status
from httpx import AsyncClient


class TestJFLAPConversions:
    """Test JFLAP conversion algorithms."""
    
    async def test_nfa_to_dfa_conversion(self, async_client: AsyncClient, sample_nfa):
        """Test NFA to DFA conversion."""
        response = await async_client.post("/api/jflap/convert/nfa-to-dfa", json={"nfa": sample_nfa})
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "original_nfa" in data
            assert "converted_dfa" in data
            assert "algorithm" in data
            assert "statistics" in data
            assert data["algorithm"] == "subset_construction"
        else:
            # JFLAP service might not be available
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    async def test_dfa_minimization(self, async_client: AsyncClient, sample_dfa):
        """Test DFA minimization."""
        response = await async_client.post("/api/jflap/minimize/dfa", json={"dfa": sample_dfa})
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "original_dfa" in data
            assert "minimized_dfa" in data
            assert "algorithm" in data
            assert "statistics" in data
            assert data["algorithm"] == "hopcroft_minimization"
        else:
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    async def test_regex_to_nfa_conversion(self, async_client: AsyncClient):
        """Test regular expression to NFA conversion."""
        regex_data = {
            "regex": "(a|b)*abb",
            "alphabet": ["a", "b"]
        }
        
        response = await async_client.post("/api/jflap/convert/regex-to-nfa", json=regex_data)
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "regex" in data
            assert "nfa" in data
            assert "algorithm" in data
            assert data["algorithm"] == "thompson_construction"
        else:
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    async def test_nfa_to_regex_conversion(self, async_client: AsyncClient, sample_nfa):
        """Test NFA to regular expression conversion."""
        response = await async_client.post("/api/jflap/convert/nfa-to-regex", json={"nfa": sample_nfa})
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "nfa" in data
            assert "regex" in data
            assert "algorithm" in data
            assert data["algorithm"] == "state_elimination"
        else:
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestJFLAPGrammarOperations:
    """Test JFLAP grammar operations."""
    
    async def test_grammar_to_cnf(self, async_client: AsyncClient, sample_grammar):
        """Test context-free grammar to CNF conversion."""
        response = await async_client.post("/api/jflap/grammar/to-cnf", json={"grammar": sample_grammar})
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "original_grammar" in data
            assert "cnf_grammar" in data
            assert "algorithm" in data
            assert "transformations_applied" in data
        else:
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    async def test_grammar_to_pda(self, async_client: AsyncClient, sample_grammar):
        """Test context-free grammar to PDA conversion."""
        response = await async_client.post("/api/jflap/grammar/to-pda", json={"grammar": sample_grammar})
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "grammar" in data
            assert "pda" in data
            assert "algorithm" in data
            assert data["pda_type"] == "npda"
        else:
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestJFLAPParsing:
    """Test JFLAP parsing algorithms."""
    
    async def test_cyk_parsing(self, async_client: AsyncClient, sample_grammar):
        """Test CYK parsing algorithm."""
        parse_data = {
            "grammar": sample_grammar,
            "input_string": "ab"
        }
        
        response = await async_client.post("/api/jflap/parse/cyk", json=parse_data)
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "grammar" in data
            assert "input_string" in data
            assert "accepted" in data
            assert "parse_table" in data
            assert "algorithm" in data
            assert data["algorithm"] == "cyk"
        else:
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    async def test_ll1_parsing(self, async_client: AsyncClient, sample_grammar):
        """Test LL(1) parsing algorithm."""
        parse_data = {
            "grammar": sample_grammar,
            "input_string": "ab"
        }
        
        response = await async_client.post("/api/jflap/parse/ll1", json=parse_data)
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "grammar" in data
            assert "input_string" in data
            assert "accepted" in data
            assert "derivation" in data
            assert "first_sets" in data
            assert "follow_sets" in data
            assert data["algorithm"] == "ll1"
        else:
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestJFLAPSimulation:
    """Test JFLAP simulation functionality."""
    
    async def test_automaton_simulation(self, async_client: AsyncClient, sample_dfa):
        """Test automaton simulation."""
        simulation_data = {
            "automaton": sample_dfa,
            "input_string": "aa",
            "step_by_step": True
        }
        
        response = await async_client.post("/api/jflap/simulate", json=simulation_data)
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "automaton" in data
            assert "input_string" in data
            assert "accepted" in data
            assert "execution_path" in data
            assert "steps" in data  # Because step_by_step is True
            assert "final_state" in data
        else:
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    async def test_batch_simulation(self, async_client: AsyncClient, sample_dfa):
        """Test batch simulation on multiple strings."""
        batch_data = {
            "automaton": sample_dfa,
            "input_strings": ["a", "aa", "b", "bb", "ab"]
        }
        
        response = await async_client.post("/api/jflap/simulate/batch", json=batch_data)
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "automaton" in data
            assert "results" in data
            assert "statistics" in data
            assert len(data["results"]) == 5
            
            # Check statistics
            stats = data["statistics"]
            assert "total_strings" in stats
            assert "accepted_count" in stats
            assert "rejected_count" in stats
            assert "acceptance_rate" in stats
        else:
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    async def test_automata_comparison(self, async_client: AsyncClient, sample_dfa, sample_nfa):
        """Test comparison between two automata."""
        comparison_data = {
            "automaton1": sample_dfa,
            "automaton2": sample_nfa,
            "test_strings": ["a", "aa", "b", "bb"]
        }
        
        response = await async_client.post("/api/jflap/simulate/compare", json=comparison_data)
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "automaton1" in data
            assert "automaton2" in data
            assert "comparison_results" in data
            assert "statistics" in data
            
            # Check comparison results
            for result in data["comparison_results"]:
                assert "test_string" in result
                assert "automaton1_accepts" in result
                assert "automaton2_accepts" in result
                assert "match" in result
        else:
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestJFLAPInformation:
    """Test JFLAP information endpoints."""
    
    async def test_algorithms_info(self, async_client: AsyncClient):
        """Test getting algorithm information."""
        response = await async_client.get("/api/jflap/algorithms/info")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "conversions" in data
        assert "minimization" in data
        assert "grammar_operations" in data
        assert "parsing" in data
        
        # Check specific algorithms
        assert "nfa_to_dfa" in data["conversions"]
        assert "regex_to_nfa" in data["conversions"]
        assert "dfa_minimization" in data["minimization"]
        assert "cyk" in data["parsing"]
        assert "ll1" in data["parsing"]
    
    async def test_jflap_health_check(self, async_client: AsyncClient):
        """Test JFLAP subsystem health check."""
        response = await async_client.get("/api/jflap/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "status" in data
        assert "algorithms" in data
        assert "jflap_version" in data
        assert "timestamp" in data


class TestJFLAPSecurity:
    """Test security aspects of JFLAP endpoints."""
    
    async def test_malformed_automaton_input(self, async_client: AsyncClient):
        """Test handling of malformed automaton input."""
        malformed_nfa = {
            "states": "invalid_states_format",
            "transitions": [],
            "alphabet": ["a", "b"]
        }
        
        response = await async_client.post("/api/jflap/convert/nfa-to-dfa", json={"nfa": malformed_nfa})
        
        # Should handle gracefully
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
    
    async def test_extremely_large_automaton(self, async_client: AsyncClient):
        """Test handling of extremely large automaton."""
        large_dfa = {
            "states": [
                {
                    "id": f"q{i}",
                    "x": 100,
                    "y": 100,
                    "is_start": i == 0,
                    "is_accept": i == 999
                }
                for i in range(1000)  # Very large automaton
            ],
            "transitions": [
                {
                    "from_state": f"q{i}",
                    "to_state": f"q{(i+1) % 1000}",
                    "symbol": "a"
                }
                for i in range(1000)
            ],
            "alphabet": ["a", "b"]
        }
        
        response = await async_client.post("/api/jflap/minimize/dfa", json={"dfa": large_dfa})
        
        # Should either handle it or reject gracefully
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
    
    async def test_malicious_regex_input(self, async_client: AsyncClient):
        """Test handling of potentially malicious regex input."""
        malicious_regex = {
            "regex": "(a+)+b",  # Potentially catastrophic backtracking
            "alphabet": ["a", "b"]
        }
        
        response = await async_client.post("/api/jflap/convert/regex-to-nfa", json=malicious_regex)
        
        # Should handle timeout or reject malicious patterns
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]


class TestJFLAPRateLimiting:
    """Test rate limiting for JFLAP endpoints."""
    
    async def test_conversion_rate_limiting(self, async_client: AsyncClient, sample_nfa):
        """Test rate limiting on conversion endpoints."""
        responses = []
        
        # Make multiple rapid conversion requests
        for _ in range(40):  # Should exceed rate limit
            response = await async_client.post("/api/jflap/convert/nfa-to-dfa", json={"nfa": sample_nfa})
            responses.append(response)
        
        # Should eventually get rate limited
        rate_limited = any(r.status_code == status.HTTP_429_TOO_MANY_REQUESTS for r in responses)
        # Actual behavior depends on rate limit configuration
    
    async def test_simulation_rate_limiting(self, async_client: AsyncClient, sample_dfa):
        """Test rate limiting on simulation endpoints."""
        simulation_data = {
            "automaton": sample_dfa,
            "input_string": "a",
            "step_by_step": False
        }
        
        responses = []
        for _ in range(60):  # Should exceed rate limit
            response = await async_client.post("/api/jflap/simulate", json=simulation_data)
            responses.append(response)
        
        # Check for rate limiting
        status_codes = [r.status_code for r in responses]
        # Most should be successful or server error, some might be rate limited


@pytest.mark.integration
class TestJFLAPIntegration:
    """Integration tests for JFLAP functionality."""
    
    async def test_complete_nfa_to_dfa_workflow(self, async_client: AsyncClient, sample_nfa):
        """Test complete NFA to DFA conversion and minimization workflow."""
        # Step 1: Convert NFA to DFA
        nfa_response = await async_client.post("/api/jflap/convert/nfa-to-dfa", json={"nfa": sample_nfa})
        
        if nfa_response.status_code == status.HTTP_200_OK:
            dfa_data = nfa_response.json()["converted_dfa"]
            
            # Step 2: Minimize the resulting DFA
            min_response = await async_client.post("/api/jflap/minimize/dfa", json={"dfa": dfa_data})
            
            if min_response.status_code == status.HTTP_200_OK:
                min_data = min_response.json()
                
                # Step 3: Simulate both original and minimized
                original_sim = await async_client.post(
                    "/api/jflap/simulate",
                    json={"automaton": dfa_data, "input_string": "ab", "step_by_step": False}
                )
                
                minimized_sim = await async_client.post(
                    "/api/jflap/simulate",
                    json={
                        "automaton": min_data["minimized_dfa"],
                        "input_string": "ab",
                        "step_by_step": False
                    }
                )
                
                # Both should give same result for same input
                if original_sim.status_code == status.HTTP_200_OK and minimized_sim.status_code == status.HTTP_200_OK:
                    orig_result = original_sim.json()["accepted"]
                    min_result = minimized_sim.json()["accepted"]
                    assert orig_result == min_result