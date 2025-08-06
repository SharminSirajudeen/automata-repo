"""
Load testing configuration for the Automata Learning Platform.
Defines various test scenarios and configurations.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    name: str
    description: str
    users: int
    spawn_rate: int  # users per second
    duration: str  # e.g., "5m", "10s", "1h"
    host: str
    tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None
    shape: Optional[str] = None


# Pre-defined test configurations
LOAD_TEST_SCENARIOS = {
    "smoke": LoadTestConfig(
        name="Smoke Test",
        description="Basic functionality test with minimal load",
        users=5,
        spawn_rate=1,
        duration="2m",
        host="http://localhost:8000",
        tags=["basic"]
    ),
    
    "functional": LoadTestConfig(
        name="Functional Test",
        description="Test all major functionality with moderate load",
        users=25,
        spawn_rate=3,
        duration="10m",
        host="http://localhost:8000",
        exclude_tags=["stress", "admin"]
    ),
    
    "load": LoadTestConfig(
        name="Load Test",
        description="Normal expected load testing",
        users=100,
        spawn_rate=10,
        duration="15m",
        host="http://localhost:8000",
        exclude_tags=["stress"]
    ),
    
    "stress": LoadTestConfig(
        name="Stress Test",
        description="High load stress testing",
        users=500,
        spawn_rate=25,
        duration="20m",
        host="http://localhost:8000"
    ),
    
    "spike": LoadTestConfig(
        name="Spike Test",
        description="Sudden load spike testing",
        users=200,
        spawn_rate=50,
        duration="10m",
        host="http://localhost:8000",
        shape="SpikeLoadShape"
    ),
    
    "step": LoadTestConfig(
        name="Step Load Test",
        description="Gradually increasing load",
        users=300,
        spawn_rate=10,
        duration="20m",
        host="http://localhost:8000",
        shape="StepLoadShape"
    ),
    
    "endurance": LoadTestConfig(
        name="Endurance Test",
        description="Long-running test for stability",
        users=50,
        spawn_rate=5,
        duration="2h",
        host="http://localhost:8000",
        exclude_tags=["stress"]
    ),
    
    "problems_focus": LoadTestConfig(
        name="Problems-Focused Test",
        description="Heavy testing of problem-solving features",
        users=75,
        spawn_rate=8,
        duration="15m",
        host="http://localhost:8000",
        tags=["problems", "validation"]
    ),
    
    "jflap_focus": LoadTestConfig(
        name="JFLAP-Focused Test",
        description="Heavy testing of JFLAP algorithms",
        users=50,
        spawn_rate=5,
        duration="12m",
        host="http://localhost:8000",
        tags=["jflap"]
    ),
    
    "ai_focus": LoadTestConfig(
        name="AI-Focused Test",
        description="Testing AI services (requires API keys)",
        users=20,
        spawn_rate=2,
        duration="10m",
        host="http://localhost:8000",
        tags=["ai"]
    )
}


# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "response_time": {
        "good": 200,      # ms
        "acceptable": 500,  # ms
        "poor": 1000,     # ms
        "timeout": 5000   # ms
    },
    "error_rate": {
        "good": 0.01,     # 1%
        "acceptable": 0.05,  # 5%
        "poor": 0.10      # 10%
    },
    "throughput": {
        "min_rps": 10,    # requests per second
        "target_rps": 50,
        "max_rps": 200
    }
}


# Test data sets
TEST_DATA = {
    "users": [
        {"email": "user1@loadtest.com", "password": "LoadTest123!", "full_name": "Load Test User 1"},
        {"email": "user2@loadtest.com", "password": "LoadTest123!", "full_name": "Load Test User 2"},
        {"email": "user3@loadtest.com", "password": "LoadTest123!", "full_name": "Load Test User 3"},
    ],
    
    "problems": [
        "sample_dfa",
        "dfa_ending_ab", 
        "nfa_basic",
        "cfg_simple",
        "pda_balanced",
        "tm_simple"
    ],
    
    "test_strings": [
        "a", "b", "aa", "ab", "ba", "bb", 
        "aaa", "aab", "aba", "abb", "baa", "bab", "bba", "bbb",
        "aaaa", "abab", "baba", "bbbb"
    ],
    
    "search_queries": [
        "automata theory",
        "finite state machines", 
        "regular languages",
        "context-free grammars",
        "pushdown automata",
        "turing machines",
        "computational complexity",
        "pumping lemma"
    ],
    
    "regex_patterns": [
        "a*",
        "a+b*",
        "(a|b)*",
        "a*b*",
        "(ab)*",
        "a(a|b)*b",
        "(a+b)*abb",
        "((a|b)(a|b))*"
    ]
}


# Environment-specific configurations
ENVIRONMENTS = {
    "local": {
        "host": "http://localhost:8000",
        "description": "Local development environment"
    },
    "staging": {
        "host": "https://staging.automata-platform.com",
        "description": "Staging environment"
    },
    "production": {
        "host": "https://automata-platform.com", 
        "description": "Production environment (use with caution!)"
    }
}


def get_locust_command(scenario_name: str, environment: str = "local") -> str:
    """Generate locust command for a given scenario and environment."""
    if scenario_name not in LOAD_TEST_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {environment}")
    
    config = LOAD_TEST_SCENARIOS[scenario_name]
    env_config = ENVIRONMENTS[environment]
    
    cmd_parts = [
        "locust",
        "-f locustfile.py",
        f"--host={env_config['host']}",
        f"-u {config.users}",
        f"-r {config.spawn_rate}",
        f"-t {config.duration}"
    ]
    
    if config.tags:
        cmd_parts.append(f"--tags {' '.join(config.tags)}")
    
    if config.exclude_tags:
        cmd_parts.append(f"--exclude-tags {' '.join(config.exclude_tags)}")
    
    if config.shape:
        cmd_parts.append(f"--shape={config.shape}")
    
    return " ".join(cmd_parts)


def print_all_scenarios():
    """Print all available test scenarios."""
    print("Available Load Test Scenarios:")
    print("=" * 50)
    
    for name, config in LOAD_TEST_SCENARIOS.items():
        print(f"\n{name}:")
        print(f"  Description: {config.description}")
        print(f"  Users: {config.users}")
        print(f"  Spawn Rate: {config.spawn_rate}/sec")
        print(f"  Duration: {config.duration}")
        if config.tags:
            print(f"  Tags: {', '.join(config.tags)}")
        if config.exclude_tags:
            print(f"  Exclude Tags: {', '.join(config.exclude_tags)}")
        if config.shape:
            print(f"  Shape: {config.shape}")
        
        print(f"  Command: {get_locust_command(name)}")


if __name__ == "__main__":
    print_all_scenarios()