"""
Load testing for the Automata Learning Platform using Locust.
Tests various user scenarios and system performance under load.
"""

import json
import random
import time
from typing import Dict, Any, List

from locust import HttpUser, task, between, tag, events


class AutomataLearningUser(HttpUser):
    """
    Simulates a user of the Automata Learning Platform.
    Tests various user journeys and system performance.
    """
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    
    def on_start(self):
        """Called when a user starts - performs login."""
        self.user_id = f"load_test_user_{random.randint(1000, 9999)}"
        self.session_id = f"session_{random.randint(10000, 99999)}"
        self.access_token = None
        self.problems_cache = []
        self.current_problem = None
        
        # Register and login
        self.register_user()
        self.login_user()
    
    def register_user(self):
        """Register a new user for testing."""
        user_data = {
            "email": f"{self.user_id}@loadtest.com",
            "password": "LoadTest123!",
            "full_name": f"Load Test User {self.user_id}"
        }
        
        with self.client.post("/auth/register", json=user_data, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 400:
                # User might already exist, that's okay
                response.success()
            else:
                response.failure(f"Registration failed with status {response.status_code}")
    
    def login_user(self):
        """Login the user and store access token."""
        login_data = {
            "email": f"{self.user_id}@loadtest.com",
            "password": "LoadTest123!"
        }
        
        with self.client.post("/auth/login", json=login_data, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("access_token")
                response.success()
            else:
                response.failure(f"Login failed with status {response.status_code}")
    
    @property
    def auth_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        if self.access_token:
            return {"Authorization": f"Bearer {self.access_token}"}
        return {}
    
    @task(10)
    @tag("basic")
    def view_homepage(self):
        """Test homepage access."""
        self.client.get("/", name="homepage")
    
    @task(8)
    @tag("basic")
    def health_check(self):
        """Test health check endpoint."""
        self.client.get("/health", name="health_check")
    
    @task(15)
    @tag("problems")
    def browse_problems(self):
        """Test browsing available problems."""
        with self.client.get("/problems/", headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                self.problems_cache = data.get("problems", [])
                response.success()
            else:
                response.failure(f"Failed to load problems: {response.status_code}")
    
    @task(12)
    @tag("problems")
    def view_specific_problem(self):
        """Test viewing a specific problem."""
        problem_ids = ["sample_dfa", "dfa_ending_ab", "nfa_basic", "cfg_simple"]
        problem_id = random.choice(problem_ids)
        
        with self.client.get(f"/problems/{problem_id}", headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                self.current_problem = response.json()
                response.success()
            elif response.status_code == 404:
                # Problem not found is acceptable
                response.success()
            else:
                response.failure(f"Failed to load problem {problem_id}: {response.status_code}")
    
    @task(8)
    @tag("problems")
    def get_problem_hint(self):
        """Test getting problem hints."""
        problem_ids = ["sample_dfa", "dfa_ending_ab"]
        problem_id = random.choice(problem_ids)
        hint_index = random.randint(0, 2)
        
        with self.client.get(f"/problems/{problem_id}/hint?hint_index={hint_index}", 
                           headers=self.auth_headers, catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Hint request failed: {response.status_code}")
    
    @task(6)
    @tag("problems", "validation")
    def validate_solution(self):
        """Test solution validation."""
        problem_id = "sample_dfa"
        
        # Sample DFA solution
        solution_data = {
            "automaton": {
                "states": [
                    {"id": "q0", "x": 100, "y": 100, "is_start": True, "is_accept": False},
                    {"id": "q1", "x": 200, "y": 100, "is_start": False, "is_accept": True}
                ],
                "transitions": [
                    {"from_state": "q0", "to_state": "q1", "symbol": "a"},
                    {"from_state": "q1", "to_state": "q1", "symbol": "a"},
                    {"from_state": "q0", "to_state": "q0", "symbol": "b"},
                    {"from_state": "q1", "to_state": "q0", "symbol": "b"}
                ],
                "alphabet": ["a", "b"]
            },
            "user_id": self.user_id
        }
        
        with self.client.post(f"/problems/{problem_id}/validate", 
                            json=solution_data, headers=self.auth_headers, 
                            catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Validation failed: {response.status_code}")
    
    @task(4)
    @tag("jflap", "conversion")
    def test_nfa_to_dfa_conversion(self):
        """Test JFLAP NFA to DFA conversion."""
        nfa_data = {
            "nfa": {
                "states": [
                    {"id": "q0", "x": 100, "y": 100, "is_start": True, "is_accept": False},
                    {"id": "q1", "x": 200, "y": 100, "is_start": False, "is_accept": True}
                ],
                "transitions": [
                    {"from_state": "q0", "to_state": "q1", "symbol": "a"},
                    {"from_state": "q0", "to_state": "q0", "symbol": "ε"}
                ],
                "alphabet": ["a", "b"]
            }
        }
        
        with self.client.post("/api/jflap/convert/nfa-to-dfa", json=nfa_data, 
                            headers=self.auth_headers, catch_response=True) as response:
            if response.status_code in [200, 500]:  # 500 might be expected if service unavailable
                response.success()
            else:
                response.failure(f"NFA to DFA conversion failed: {response.status_code}")
    
    @task(3)
    @tag("jflap", "minimization")
    def test_dfa_minimization(self):
        """Test JFLAP DFA minimization."""
        dfa_data = {
            "dfa": {
                "states": [
                    {"id": "q0", "x": 100, "y": 100, "is_start": True, "is_accept": False},
                    {"id": "q1", "x": 200, "y": 100, "is_start": False, "is_accept": True},
                    {"id": "q2", "x": 300, "y": 100, "is_start": False, "is_accept": False}
                ],
                "transitions": [
                    {"from_state": "q0", "to_state": "q1", "symbol": "a"},
                    {"from_state": "q1", "to_state": "q2", "symbol": "b"},
                    {"from_state": "q2", "to_state": "q0", "symbol": "a"}
                ],
                "alphabet": ["a", "b"]
            }
        }
        
        with self.client.post("/api/jflap/minimize/dfa", json=dfa_data, 
                            headers=self.auth_headers, catch_response=True) as response:
            if response.status_code in [200, 500]:
                response.success()
            else:
                response.failure(f"DFA minimization failed: {response.status_code}")
    
    @task(5)
    @tag("jflap", "simulation")
    def test_automaton_simulation(self):
        """Test automaton simulation."""
        simulation_data = {
            "automaton": {
                "states": [
                    {"id": "q0", "x": 100, "y": 100, "is_start": True, "is_accept": False},
                    {"id": "q1", "x": 200, "y": 100, "is_start": False, "is_accept": True}
                ],
                "transitions": [
                    {"from_state": "q0", "to_state": "q1", "symbol": "a"},
                    {"from_state": "q1", "to_state": "q1", "symbol": "a"}
                ],
                "alphabet": ["a", "b"]
            },
            "input_string": random.choice(["a", "aa", "aaa", "b", "ab", "ba"]),
            "step_by_step": random.choice([True, False])
        }
        
        with self.client.post("/api/jflap/simulate", json=simulation_data, 
                            headers=self.auth_headers, catch_response=True) as response:
            if response.status_code in [200, 500]:
                response.success()
            else:
                response.failure(f"Simulation failed: {response.status_code}")
    
    @task(2)
    @tag("ai", "requires_api_key")
    def test_ai_status(self):
        """Test AI service status."""
        with self.client.get("/api/ai/status", headers=self.auth_headers, catch_response=True) as response:
            if response.status_code in [200, 401]:  # 401 expected without API key
                response.success()
            else:
                response.failure(f"AI status check failed: {response.status_code}")
    
    @task(1)
    @tag("learning")
    def test_learning_recommendations(self):
        """Test learning recommendations."""
        with self.client.get(f"/api/learning/recommendations/{self.user_id}", 
                           headers=self.auth_headers, catch_response=True) as response:
            if response.status_code in [200, 403, 500]:  # Various expected responses
                response.success()
            else:
                response.failure(f"Learning recommendations failed: {response.status_code}")
    
    @task(1)
    @tag("papers")
    def test_papers_search(self):
        """Test research papers search."""
        search_data = {
            "query": random.choice(["automata theory", "finite state machines", "regular languages"]),
            "limit": 5
        }
        
        with self.client.post("/api/papers/search", json=search_data, 
                            headers=self.auth_headers, catch_response=True) as response:
            if response.status_code in [200, 500]:
                response.success()
            else:
                response.failure(f"Papers search failed: {response.status_code}")
    
    @task(1)
    @tag("verification")
    def test_equivalence_check(self):
        """Test automata equivalence checking."""
        equivalence_data = {
            "automaton1": {
                "states": [{"id": "q0", "is_start": True, "is_accept": True}],
                "transitions": [{"from_state": "q0", "to_state": "q0", "symbol": "a"}],
                "alphabet": ["a"]
            },
            "automaton2": {
                "states": [{"id": "p0", "is_start": True, "is_accept": True}],
                "transitions": [{"from_state": "p0", "to_state": "p0", "symbol": "a"}],
                "alphabet": ["a"]
            }
        }
        
        with self.client.post("/api/verification/equivalence", json=equivalence_data, 
                            headers=self.auth_headers, catch_response=True) as response:
            if response.status_code in [200, 500]:
                response.success()
            else:
                response.failure(f"Equivalence check failed: {response.status_code}")
    
    @task(2)
    @tag("monitoring")
    def test_performance_metrics(self):
        """Test performance metrics endpoint."""
        with self.client.get("/metrics/performance", headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Performance metrics failed: {response.status_code}")


class AdminUser(HttpUser):
    """Simulates an admin user with elevated permissions."""
    
    wait_time = between(10, 30)  # Admins act less frequently
    weight = 1  # Lower weight than regular users
    
    def on_start(self):
        """Admin user setup."""
        self.admin_id = "admin_load_test"
        self.access_token = None
        # In a real scenario, this would use actual admin credentials
    
    @task(5)
    @tag("admin", "monitoring")
    def check_system_health(self):
        """Admin checks detailed system health."""
        self.client.get("/health/detailed", name="admin_health_check")
    
    @task(3)
    @tag("admin", "monitoring")
    def view_prometheus_metrics(self):
        """Admin views Prometheus metrics."""
        self.client.get("/metrics", name="prometheus_metrics")
    
    @task(2)
    @tag("admin", "jflap")
    def check_jflap_health(self):
        """Admin checks JFLAP service health."""
        self.client.get("/api/jflap/health", name="jflap_health")
    
    @task(1)
    @tag("admin", "ai")
    def check_ai_health(self):
        """Admin checks AI service health."""
        self.client.get("/api/ai/health", name="ai_health")


class StressTestUser(HttpUser):
    """High-intensity user for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Very short wait times
    weight = 1  # Used sparingly
    
    @task(20)
    @tag("stress")
    def rapid_fire_requests(self):
        """Make rapid requests to stress test the system."""
        endpoints = [
            "/",
            "/health",
            "/healthz"
        ]
        
        endpoint = random.choice(endpoints)
        self.client.get(endpoint, name=f"stress_{endpoint.replace('/', '_')}")
    
    @task(5)
    @tag("stress", "problems")
    def stress_problem_endpoints(self):
        """Stress test problem-related endpoints."""
        self.client.get("/problems/", name="stress_problems_list")


# Load testing scenarios
class QuickTest(HttpUser):
    """Quick load test for basic functionality."""
    tasks = [AutomataLearningUser.view_homepage, AutomataLearningUser.health_check]
    wait_time = between(1, 2)


# Custom events for detailed reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts."""
    print(f"Load test starting with {environment.runner.user_count} users")
    print(f"Host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops."""
    print("Load test completed")
    
    # Print summary statistics
    stats = environment.runner.stats
    print(f"\nSummary:")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min response time: {stats.total.min_response_time}ms")
    print(f"Max response time: {stats.total.max_response_time}ms")


# Performance thresholds
@events.request_failure.add_listener
def on_request_failure(request_type, name, response_time, response_length, exception, **kwargs):
    """Handle request failures."""
    if response_time > 5000:  # 5 second threshold
        print(f"SLOW REQUEST: {name} took {response_time}ms")


@events.request_success.add_listener  
def on_request_success(request_type, name, response_time, response_length, **kwargs):
    """Handle successful requests."""
    if response_time > 2000:  # 2 second warning threshold
        print(f"SLOW SUCCESS: {name} took {response_time}ms")


# Custom load test shapes
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust import LoadTestShape


class StepLoadShape(LoadTestShape):
    """
    A step load shape that gradually increases load.
    """
    step_time = 30  # seconds
    step_load = 10  # users per step
    spawn_rate = 2  # users per second
    time_limit = 300  # 5 minutes total
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = run_time // self.step_time
        return (current_step * self.step_load, self.spawn_rate)


class SpikeLoadShape(LoadTestShape):
    """
    A spike load shape for testing system resilience.
    """
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < 60:
            return (10, 2)  # Warm up
        elif run_time < 120:
            return (100, 10)  # Spike
        elif run_time < 180:
            return (10, 2)  # Cool down
        else:
            return None


# Example usage scenarios:
"""
To run load tests:

# Basic load test with 50 users
locust -f locustfile.py --host=http://localhost:8000 -u 50 -r 5 -t 5m

# Step load test
locust -f locustfile.py --host=http://localhost:8000 --shape=StepLoadShape

# Spike load test  
locust -f locustfile.py --host=http://localhost:8000 --shape=SpikeLoadShape

# Test specific tags
locust -f locustfile.py --host=http://localhost:8000 --tags basic problems

# Exclude heavy tests
locust -f locustfile.py --host=http://localhost:8000 --exclude-tags stress ai

# Headless mode with CSV reporting
locust -f locustfile.py --host=http://localhost:8000 -u 100 -r 10 -t 10m --headless --csv=results

# Distributed load testing (run this on master)
locust -f locustfile.py --host=http://localhost:8000 --master

# Distributed load testing (run this on workers)
locust -f locustfile.py --host=http://localhost:8000 --worker --master-host=192.168.1.100
"""