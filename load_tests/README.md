# Load Testing for Automata Learning Platform

This directory contains load testing scripts and configurations for the Automata Learning Platform using [Locust](https://locust.io/).

## Overview

The load testing suite simulates realistic user behavior patterns to evaluate system performance under various load conditions. It tests all major platform features including authentication, problem solving, JFLAP algorithms, AI services, and more.

## Files

- `locustfile.py` - Main Locust test definitions with user behavior simulation
- `load_test_config.py` - Pre-defined test scenarios and configurations
- `README.md` - This documentation file

## Prerequisites

1. Install Locust:
```bash
pip install locust
```

2. Ensure the Automata Learning Platform is running:
```bash
cd ../backend
uvicorn app.main:app --reload
```

3. (Optional) Set up test data and API keys for comprehensive testing

## Quick Start

### Basic Load Test
```bash
# Run with 50 users for 5 minutes
locust -f locustfile.py --host=http://localhost:8000 -u 50 -r 5 -t 5m
```

### Web UI Mode
```bash
# Start Locust web interface (default: http://localhost:8089)
locust -f locustfile.py --host=http://localhost:8000
```

### Headless Mode with CSV Export
```bash
# Run headless with results export
locust -f locustfile.py --host=http://localhost:8000 -u 100 -r 10 -t 10m --headless --csv=results
```

## Test Scenarios

### Pre-defined Scenarios

Run pre-configured scenarios using the config helper:

```bash
# View all available scenarios
python load_test_config.py

# Example scenarios:
locust -f locustfile.py --host=http://localhost:8000 -u 5 -r 1 -t 2m --tags basic     # Smoke test
locust -f locustfile.py --host=http://localhost:8000 -u 25 -r 3 -t 10m               # Functional test  
locust -f locustfile.py --host=http://localhost:8000 -u 100 -r 10 -t 15m             # Load test
locust -f locustfile.py --host=http://localhost:8000 -u 500 -r 25 -t 20m             # Stress test
```

### Test Categories

Use tags to focus on specific features:

```bash
# Test only basic functionality
locust -f locustfile.py --host=http://localhost:8000 --tags basic

# Test problem-solving features
locust -f locustfile.py --host=http://localhost:8000 --tags problems validation

# Test JFLAP algorithms
locust -f locustfile.py --host=http://localhost:8000 --tags jflap

# Exclude heavy tests
locust -f locustfile.py --host=http://localhost:8000 --exclude-tags stress ai
```

## Load Test Shapes

### Step Load
Gradually increases load over time:
```bash
locust -f locustfile.py --host=http://localhost:8000 --shape=StepLoadShape
```

### Spike Load
Tests system resilience with sudden load spikes:
```bash
locust -f locustfile.py --host=http://localhost:8000 --shape=SpikeLoadShape
```

## User Types

The test suite includes different user types:

1. **AutomataLearningUser** (90% of users)
   - Regular students using the platform
   - Browses problems, gets hints, validates solutions
   - Tests core learning functionality

2. **AdminUser** (5% of users)
   - Administrative users
   - Checks system health and monitoring endpoints
   - Lower frequency, higher privilege operations

3. **StressTestUser** (5% of users)
   - High-intensity users for stress testing
   - Rapid-fire requests to test system limits
   - Very short wait times between requests

## Test Coverage

The load tests cover:

### Authentication
- User registration
- User login
- Token-based authentication
- Session management

### Problems
- Browse problem catalog
- View specific problems
- Get hints (static and AI-powered)
- Validate solutions
- Submit solutions

### JFLAP Algorithms
- NFA to DFA conversion
- DFA minimization
- Regular expression conversions
- Automaton simulation
- Grammar operations

### AI Services (requires API keys)
- AI status checks
- Prompt generation
- Model orchestration
- Proof assistance
- Semantic search

### Learning System
- Performance tracking
- Recommendations
- Analytics
- Session management

### Research Papers
- Paper search
- Recommendations
- Citations

### Verification
- Equivalence checking
- Formal verification
- Pumping lemma validation

### Monitoring
- Health checks
- Performance metrics
- System status

## Performance Thresholds

The tests monitor for:

- **Response Time**: < 500ms acceptable, < 200ms good
- **Error Rate**: < 5% acceptable, < 1% good  
- **Throughput**: > 50 RPS target, > 200 RPS maximum

## Distributed Load Testing

For higher loads, run distributed tests:

### Master Node
```bash
locust -f locustfile.py --host=http://localhost:8000 --master
```

### Worker Nodes
```bash
locust -f locustfile.py --host=http://localhost:8000 --worker --master-host=MASTER_IP
```

## Results Analysis

### Real-time Monitoring
- Web UI: http://localhost:8089 (during test)
- Watch response times, error rates, and RPS
- Monitor individual endpoint performance

### CSV Export
Results are exported to CSV files:
- `results_failures.csv` - Failed requests
- `results_stats.csv` - Request statistics  
- `results_stats_history.csv` - Historical data

### Key Metrics to Monitor
1. **Average Response Time** - Should stay under thresholds
2. **95th Percentile Response Time** - Catch outliers
3. **Error Rate** - Should be minimal (< 1-5%)
4. **Requests per Second** - System throughput
5. **Concurrent Users** - Load capacity

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Ensure the backend server is running
   - Check host URL and port
   - Verify network connectivity

2. **High Error Rates**
   - Check server logs for errors
   - Monitor system resources (CPU, memory)
   - Verify database connectivity

3. **Rate Limiting**
   - Some tests may trigger rate limits
   - This is expected behavior for security testing
   - Adjust user spawn rates if needed

4. **Authentication Failures**
   - Tests create temporary users
   - Database should allow new registrations
   - Check auth token expiration

### Performance Issues

1. **Slow Response Times**
   - Monitor server CPU and memory usage
   - Check database query performance
   - Review application logs

2. **Memory Leaks**
   - Run endurance tests to detect leaks
   - Monitor memory usage over time
   - Check for proper resource cleanup

3. **Database Bottlenecks**
   - Monitor database connections
   - Check query execution times
   - Consider connection pooling

## Best Practices

1. **Start Small**: Begin with smoke tests before heavy loads
2. **Gradual Increase**: Use step loads to find breaking points
3. **Monitor Resources**: Watch server CPU, memory, and disk
4. **Clean Data**: Reset test data between major test runs
5. **Realistic Scenarios**: Model actual user behavior patterns
6. **Document Results**: Keep records of test results and configurations

## Environment Testing

### Local Development
```bash
# Light testing for development
locust -f locustfile.py --host=http://localhost:8000 -u 10 -r 2 -t 5m
```

### Staging Environment
```bash
# More intensive testing
locust -f locustfile.py --host=https://staging.automata-platform.com -u 100 -r 10 -t 15m
```

### Production (Use with Caution!)
```bash
# Only during maintenance windows
locust -f locustfile.py --host=https://automata-platform.com -u 50 -r 5 -t 10m --tags basic
```

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
name: Load Test
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install locust
      - name: Run load test
        run: |
          cd load_tests
          locust -f locustfile.py --host=${{ secrets.STAGING_URL }} -u 50 -r 5 -t 10m --headless --csv=results
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: load-test-results
          path: load_tests/results*.csv
```

## Security Considerations

- Tests create temporary user accounts
- No sensitive data should be used in tests  
- API keys for AI services should be test keys only
- Production testing should be done carefully and during maintenance windows
- Rate limiting may block excessive requests (this is expected)

## Support

For issues with load testing:
1. Check server logs during test execution
2. Monitor system resources
3. Review Locust documentation: https://docs.locust.io/
4. Verify test configuration matches your environment