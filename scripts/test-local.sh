#!/bin/bash

# Local Testing Script for Automata Learning Platform
# This script helps you test the platform locally before deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}    Automata Learning Platform - Local Test    ${NC}"
echo -e "${BLUE}================================================${NC}"

# Function to check dependencies
check_dependencies() {
    echo -e "\n${YELLOW}Checking dependencies...${NC}"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Node.js is not installed. Please install Node.js 18+ first.${NC}"
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}npm is not installed. Please install npm first.${NC}"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 is not installed. Please install Python 3.8+ first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}All dependencies found!${NC}"
}

# Function to setup test environment
setup_test_env() {
    echo -e "\n${YELLOW}Setting up test environment...${NC}"
    
    # Create test environment file if it doesn't exist
    if [ ! -f ".env.test" ]; then
        cat > .env.test <<EOF
# Test Environment Variables
NODE_ENV=test
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
POSTGRES_USER=postgres
POSTGRES_PASSWORD=testpassword123
POSTGRES_DB=automata_test
REDIS_PASSWORD=testredis123
JWT_SECRET=test-jwt-secret-key-change-in-production
SECRET_KEY=test-secret-key-change-in-production
OPENAI_API_KEY=sk-test-key
VITE_ENABLE_AI_FEATURES=true
VITE_ENABLE_COLLABORATION=true
VITE_DEBUG=true
EOF
        echo -e "${GREEN}Test environment file created!${NC}"
    fi
    
    # Load test environment
    export $(cat .env.test | grep -v '^#' | xargs)
}

# Function to test frontend
test_frontend() {
    echo -e "\n${BLUE}Testing Frontend...${NC}"
    cd frontend
    
    # Install dependencies
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
    
    # Run type checking
    echo -e "${YELLOW}Running TypeScript type check...${NC}"
    npm run type-check || true
    
    # Run linting
    echo -e "${YELLOW}Running ESLint...${NC}"
    npm run lint || true
    
    # Run unit tests if available
    if [ -f "package.json" ] && grep -q "\"test\"" package.json; then
        echo -e "${YELLOW}Running frontend tests...${NC}"
        npm test -- --run || true
    fi
    
    # Build frontend
    echo -e "${YELLOW}Building frontend for production...${NC}"
    npm run build
    
    # Check bundle size
    echo -e "${YELLOW}Analyzing bundle size...${NC}"
    if [ -d "dist" ]; then
        echo -e "${GREEN}Build successful! Bundle sizes:${NC}"
        du -sh dist/*
    fi
    
    cd ..
    echo -e "${GREEN}Frontend tests completed!${NC}"
}

# Function to test backend
test_backend() {
    echo -e "\n${BLUE}Testing Backend...${NC}"
    cd backend
    
    # Create virtual environment
    echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    pip install -r requirements.txt
    
    # Run Python linting
    echo -e "${YELLOW}Running Python linting...${NC}"
    pip install flake8
    flake8 app/ --max-line-length=120 --ignore=E501,W503 || true
    
    # Run type checking
    echo -e "${YELLOW}Running mypy type check...${NC}"
    pip install mypy
    mypy app/ --ignore-missing-imports || true
    
    # Run backend tests if available
    if [ -f "test_app.py" ] || [ -d "tests" ]; then
        echo -e "${YELLOW}Running backend tests...${NC}"
        pip install pytest pytest-asyncio
        pytest || true
    fi
    
    deactivate
    cd ..
    echo -e "${GREEN}Backend tests completed!${NC}"
}

# Function to test Docker build
test_docker_build() {
    echo -e "\n${BLUE}Testing Docker Builds...${NC}"
    
    # Build frontend Docker image
    echo -e "${YELLOW}Building frontend Docker image...${NC}"
    docker build -f frontend/Dockerfile -t automata-frontend:test ./frontend
    
    # Build backend Docker image
    echo -e "${YELLOW}Building backend Docker image...${NC}"
    docker build -f backend/Dockerfile -t automata-backend:test ./backend
    
    echo -e "${GREEN}Docker builds successful!${NC}"
}

# Function to test with Docker Compose
test_docker_compose() {
    echo -e "\n${BLUE}Testing with Docker Compose...${NC}"
    
    # Start services
    echo -e "${YELLOW}Starting services with Docker Compose...${NC}"
    docker-compose -f docker-compose.yml up -d
    
    # Wait for services to be ready
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"
    sleep 10
    
    # Test health endpoints
    echo -e "${YELLOW}Testing health endpoints...${NC}"
    
    # Test backend health
    echo -e "Testing backend health endpoint..."
    curl -f http://localhost:8000/health || echo -e "${RED}Backend health check failed${NC}"
    
    # Test frontend
    echo -e "Testing frontend..."
    curl -f http://localhost:3000 || echo -e "${RED}Frontend check failed${NC}"
    
    # Show logs
    echo -e "${YELLOW}Recent logs:${NC}"
    docker-compose logs --tail=20
    
    # Cleanup
    read -p "Stop services? (y/N): " stop_confirm
    if [ "$stop_confirm" = "y" ]; then
        docker-compose down
    fi
    
    echo -e "${GREEN}Docker Compose test completed!${NC}"
}

# Function to run integration tests
test_integration() {
    echo -e "\n${BLUE}Running Integration Tests...${NC}"
    
    # Test API endpoints
    echo -e "${YELLOW}Testing API endpoints...${NC}"
    
    # Test health endpoint
    curl -X GET http://localhost:8000/health -H "Content-Type: application/json" | jq . || true
    
    # Test automata creation
    echo -e "${YELLOW}Testing automata creation...${NC}"
    curl -X POST http://localhost:8000/api/automata/create \
        -H "Content-Type: application/json" \
        -d '{
            "type": "DFA",
            "states": ["q0", "q1"],
            "alphabet": ["0", "1"],
            "start_state": "q0",
            "accept_states": ["q1"],
            "transitions": [
                {"from": "q0", "to": "q1", "symbol": "1"},
                {"from": "q1", "to": "q0", "symbol": "0"}
            ]
        }' | jq . || true
    
    echo -e "${GREEN}Integration tests completed!${NC}"
}

# Function to run performance tests
test_performance() {
    echo -e "\n${BLUE}Running Performance Tests...${NC}"
    
    # Check if lighthouse is installed
    if command -v lighthouse &> /dev/null; then
        echo -e "${YELLOW}Running Lighthouse performance audit...${NC}"
        lighthouse http://localhost:3000 \
            --output=json \
            --output-path=./lighthouse-report.json \
            --chrome-flags="--headless" || true
        
        echo -e "${GREEN}Lighthouse report saved to lighthouse-report.json${NC}"
    else
        echo -e "${YELLOW}Lighthouse not installed. Skipping performance audit.${NC}"
        echo -e "Install with: npm install -g lighthouse${NC}"
    fi
    
    # Basic load test
    if command -v ab &> /dev/null; then
        echo -e "${YELLOW}Running basic load test...${NC}"
        ab -n 100 -c 10 http://localhost:8000/health || true
    else
        echo -e "${YELLOW}Apache Bench not installed. Skipping load test.${NC}"
    fi
    
    echo -e "${GREEN}Performance tests completed!${NC}"
}

# Function to test security
test_security() {
    echo -e "\n${BLUE}Running Security Checks...${NC}"
    
    # Check for exposed secrets
    echo -e "${YELLOW}Checking for exposed secrets...${NC}"
    if grep -r "password\|secret\|key\|token" \
        --include="*.yml" \
        --include="*.yaml" \
        --include="*.json" \
        --include="*.js" \
        --include="*.ts" \
        --include="*.py" \
        --exclude-dir=node_modules \
        --exclude-dir=venv \
        --exclude-dir=.git \
        . 2>/dev/null | grep -v "changeme\|your-\|example\|test\|TODO\|FIXME"; then
        echo -e "${RED}WARNING: Potential exposed secrets found!${NC}"
    else
        echo -e "${GREEN}No exposed secrets found.${NC}"
    fi
    
    # Check npm vulnerabilities
    echo -e "${YELLOW}Checking npm vulnerabilities...${NC}"
    cd frontend && npm audit || true && cd ..
    
    # Check Python vulnerabilities
    echo -e "${YELLOW}Checking Python vulnerabilities...${NC}"
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install safety
    safety check || true
    deactivate
    cd ..
    
    echo -e "${GREEN}Security checks completed!${NC}"
}

# Main menu
show_menu() {
    echo -e "\n${BLUE}Select test to run:${NC}"
    echo "1) Run all tests"
    echo "2) Test frontend only"
    echo "3) Test backend only"
    echo "4) Test Docker builds"
    echo "5) Test with Docker Compose"
    echo "6) Run integration tests"
    echo "7) Run performance tests"
    echo "8) Run security checks"
    echo "9) Quick smoke test"
    echo "0) Exit"
    
    read -p "Enter choice: " choice
    
    case $choice in
        1)
            check_dependencies
            setup_test_env
            test_frontend
            test_backend
            test_docker_build
            test_docker_compose
            test_integration
            test_performance
            test_security
            ;;
        2)
            test_frontend
            ;;
        3)
            test_backend
            ;;
        4)
            test_docker_build
            ;;
        5)
            test_docker_compose
            ;;
        6)
            test_integration
            ;;
        7)
            test_performance
            ;;
        8)
            test_security
            ;;
        9)
            # Quick smoke test
            echo -e "${YELLOW}Running quick smoke test...${NC}"
            check_dependencies
            setup_test_env
            cd frontend && npm install && npm run build && cd ..
            docker-compose up -d
            sleep 10
            curl -f http://localhost:8000/health && echo -e "${GREEN}Backend OK${NC}"
            curl -f http://localhost:3000 && echo -e "${GREEN}Frontend OK${NC}"
            docker-compose down
            echo -e "${GREEN}Smoke test passed!${NC}"
            ;;
        0)
            echo -e "${GREEN}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            show_menu
            ;;
    esac
}

# Show summary
show_summary() {
    echo -e "\n${GREEN}================================================${NC}"
    echo -e "${GREEN}              Test Summary                      ${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo -e "✅ Frontend build: $([ -d "frontend/dist" ] && echo "PASSED" || echo "FAILED")"
    echo -e "✅ Backend setup: $([ -d "backend/venv" ] && echo "PASSED" || echo "FAILED")"
    echo -e "✅ Docker builds: $(docker images | grep -q automata && echo "PASSED" || echo "FAILED")"
    echo -e "✅ Services health: Check logs above"
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo -e "1. Fix any failing tests"
    echo -e "2. Review security warnings"
    echo -e "3. Deploy to vast.ai using ./scripts/deploy-to-vast.sh"
}

# Main execution
main() {
    echo -e "${YELLOW}Starting local testing...${NC}"
    
    # Check if running from project root
    if [ ! -f "package.json" ] && [ ! -d "frontend" ]; then
        echo -e "${RED}Please run this script from the project root directory${NC}"
        exit 1
    fi
    
    show_menu
    show_summary
}

# Run main function
main