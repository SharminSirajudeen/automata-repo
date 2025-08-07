#!/bin/bash

# Production Startup Script for Automata-Repo
# Comprehensive health checks, dependency verification, and error handling
# Optimized for Valkey and Ollama deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# Environment settings
ENVIRONMENT=${ENVIRONMENT:-production}
LOG_LEVEL=${LOG_LEVEL:-INFO}
MAX_STARTUP_TIME=${MAX_STARTUP_TIME:-300}  # 5 minutes
HEALTH_CHECK_INTERVAL=${HEALTH_CHECK_INTERVAL:-5}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-60}

# Service endpoints
VALKEY_HOST=${VALKEY_HOST:-localhost}
VALKEY_PORT=${VALKEY_PORT:-6379}
POSTGRES_HOST=${POSTGRES_HOST:-localhost}
POSTGRES_PORT=${POSTGRES_PORT:-5432}
OLLAMA_HOST=${OLLAMA_HOST:-localhost}
OLLAMA_PORT=${OLLAMA_PORT:-11434}

# Application settings
BACKEND_HOST=${BACKEND_HOST:-0.0.0.0}
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_HOST=${FRONTEND_HOST:-0.0.0.0}
FRONTEND_PORT=${FRONTEND_PORT:-3000}

# Process tracking
BACKEND_PID=""
FRONTEND_PID=""
CLEANUP_DONE=false

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PROJECT_ROOT/startup.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PROJECT_ROOT/startup.log"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PROJECT_ROOT/startup.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PROJECT_ROOT/startup.log"
}

log_debug() {
    if [[ "$LOG_LEVEL" == "DEBUG" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PROJECT_ROOT/startup.log"
    fi
}

# Cleanup function
cleanup() {
    if [[ "$CLEANUP_DONE" == "true" ]]; then
        return
    fi
    
    log_info "Performing cleanup..."
    CLEANUP_DONE=true
    
    # Stop backend
    if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        log_info "Stopping backend process (PID: $BACKEND_PID)"
        kill -TERM "$BACKEND_PID" 2>/dev/null || true
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 "$BACKEND_PID" 2>/dev/null; then
            log_warning "Force killing backend process"
            kill -KILL "$BACKEND_PID" 2>/dev/null || true
        fi
    fi
    
    # Stop frontend
    if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        log_info "Stopping frontend process (PID: $FRONTEND_PID)"
        kill -TERM "$FRONTEND_PID" 2>/dev/null || true
        
        # Wait for graceful shutdown
        for i in {1..5}; do
            if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 "$FRONTEND_PID" 2>/dev/null; then
            log_warning "Force killing frontend process"
            kill -KILL "$FRONTEND_PID" 2>/dev/null || true
        fi
    fi
    
    log_info "Cleanup completed"
}

# Set up signal handlers
trap cleanup EXIT TERM INT

# Error handler
error_exit() {
    log_error "$1"
    cleanup
    exit 1
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check available memory
    if command_exists free; then
        local available_memory_kb
        available_memory_kb=$(free -k | awk '/^Mem:/{print $7}')
        local available_memory_gb=$((available_memory_kb / 1024 / 1024))
        
        if [[ $available_memory_gb -lt 2 ]]; then
            log_warning "Low available memory: ${available_memory_gb}GB (recommended: 4GB+)"
        else
            log_debug "Available memory: ${available_memory_gb}GB"
        fi
    fi
    
    # Check disk space
    local free_space_gb
    free_space_gb=$(df "$PROJECT_ROOT" | awk 'NR==2 {print int($4/1024/1024)}')
    if [[ $free_space_gb -lt 1 ]]; then
        error_exit "Insufficient disk space: ${free_space_gb}GB (minimum: 1GB)"
    fi
    log_debug "Free disk space: ${free_space_gb}GB"
    
    # Check CPU cores
    local cpu_cores
    cpu_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
    log_debug "CPU cores: $cpu_cores"
    
    log_success "System requirements check passed"
}

# Check network connectivity
check_network() {
    log_info "Checking network connectivity..."
    
    # Test DNS resolution
    if ! nslookup google.com >/dev/null 2>&1; then
        log_warning "DNS resolution test failed"
        return 1
    fi
    
    # Test internet connectivity
    if ! curl -s --connect-timeout 5 http://httpbin.org/status/200 >/dev/null; then
        log_warning "Internet connectivity test failed"
        return 1
    fi
    
    log_success "Network connectivity check passed"
    return 0
}

# Check if port is available
check_port_available() {
    local host="$1"
    local port="$2"
    local service_name="$3"
    
    if command_exists nc; then
        if nc -z "$host" "$port" 2>/dev/null; then
            return 0  # Port is open
        fi
    elif command_exists telnet; then
        if echo "" | timeout 3 telnet "$host" "$port" 2>/dev/null | grep -q "Connected"; then
            return 0  # Port is open
        fi
    else
        # Fallback using /dev/tcp (bash built-in)
        if timeout 3 bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null; then
            return 0  # Port is open
        fi
    fi
    
    return 1  # Port is not accessible
}

# Wait for service to be ready
wait_for_service() {
    local host="$1"
    local port="$2"
    local service_name="$3"
    local timeout="${4:-60}"
    local interval="${5:-2}"
    
    log_info "Waiting for $service_name at $host:$port..."
    
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if check_port_available "$host" "$port" "$service_name"; then
            log_success "$service_name is ready at $host:$port"
            return 0
        fi
        
        sleep "$interval"
        elapsed=$((elapsed + interval))
        
        if [[ $((elapsed % 10)) -eq 0 ]]; then
            log_debug "Still waiting for $service_name... (${elapsed}s elapsed)"
        fi
    done
    
    log_error "$service_name at $host:$port is not ready after ${timeout}s"
    return 1
}

# Check PostgreSQL connection
check_postgresql() {
    log_info "Checking PostgreSQL connection..."
    
    if ! wait_for_service "$POSTGRES_HOST" "$POSTGRES_PORT" "PostgreSQL"; then
        return 1
    fi
    
    # Test actual database connection if credentials are available
    if [[ -n "${DATABASE_URL:-}" ]]; then
        log_debug "Testing database connection..."
        if command_exists psql; then
            if echo "SELECT 1;" | psql "$DATABASE_URL" >/dev/null 2>&1; then
                log_success "PostgreSQL database connection successful"
            else
                log_warning "PostgreSQL port open but database connection failed"
                return 1
            fi
        fi
    fi
    
    return 0
}

# Check Valkey connection
check_valkey() {
    log_info "Checking Valkey connection..."
    
    if ! wait_for_service "$VALKEY_HOST" "$VALKEY_PORT" "Valkey"; then
        return 1
    fi
    
    # Test actual Valkey commands if available
    if command_exists redis-cli || command_exists valkey-cli; then
        local cli_cmd="redis-cli"
        if command_exists valkey-cli; then
            cli_cmd="valkey-cli"
        fi
        
        log_debug "Testing Valkey commands..."
        if echo "PING" | $cli_cmd -h "$VALKEY_HOST" -p "$VALKEY_PORT" 2>/dev/null | grep -q "PONG"; then
            log_success "Valkey connection and commands working"
        else
            log_warning "Valkey port open but commands failed"
            return 1
        fi
    fi
    
    return 0
}

# Check Ollama connection and models
check_ollama() {
    log_info "Checking Ollama connection..."
    
    if ! wait_for_service "$OLLAMA_HOST" "$OLLAMA_PORT" "Ollama"; then
        return 1
    fi
    
    # Check Ollama API
    local ollama_url="http://$OLLAMA_HOST:$OLLAMA_PORT"
    if ! curl -s --connect-timeout 10 "$ollama_url/api/tags" >/dev/null; then
        log_error "Ollama API not responding"
        return 1
    fi
    
    log_debug "Checking available models..."
    local models_response
    models_response=$(curl -s --connect-timeout 10 "$ollama_url/api/tags" 2>/dev/null || echo '{"models":[]}')
    local model_count
    model_count=$(echo "$models_response" | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data.get('models', [])))" 2>/dev/null || echo "0")
    
    if [[ "$model_count" -eq 0 ]]; then
        log_warning "No Ollama models found - application may not work properly"
    else
        log_success "Ollama is ready with $model_count models available"
    fi
    
    return 0
}

# Check Python environment and dependencies
check_python_environment() {
    log_info "Checking Python environment..."
    
    cd "$BACKEND_DIR"
    
    # Check Python version
    if ! command_exists python3; then
        error_exit "Python 3 is not installed"
    fi
    
    local python_version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_debug "Python version: $python_version"
    
    # Check if virtual environment exists
    if [[ ! -d "venv" ]] && [[ ! -d ".venv" ]] && [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log_warning "No Python virtual environment detected"
    fi
    
    # Check required packages
    local missing_packages=()
    local required_packages=("fastapi" "uvicorn" "sqlalchemy" "asyncpg" "valkey" "lz4")
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log_warning "Missing Python packages: ${missing_packages[*]}"
        log_info "Installing missing packages..."
        
        if [[ -f "requirements.txt" ]]; then
            python3 -m pip install -r requirements.txt --quiet || {
                log_warning "Failed to install from requirements.txt"
            }
        fi
    fi
    
    log_success "Python environment check completed"
}

# Check Node.js environment
check_node_environment() {
    log_info "Checking Node.js environment..."
    
    cd "$FRONTEND_DIR"
    
    # Check Node.js version
    if ! command_exists node; then
        log_warning "Node.js is not installed - frontend will not be available"
        return 1
    fi
    
    local node_version
    node_version=$(node --version)
    log_debug "Node.js version: $node_version"
    
    # Check if node_modules exists
    if [[ ! -d "node_modules" ]]; then
        log_info "Installing Node.js dependencies..."
        if command_exists npm; then
            npm install --production --quiet || {
                log_warning "Failed to install Node.js dependencies"
                return 1
            }
        elif command_exists yarn; then
            yarn install --production --silent || {
                log_warning "Failed to install Node.js dependencies with yarn"
                return 1
            }
        else
            log_warning "No package manager (npm/yarn) found"
            return 1
        fi
    fi
    
    log_success "Node.js environment check completed"
    return 0
}

# Start backend service
start_backend() {
    log_info "Starting backend service..."
    
    cd "$BACKEND_DIR"
    
    # Set environment variables
    export ENVIRONMENT="$ENVIRONMENT"
    export LOG_LEVEL="$LOG_LEVEL"
    export VALKEY_URL="valkey://$VALKEY_HOST:$VALKEY_PORT"
    export OLLAMA_BASE_URL="http://$OLLAMA_HOST:$OLLAMA_PORT"
    
    # Start backend with proper logging
    local backend_log="$PROJECT_ROOT/backend.log"
    local backend_cmd="python3 -m uvicorn app.main:app --host $BACKEND_HOST --port $BACKEND_PORT --log-level info"
    
    log_debug "Backend command: $backend_cmd"
    log_debug "Backend log: $backend_log"
    
    # Start backend in background
    nohup $backend_cmd > "$backend_log" 2>&1 &
    BACKEND_PID=$!
    
    # Wait for backend to start
    local startup_timeout=60
    local elapsed=0
    log_info "Waiting for backend to start (PID: $BACKEND_PID)..."
    
    while [[ $elapsed -lt $startup_timeout ]]; do
        if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
            log_error "Backend process died during startup"
            if [[ -f "$backend_log" ]]; then
                log_error "Backend log tail:"
                tail -20 "$backend_log" | while read -r line; do
                    log_error "  $line"
                done
            fi
            return 1
        fi
        
        # Check if backend is responding
        if curl -s --connect-timeout 2 "http://$BACKEND_HOST:$BACKEND_PORT/health" >/dev/null 2>&1; then
            log_success "Backend started successfully on $BACKEND_HOST:$BACKEND_PORT"
            return 0
        fi
        
        sleep 2
        elapsed=$((elapsed + 2))
        
        if [[ $((elapsed % 10)) -eq 0 ]]; then
            log_debug "Still waiting for backend... (${elapsed}s elapsed)"
        fi
    done
    
    log_error "Backend failed to start within ${startup_timeout}s"
    return 1
}

# Start frontend service
start_frontend() {
    log_info "Starting frontend service..."
    
    cd "$FRONTEND_DIR"
    
    # Check if we can start frontend
    if ! check_node_environment; then
        log_warning "Skipping frontend startup due to Node.js environment issues"
        return 0
    fi
    
    # Set environment variables
    export VITE_API_URL="http://$BACKEND_HOST:$BACKEND_PORT"
    export NODE_ENV="production"
    
    # Build frontend if needed
    if [[ ! -d "dist" ]] && [[ -f "package.json" ]]; then
        log_info "Building frontend..."
        if command_exists npm; then
            npm run build --silent || {
                log_warning "Frontend build failed"
                return 1
            }
        fi
    fi
    
    # Start frontend
    local frontend_log="$PROJECT_ROOT/frontend.log"
    local frontend_cmd=""
    
    # Choose appropriate start command
    if [[ -d "dist" ]]; then
        # Serve built files
        if command_exists serve; then
            frontend_cmd="serve -s dist -l $FRONTEND_PORT"
        elif command_exists python3; then
            frontend_cmd="python3 -m http.server $FRONTEND_PORT --directory dist"
        else
            log_warning "No web server available to serve frontend"
            return 1
        fi
    elif [[ -f "package.json" ]]; then
        # Development server
        if command_exists npm; then
            frontend_cmd="npm start"
        elif command_exists yarn; then
            frontend_cmd="yarn start"
        else
            log_warning "No package manager available to start frontend"
            return 1
        fi
    else
        log_warning "No frontend configuration found"
        return 1
    fi
    
    log_debug "Frontend command: $frontend_cmd"
    log_debug "Frontend log: $frontend_log"
    
    # Start frontend in background
    nohup $frontend_cmd > "$frontend_log" 2>&1 &
    FRONTEND_PID=$!
    
    # Wait for frontend to start
    local startup_timeout=30
    local elapsed=0
    log_info "Waiting for frontend to start (PID: $FRONTEND_PID)..."
    
    while [[ $elapsed -lt $startup_timeout ]]; do
        if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
            log_warning "Frontend process died during startup"
            return 1
        fi
        
        # Check if frontend is responding
        if curl -s --connect-timeout 2 "http://$FRONTEND_HOST:$FRONTEND_PORT" >/dev/null 2>&1; then
            log_success "Frontend started successfully on $FRONTEND_HOST:$FRONTEND_PORT"
            return 0
        fi
        
        sleep 2
        elapsed=$((elapsed + 2))
    done
    
    log_warning "Frontend may not be fully ready yet"
    return 0  # Don't fail startup if frontend has issues
}

# Perform application health checks
perform_health_checks() {
    log_info "Performing application health checks..."
    
    # Backend health check
    if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        local health_url="http://$BACKEND_HOST:$BACKEND_PORT/health"
        if curl -s --connect-timeout 5 "$health_url" | grep -q "healthy"; then
            log_success "Backend health check passed"
        else
            log_warning "Backend health check failed"
        fi
    fi
    
    # Frontend health check
    if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        if curl -s --connect-timeout 5 "http://$FRONTEND_HOST:$FRONTEND_PORT" >/dev/null; then
            log_success "Frontend health check passed"
        else
            log_warning "Frontend health check failed"
        fi
    fi
    
    # Service integration check
    log_info "Checking service integration..."
    local api_test_url="http://$BACKEND_HOST:$BACKEND_PORT/api/health"
    if curl -s --connect-timeout 5 "$api_test_url" >/dev/null; then
        log_success "API integration check passed"
    else
        log_warning "API integration check failed"
    fi
}

# Display startup summary
display_startup_summary() {
    log_success "=== STARTUP SUMMARY ==="
    log_success "Environment: $ENVIRONMENT"
    log_success "Project Root: $PROJECT_ROOT"
    
    if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        log_success "Backend: Running (PID: $BACKEND_PID) - http://$BACKEND_HOST:$BACKEND_PORT"
    else
        log_error "Backend: Not running"
    fi
    
    if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        log_success "Frontend: Running (PID: $FRONTEND_PID) - http://$FRONTEND_HOST:$FRONTEND_PORT"
    else
        log_warning "Frontend: Not running"
    fi
    
    log_success "Logs:"
    log_success "  Startup: $PROJECT_ROOT/startup.log"
    log_success "  Backend: $PROJECT_ROOT/backend.log"
    if [[ -n "$FRONTEND_PID" ]]; then
        log_success "  Frontend: $PROJECT_ROOT/frontend.log"
    fi
    
    log_success "========================="
}

# Monitor services
monitor_services() {
    log_info "Starting service monitoring..."
    log_info "Press Ctrl+C to stop all services"
    
    while true; do
        local all_healthy=true
        
        # Check backend
        if [[ -n "$BACKEND_PID" ]]; then
            if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
                log_error "Backend process died unexpectedly"
                all_healthy=false
            fi
        fi
        
        # Check frontend
        if [[ -n "$FRONTEND_PID" ]]; then
            if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
                log_warning "Frontend process died"
                # Don't mark as unhealthy, frontend is optional
            fi
        fi
        
        if [[ "$all_healthy" == false ]]; then
            log_error "Critical services are down - initiating shutdown"
            break
        fi
        
        sleep "$HEALTH_CHECK_INTERVAL"
    done
}

# Main startup sequence
main() {
    log_info "Starting Automata-Repo in $ENVIRONMENT mode..."
    log_info "Startup log: $PROJECT_ROOT/startup.log"
    
    # Pre-flight checks
    check_system_requirements
    
    # Check network connectivity (non-critical)
    if ! check_network; then
        log_warning "Network connectivity issues detected - continuing anyway"
    fi
    
    # Check dependencies
    log_info "Checking service dependencies..."
    
    local dependency_failures=()
    
    if ! check_postgresql; then
        dependency_failures+=("PostgreSQL")
    fi
    
    if ! check_valkey; then
        dependency_failures+=("Valkey")
    fi
    
    if ! check_ollama; then
        dependency_failures+=("Ollama")
    fi
    
    if [[ ${#dependency_failures[@]} -gt 0 ]]; then
        log_error "Critical dependencies not available: ${dependency_failures[*]}"
        log_error "Please ensure all required services are running:"
        log_error "  - PostgreSQL on $POSTGRES_HOST:$POSTGRES_PORT"
        log_error "  - Valkey on $VALKEY_HOST:$VALKEY_PORT"
        log_error "  - Ollama on $OLLAMA_HOST:$OLLAMA_PORT"
        error_exit "Dependency check failed"
    fi
    
    # Check application environments
    check_python_environment
    
    # Start services
    log_info "Starting application services..."
    
    if ! start_backend; then
        error_exit "Failed to start backend service"
    fi
    
    # Frontend is optional - don't fail if it doesn't start
    start_frontend || log_warning "Frontend startup had issues but continuing..."
    
    # Perform health checks
    sleep 5  # Give services time to fully initialize
    perform_health_checks
    
    # Display summary
    display_startup_summary
    
    # Start monitoring
    monitor_services
}

# Script usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Production startup script for Automata-Repo with comprehensive health checks.

OPTIONS:
    -h, --help              Show this help message
    -e, --environment ENV   Set environment (default: production)
    -l, --log-level LEVEL   Set log level (DEBUG, INFO, WARNING, ERROR)
    -t, --timeout SECONDS   Set maximum startup time (default: 300)
    --backend-only          Start only the backend service
    --skip-frontend         Skip frontend startup
    --dry-run               Perform checks without starting services

ENVIRONMENT VARIABLES:
    ENVIRONMENT             Deployment environment
    LOG_LEVEL              Logging verbosity
    VALKEY_HOST            Valkey server host (default: localhost)
    VALKEY_PORT            Valkey server port (default: 6379)
    POSTGRES_HOST          PostgreSQL host (default: localhost)
    POSTGRES_PORT          PostgreSQL port (default: 5432)
    OLLAMA_HOST            Ollama server host (default: localhost)
    OLLAMA_PORT            Ollama server port (default: 11434)
    BACKEND_HOST           Backend bind host (default: 0.0.0.0)
    BACKEND_PORT           Backend port (default: 8000)
    FRONTEND_HOST          Frontend bind host (default: 0.0.0.0)
    FRONTEND_PORT          Frontend port (default: 3000)

EXAMPLES:
    $0                              # Start in production mode
    $0 -e development -l DEBUG      # Start in development with debug logging
    $0 --backend-only               # Start only backend service
    $0 --dry-run                    # Check dependencies without starting

EOF
}

# Parse command line arguments
BACKEND_ONLY=false
SKIP_FRONTEND=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -t|--timeout)
            MAX_STARTUP_TIME="$2"
            shift 2
            ;;
        --backend-only)
            BACKEND_ONLY=true
            shift
            ;;
        --skip-frontend)
            SKIP_FRONTEND=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    log_warning "Invalid environment: $ENVIRONMENT (using production)"
    ENVIRONMENT="production"
fi

if [[ ! "$LOG_LEVEL" =~ ^(DEBUG|INFO|WARNING|ERROR)$ ]]; then
    log_warning "Invalid log level: $LOG_LEVEL (using INFO)"
    LOG_LEVEL="INFO"
fi

# Override frontend settings based on flags
if [[ "$BACKEND_ONLY" == true ]] || [[ "$SKIP_FRONTEND" == true ]]; then
    start_frontend() {
        log_info "Frontend startup skipped (--backend-only or --skip-frontend)"
        return 0
    }
fi

# Run dry-run if requested
if [[ "$DRY_RUN" == true ]]; then
    log_info "Performing dry-run checks..."
    check_system_requirements
    check_network || true
    check_postgresql || log_warning "PostgreSQL check failed"
    check_valkey || log_warning "Valkey check failed"
    check_ollama || log_warning "Ollama check failed"
    check_python_environment
    check_node_environment || true
    log_success "Dry-run completed"
    exit 0
fi

# Run main startup sequence
main