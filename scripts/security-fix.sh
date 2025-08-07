#!/bin/bash

# Security Fix Script for Automata Learning Platform
# This script helps users secure their deployment by generating secrets,
# setting up proper permissions, and validating configurations.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS_DIR="${PROJECT_ROOT}/secrets"
ENV_FILE="${PROJECT_ROOT}/.env"
ENV_EXAMPLE_FILE="${PROJECT_ROOT}/.env.example"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Generate secure random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 $length | tr -d "=+/" | cut -c1-$length
}

# Generate JWT secret (256-bit minimum)
generate_jwt_secret() {
    openssl rand -base64 64 | tr -d "=+/" | cut -c1-64
}

# Generate secret key for backend
generate_secret_key() {
    openssl rand -base64 64 | tr -d "=+/" | cut -c1-64
}

# Create secrets directory with proper permissions
create_secrets_directory() {
    log_info "Creating secrets directory..."
    
    if [ ! -d "$SECRETS_DIR" ]; then
        mkdir -p "$SECRETS_DIR"
        log_success "Created secrets directory: $SECRETS_DIR"
    fi
    
    # Set restrictive permissions (owner read/write only)
    chmod 700 "$SECRETS_DIR"
    log_success "Set secure permissions on secrets directory"
}

# Generate secret files for Docker Compose production
generate_secret_files() {
    log_info "Generating secret files..."
    
    # Generate postgres password
    if [ ! -f "$SECRETS_DIR/postgres_password.txt" ]; then
        generate_password 32 > "$SECRETS_DIR/postgres_password.txt"
        chmod 600 "$SECRETS_DIR/postgres_password.txt"
        log_success "Generated postgres password"
    else
        log_warning "Postgres password already exists, skipping..."
    fi
    
    # Generate backend secret key
    if [ ! -f "$SECRETS_DIR/secret_key.txt" ]; then
        generate_secret_key > "$SECRETS_DIR/secret_key.txt"
        chmod 600 "$SECRETS_DIR/secret_key.txt"
        log_success "Generated backend secret key"
    else
        log_warning "Backend secret key already exists, skipping..."
    fi
    
    # Generate OpenAI API key placeholder (user must fill this)
    if [ ! -f "$SECRETS_DIR/openai_api_key.txt" ]; then
        echo "your-openai-api-key-here" > "$SECRETS_DIR/openai_api_key.txt"
        chmod 600 "$SECRETS_DIR/openai_api_key.txt"
        log_success "Created OpenAI API key placeholder"
        log_warning "Please update $SECRETS_DIR/openai_api_key.txt with your actual OpenAI API key"
    else
        log_warning "OpenAI API key file already exists, skipping..."
    fi
    
    # Generate Grafana password
    if [ ! -f "$SECRETS_DIR/grafana_password.txt" ]; then
        generate_password 24 > "$SECRETS_DIR/grafana_password.txt"
        chmod 600 "$SECRETS_DIR/grafana_password.txt"
        log_success "Generated Grafana password"
    else
        log_warning "Grafana password already exists, skipping..."
    fi
}

# Create .env file from example if it doesn't exist
create_env_file() {
    log_info "Creating environment file..."
    
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "$ENV_EXAMPLE_FILE" ]; then
            cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"
            log_success "Created .env file from .env.example"
            
            # Replace placeholder values with generated secrets
            local postgres_password=$(cat "$SECRETS_DIR/postgres_password.txt" 2>/dev/null || generate_password 32)
            local secret_key=$(cat "$SECRETS_DIR/secret_key.txt" 2>/dev/null || generate_secret_key)
            local jwt_secret=$(generate_jwt_secret)
            
            # Use sed to replace placeholder values
            sed -i.bak "s/your_secure_postgres_password_here/$postgres_password/g" "$ENV_FILE"
            sed -i.bak "s/your_secure_supabase_password_here/$(generate_password 32)/g" "$ENV_FILE"
            sed -i.bak "s/your-super-secret-jwt-token-with-minimum-256-bits/$jwt_secret/g" "$ENV_FILE"
            sed -i.bak "s/your-backend-secret-key-minimum-256-bits/$secret_key/g" "$ENV_FILE"
            
            # Remove backup file
            rm -f "$ENV_FILE.bak"
            
            log_success "Updated .env file with generated secrets"
        else
            log_error ".env.example file not found!"
            exit 1
        fi
    else
        log_warning ".env file already exists, skipping..."
    fi
    
    # Set secure permissions on .env file
    chmod 600 "$ENV_FILE"
    log_success "Set secure permissions on .env file"
}

# Validate environment variables
validate_env_vars() {
    log_info "Validating environment variables..."
    
    if [ ! -f "$ENV_FILE" ]; then
        log_error ".env file not found! Run with --setup first."
        exit 1
    fi
    
    # Source the .env file
    set -a
    source "$ENV_FILE"
    set +a
    
    local errors=0
    
    # Check critical variables
    if [[ -z "${POSTGRES_PASSWORD:-}" ]] || [[ "$POSTGRES_PASSWORD" == "your_secure_postgres_password_here" ]]; then
        log_error "POSTGRES_PASSWORD is not set or using default value"
        ((errors++))
    fi
    
    if [[ -z "${SECRET_KEY:-}" ]] || [[ "$SECRET_KEY" == "your-backend-secret-key-minimum-256-bits" ]]; then
        log_error "SECRET_KEY is not set or using default value"
        ((errors++))
    fi
    
    if [[ -z "${GOTRUE_JWT_SECRET:-}" ]] || [[ "$GOTRUE_JWT_SECRET" == "your-super-secret-jwt-token-with-minimum-256-bits" ]]; then
        log_error "GOTRUE_JWT_SECRET is not set or using default value"
        ((errors++))
    fi
    
    # Check password strength (minimum 16 characters for production)
    if [[ ${#POSTGRES_PASSWORD} -lt 16 ]]; then
        log_warning "POSTGRES_PASSWORD should be at least 16 characters for production"
    fi
    
    if [[ ${#SECRET_KEY} -lt 32 ]]; then
        log_warning "SECRET_KEY should be at least 32 characters for production"
    fi
    
    if [[ ${#GOTRUE_JWT_SECRET} -lt 32 ]]; then
        log_warning "GOTRUE_JWT_SECRET should be at least 32 characters for production"
    fi
    
    if [ $errors -eq 0 ]; then
        log_success "Environment variables validation passed"
    else
        log_error "Found $errors critical environment variable issues"
        exit 1
    fi
}

# Check Docker Compose files for security issues
validate_docker_compose() {
    log_info "Validating Docker Compose configuration..."
    
    local compose_file="${PROJECT_ROOT}/docker-compose.yml"
    local compose_prod_file="${PROJECT_ROOT}/docker-compose.prod.yml"
    
    if [ -f "$compose_file" ]; then
        # Check for hardcoded passwords
        if grep -q "password.*:" "$compose_file" && ! grep -q "\${" "$compose_file"; then
            log_warning "docker-compose.yml may contain hardcoded passwords"
        fi
        
        # Check for insecure configurations
        if grep -q "privileged.*true" "$compose_file"; then
            log_error "Found privileged containers in docker-compose.yml"
        fi
        
        log_success "Docker Compose development file validation completed"
    fi
    
    if [ -f "$compose_prod_file" ]; then
        # Check for secrets usage in production
        if ! grep -q "secrets:" "$compose_prod_file"; then
            log_warning "Production Docker Compose file should use secrets"
        fi
        
        log_success "Docker Compose production file validation completed"
    fi
}

# Set up SSL/TLS certificates directory
setup_ssl() {
    log_info "Setting up SSL directory structure..."
    
    local ssl_dir="${PROJECT_ROOT}/docker/nginx-ssl"
    
    if [ ! -d "$ssl_dir" ]; then
        mkdir -p "$ssl_dir"
        chmod 700 "$ssl_dir"
        log_success "Created SSL directory: $ssl_dir"
    fi
    
    # Create self-signed certificate for development/testing
    if [ ! -f "$ssl_dir/cert.pem" ] || [ ! -f "$ssl_dir/key.pem" ]; then
        log_info "Generating self-signed SSL certificate for development..."
        
        openssl req -x509 -newkey rsa:4096 -keyout "$ssl_dir/key.pem" -out "$ssl_dir/cert.pem" \
            -days 365 -nodes -subj "/C=US/ST=Dev/L=Local/O=Automata/CN=localhost" 2>/dev/null
        
        chmod 600 "$ssl_dir/key.pem"
        chmod 644 "$ssl_dir/cert.pem"
        
        log_success "Generated self-signed SSL certificate"
        log_warning "For production, replace with certificates from a trusted CA"
    else
        log_warning "SSL certificates already exist, skipping..."
    fi
}

# Audit file permissions
audit_permissions() {
    log_info "Auditing file permissions..."
    
    local issues=0
    
    # Check .env file permissions
    if [ -f "$ENV_FILE" ]; then
        local env_perms=$(stat -c "%a" "$ENV_FILE" 2>/dev/null || stat -f "%A" "$ENV_FILE" 2>/dev/null || echo "unknown")
        if [[ "$env_perms" != "600" ]]; then
            log_warning ".env file has permissions $env_perms, should be 600"
            chmod 600 "$ENV_FILE"
            log_success "Fixed .env file permissions"
        fi
    fi
    
    # Check secrets directory permissions
    if [ -d "$SECRETS_DIR" ]; then
        local secrets_perms=$(stat -c "%a" "$SECRETS_DIR" 2>/dev/null || stat -f "%A" "$SECRETS_DIR" 2>/dev/null || echo "unknown")
        if [[ "$secrets_perms" != "700" ]]; then
            log_warning "Secrets directory has permissions $secrets_perms, should be 700"
            chmod 700 "$SECRETS_DIR"
            log_success "Fixed secrets directory permissions"
        fi
        
        # Check individual secret files
        for secret_file in "$SECRETS_DIR"/*.txt; do
            if [ -f "$secret_file" ]; then
                local file_perms=$(stat -c "%a" "$secret_file" 2>/dev/null || stat -f "%A" "$secret_file" 2>/dev/null || echo "unknown")
                if [[ "$file_perms" != "600" ]]; then
                    log_warning "$(basename $secret_file) has permissions $file_perms, should be 600"
                    chmod 600 "$secret_file"
                    log_success "Fixed $(basename $secret_file) permissions"
                fi
            fi
        done
    fi
    
    log_success "File permissions audit completed"
}

# Create gitignore entries for security
update_gitignore() {
    log_info "Updating .gitignore for security..."
    
    local gitignore_file="${PROJECT_ROOT}/.gitignore"
    local security_entries=(
        ".env"
        ".env.local"
        ".env.production"
        "secrets/"
        "*.key"
        "*.pem"
        "*.crt"
        "*.p12"
        "*.pfx"
    )
    
    for entry in "${security_entries[@]}"; do
        if ! grep -q "^${entry}$" "$gitignore_file" 2>/dev/null; then
            echo "$entry" >> "$gitignore_file"
            log_success "Added $entry to .gitignore"
        fi
    done
}

# Main setup function
setup_security() {
    log_info "Starting security setup..."
    
    create_secrets_directory
    generate_secret_files
    create_env_file
    setup_ssl
    update_gitignore
    audit_permissions
    
    log_success "Security setup completed!"
    
    # Display important information
    echo ""
    log_info "IMPORTANT SECURITY NOTES:"
    echo "1. Your secrets are stored in: $SECRETS_DIR"
    echo "2. Your environment file is: $ENV_FILE"
    echo "3. Please update the OpenAI API key in: $SECRETS_DIR/openai_api_key.txt"
    echo "4. For production, replace self-signed certificates with trusted CA certificates"
    echo "5. Never commit .env files or secrets/ directory to version control"
    echo ""
    log_warning "Run '$0 --validate' to check your configuration before deployment"
}

# Validate all security configurations
validate_security() {
    log_info "Starting security validation..."
    
    validate_env_vars
    validate_docker_compose
    audit_permissions
    
    log_success "Security validation completed!"
}

# Show help information
show_help() {
    echo "Security Fix Script for Automata Learning Platform"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --setup       Set up secure deployment (generate secrets, create .env, etc.)"
    echo "  --validate    Validate current security configuration"
    echo "  --audit       Audit file permissions and security settings"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --setup      # Initial security setup"
    echo "  $0 --validate   # Validate before deployment"
    echo "  $0 --audit      # Check permissions and settings"
    echo ""
}

# Main script logic
main() {
    case "${1:-}" in
        --setup)
            setup_security
            ;;
        --validate)
            validate_security
            ;;
        --audit)
            audit_permissions
            ;;
        --help)
            show_help
            ;;
        *)
            log_error "Unknown option: ${1:-}"
            show_help
            exit 1
            ;;
    esac
}

# Check if running as root (not recommended for development)
if [[ $EUID -eq 0 ]]; then
    log_warning "Running as root is not recommended for development"
    log_warning "Consider running as a non-root user for better security"
fi

# Check for required tools
command -v openssl >/dev/null 2>&1 || { log_error "openssl is required but not installed. Aborting."; exit 1; }

# Run main function with all arguments
main "$@"