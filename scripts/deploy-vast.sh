#!/bin/bash
set -euo pipefail

# Automata Theory Platform Deployment Script for vast.ai
# Optimized for GPU instances with comprehensive monitoring and scaling

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables
NAMESPACE="automata-app"
RELEASE_NAME="automata"
DOCKER_REGISTRY="registry.vast.ai"
IMAGE_TAG="${1:-latest}"
ENVIRONMENT="${2:-production}"
VAST_AI_INSTANCE_ID="${VAST_AI_INSTANCE_ID:-}"
DOMAIN="${DOMAIN:-automata.vast.ai}"

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if running on vast.ai
    if [[ -z "${VAST_AI_INSTANCE_ID}" ]]; then
        warning "VAST_AI_INSTANCE_ID not set. Assuming local development."
    fi
    
    # Check required tools
    for tool in kubectl docker helm; do
        if ! command -v $tool &> /dev/null; then
            error "$tool is required but not installed"
        fi
    done
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    success "Prerequisites check passed"
}

# Setup vast.ai specific configurations
setup_vast_ai() {
    log "Setting up vast.ai specific configurations..."
    
    # Get instance information
    if [[ -n "${VAST_AI_INSTANCE_ID}" ]]; then
        log "Configuring for vast.ai instance: ${VAST_AI_INSTANCE_ID}"
        
        # Get GPU information
        GPU_COUNT=$(nvidia-smi -L | wc -l || echo "0")
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 || echo "0")
        
        log "Detected ${GPU_COUNT} GPU(s) with ${GPU_MEMORY}MB memory each"
        
        # Update configurations based on GPU availability
        if [[ ${GPU_COUNT} -gt 0 ]]; then
            export ENABLE_GPU_ACCELERATION=true
            export OLLAMA_GPU_LAYERS=35
        fi
        
        # Get instance storage information
        STORAGE_SIZE=$(df -h / | awk 'NR==2 {print $2}')
        log "Available storage: ${STORAGE_SIZE}"
    fi
}

# Build and push Docker images
build_images() {
    log "Building Docker images..."
    
    # Build backend image
    log "Building backend image..."
    docker build -f docker/Dockerfile.backend \
        --target production \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        -t ${DOCKER_REGISTRY}/${RELEASE_NAME}-backend:${IMAGE_TAG} \
        ./backend/
    
    # Build frontend image
    log "Building frontend image..."
    docker build -f docker/Dockerfile.frontend \
        --target production \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --build-arg VITE_API_URL="https://api.${DOMAIN}" \
        --build-arg VITE_WS_URL="wss://api.${DOMAIN}/ws" \
        -t ${DOCKER_REGISTRY}/${RELEASE_NAME}-frontend:${IMAGE_TAG} \
        ./frontend/
    
    success "Images built successfully"
}

# Push images to registry
push_images() {
    log "Pushing images to registry..."
    
    # Login to registry if credentials available
    if [[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]]; then
        echo "${DOCKER_PASSWORD}" | docker login ${DOCKER_REGISTRY} -u "${DOCKER_USERNAME}" --password-stdin
    fi
    
    docker push ${DOCKER_REGISTRY}/${RELEASE_NAME}-backend:${IMAGE_TAG}
    docker push ${DOCKER_REGISTRY}/${RELEASE_NAME}-frontend:${IMAGE_TAG}
    
    success "Images pushed successfully"
}

# Create namespace and basic resources
create_namespace() {
    log "Creating namespace and basic resources..."
    
    kubectl apply -f k8s/namespace.yaml
    
    # Wait for namespace to be ready
    kubectl wait --for=condition=Ready namespace/${NAMESPACE} --timeout=30s
    
    success "Namespace created"
}

# Deploy secrets
deploy_secrets() {
    log "Deploying secrets..."
    
    # Check if secrets exist
    if kubectl get secret automata-secrets -n ${NAMESPACE} &> /dev/null; then
        warning "Secrets already exist, skipping creation"
        return
    fi
    
    # Validate required environment variables for secrets
    required_vars=(\n        "POSTGRES_PASSWORD"\n        "SECRET_KEY"\n        "JWT_SECRET"\n        "OPENAI_API_KEY"\n    )\n    \n    for var in "${required_vars[@]}"; do\n        if [[ -z "${!var:-}" ]]; then\n            error "Required environment variable ${var} is not set"\n        fi\n    done\n    \n    # Create secrets from environment variables\n    kubectl create secret generic automata-secrets \\\n        --namespace=${NAMESPACE} \\\n        --from-literal=postgres-password="${POSTGRES_PASSWORD}" \\\n        --from-literal=redis-password="${REDIS_PASSWORD:-}" \\\n        --from-literal=secret-key="${SECRET_KEY}" \\\n        --from-literal=jwt-secret="${JWT_SECRET}" \\\n        --from-literal=openai-api-key="${OPENAI_API_KEY}" \\\n        --from-literal=database-url="postgresql://automata:${POSTGRES_PASSWORD}@automata-postgres:5432/automata_db"\n    \n    success "Secrets deployed"\n}\n\n# Deploy persistent volumes\ndeploy_storage() {\n    log "Deploying storage resources..."\n    \n    kubectl apply -f k8s/pvc.yaml\n    \n    # Wait for PVCs to be bound\n    log "Waiting for PVCs to be bound..."\n    kubectl wait --for=condition=Bound pvc --all -n ${NAMESPACE} --timeout=300s\n    \n    success "Storage deployed"\n}\n\n# Deploy applications\ndeploy_applications() {\n    log "Deploying applications..."\n    \n    # Update image tags in deployments\n    sed -i.bak "s|image: automata-backend:latest|image: ${DOCKER_REGISTRY}/${RELEASE_NAME}-backend:${IMAGE_TAG}|g" k8s/backend-deployment.yaml\n    sed -i.bak "s|image: automata-frontend:latest|image: ${DOCKER_REGISTRY}/${RELEASE_NAME}-frontend:${IMAGE_TAG}|g" k8s/frontend-deployment.yaml\n    \n    # Deploy in order: database -> cache -> backend -> frontend\n    kubectl apply -f k8s/postgres-deployment.yaml\n    kubectl apply -f k8s/redis-deployment.yaml\n    \n    # Wait for database to be ready\n    log "Waiting for database to be ready..."\n    kubectl wait --for=condition=Ready pod -l app=automata-postgres -n ${NAMESPACE} --timeout=300s\n    \n    # Wait for Redis to be ready\n    log "Waiting for Redis to be ready..."\n    kubectl wait --for=condition=Ready pod -l app=automata-redis -n ${NAMESPACE} --timeout=180s\n    \n    # Deploy ConfigMaps\n    kubectl apply -f k8s/configmap.yaml\n    \n    # Deploy backend\n    kubectl apply -f k8s/backend-deployment.yaml\n    \n    # Wait for backend to be ready\n    log "Waiting for backend to be ready..."\n    kubectl wait --for=condition=Ready pod -l app=automata-backend -n ${NAMESPACE} --timeout=600s\n    \n    # Deploy frontend\n    kubectl apply -f k8s/frontend-deployment.yaml\n    \n    # Wait for frontend to be ready\n    log "Waiting for frontend to be ready..."\n    kubectl wait --for=condition=Ready pod -l app=automata-frontend -n ${NAMESPACE} --timeout=300s\n    \n    success "Applications deployed"\n}\n\n# Deploy networking\ndeploy_networking() {\n    log "Deploying networking resources..."\n    \n    # Update domain in ingress\n    sed -i.bak "s|automata.vast.ai|${DOMAIN}|g" k8s/ingress.yaml\n    \n    kubectl apply -f k8s/ingress.yaml\n    \n    success "Networking deployed"\n}\n\n# Deploy autoscaling\ndeploy_autoscaling() {\n    log "Deploying autoscaling resources..."\n    \n    kubectl apply -f k8s/hpa.yaml\n    \n    success "Autoscaling deployed"\n}\n\n# Setup monitoring\nsetup_monitoring() {\n    log "Setting up monitoring..."\n    \n    # Deploy monitoring stack using Helm\n    if helm list -n monitoring | grep -q prometheus; then\n        warning "Prometheus already installed, upgrading..."\n        helm upgrade prometheus prometheus-community/kube-prometheus-stack -n monitoring\n    else\n        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts\n        helm repo update\n        \n        kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -\n        \n        helm install prometheus prometheus-community/kube-prometheus-stack \\\n            --namespace monitoring \\\n            --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \\\n            --set prometheus.prometheusSpec.retention=15d \\\n            --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi\n    fi\n    \n    success "Monitoring setup complete"\n}\n\n# Verify deployment\nverify_deployment() {\n    log "Verifying deployment..."\n    \n    # Check pod status\n    kubectl get pods -n ${NAMESPACE}\n    \n    # Check service endpoints\n    kubectl get svc -n ${NAMESPACE}\n    \n    # Check ingress\n    kubectl get ingress -n ${NAMESPACE}\n    \n    # Perform health checks\n    log "Performing health checks..."\n    \n    # Wait for all deployments to be ready\n    kubectl wait --for=condition=Available deployment --all -n ${NAMESPACE} --timeout=600s\n    \n    # Check if we can reach the application\n    if command -v curl &> /dev/null; then\n        sleep 30  # Give ingress time to configure\n        \n        if curl -f -s "https://${DOMAIN}/health" > /dev/null; then\n            success "Application is responding to health checks"\n        else\n            warning "Application health check failed, but deployment completed"\n        fi\n    fi\n    \n    success "Deployment verification complete"\n}\n\n# Cleanup function for rollback\ncleanup() {\n    log "Cleaning up on error..."\n    \n    # Restore original deployment files\n    if [[ -f k8s/backend-deployment.yaml.bak ]]; then\n        mv k8s/backend-deployment.yaml.bak k8s/backend-deployment.yaml\n    fi\n    if [[ -f k8s/frontend-deployment.yaml.bak ]]; then\n        mv k8s/frontend-deployment.yaml.bak k8s/frontend-deployment.yaml\n    fi\n    if [[ -f k8s/ingress.yaml.bak ]]; then\n        mv k8s/ingress.yaml.bak k8s/ingress.yaml\n    fi\n}\n\n# Main deployment function\nmain() {\n    log "Starting Automata Platform deployment on vast.ai"\n    log "Environment: ${ENVIRONMENT}"\n    log "Image Tag: ${IMAGE_TAG}"\n    log "Domain: ${DOMAIN}"\n    \n    # Set trap for cleanup\n    trap cleanup EXIT\n    \n    check_prerequisites\n    setup_vast_ai\n    \n    if [[ "${SKIP_BUILD:-false}" != "true" ]]; then\n        build_images\n        push_images\n    fi\n    \n    create_namespace\n    deploy_secrets\n    deploy_storage\n    deploy_applications\n    deploy_networking\n    deploy_autoscaling\n    \n    if [[ "${SETUP_MONITORING:-true}" == "true" ]]; then\n        setup_monitoring\n    fi\n    \n    verify_deployment\n    \n    success "Deployment completed successfully!"\n    success "Application should be available at: https://${DOMAIN}"\n    \n    log "Useful commands:"\n    log "  View logs: kubectl logs -f deployment/automata-backend -n ${NAMESPACE}"\n    log "  Scale backend: kubectl scale deployment automata-backend --replicas=4 -n ${NAMESPACE}"\n    log "  Port forward: kubectl port-forward svc/automata-frontend 8080:80 -n ${NAMESPACE}"\n    log "  Monitor: kubectl get pods -n ${NAMESPACE} -w"\n}\n\n# Run main function if script is executed directly\nif [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then\n    main "$@"\nfi