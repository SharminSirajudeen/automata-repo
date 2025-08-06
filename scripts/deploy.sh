#!/bin/bash

# Automata App Deployment Script
# This script provides various deployment options for the automata application

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE=${NAMESPACE:-automata-app}
ENVIRONMENT=${ENVIRONMENT:-production}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-docker.io}
IMAGE_TAG=${IMAGE_TAG:-latest}

# Helper functions
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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_deps=()
    
    if ! command_exists kubectl; then
        missing_deps+=("kubectl")
    fi
    
    if ! command_exists docker; then
        missing_deps+=("docker")
    fi
    
    if [[ ${#missing_deps[@]} -ne 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubectl configuration."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build and push Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build backend
    log_info "Building backend image..."
    docker build -t "${DOCKER_REGISTRY}/automata/backend:${IMAGE_TAG}" ./backend/
    docker push "${DOCKER_REGISTRY}/automata/backend:${IMAGE_TAG}"
    
    # Build frontend
    log_info "Building frontend image..."
    docker build -t "${DOCKER_REGISTRY}/automata/frontend:${IMAGE_TAG}" ./frontend/
    docker push "${DOCKER_REGISTRY}/automata/frontend:${IMAGE_TAG}"
    
    log_success "Docker images built and pushed"
}

# Deploy with kubectl (raw manifests)
deploy_kubectl() {
    log_info "Deploying with kubectl..."
    
    # Create namespace
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply manifests in order
    local manifests=(
        "k8s/namespace.yaml"
        "k8s/configmap.yaml"
        "k8s/secrets.yaml"
        "k8s/postgres.yaml"
        "k8s/redis.yaml"
        "k8s/backend.yaml"
        "k8s/frontend.yaml"
        "k8s/ingress.yaml"
        "k8s/monitoring.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        if [[ -f "$manifest" ]]; then
            log_info "Applying $manifest..."
            kubectl apply -f "$manifest"
        else
            log_warning "Manifest $manifest not found, skipping..."
        fi
    done
    
    log_success "Kubectl deployment completed"
}

# Deploy with Helm
deploy_helm() {
    log_info "Deploying with Helm..."
    
    if ! command_exists helm; then
        log_error "Helm is not installed. Please install Helm first."
        exit 1
    fi
    
    # Add Bitnami repository for dependencies
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    # Install/upgrade the application
    local values_file="helm/automata-app/values.yaml"
    if [[ "$ENVIRONMENT" != "production" ]]; then
        values_file="helm/automata-app/values-${ENVIRONMENT}.yaml"
    fi
    
    if [[ -f "$values_file" ]]; then
        helm upgrade --install automata-app ./helm/automata-app \
            --namespace ${NAMESPACE} \
            --create-namespace \
            --values "$values_file" \
            --set image.tag="${IMAGE_TAG}" \
            --wait \
            --timeout 10m
    else
        helm upgrade --install automata-app ./helm/automata-app \
            --namespace ${NAMESPACE} \
            --create-namespace \
            --set image.tag="${IMAGE_TAG}" \
            --wait \
            --timeout 10m
    fi
    
    log_success "Helm deployment completed"
}

# Scale deployment
scale_deployment() {
    local component=$1
    local replicas=$2
    
    log_info "Scaling ${component} to ${replicas} replicas..."
    
    if [[ -n $(helm list -n ${NAMESPACE} -q -f "^automata-app$") ]]; then
        # Helm deployment
        case $component in
            backend)
                helm upgrade automata-app ./helm/automata-app \
                    --namespace ${NAMESPACE} \
                    --reuse-values \
                    --set backend.replicaCount=${replicas}
                ;;
            frontend)
                helm upgrade automata-app ./helm/automata-app \
                    --namespace ${NAMESPACE} \
                    --reuse-values \
                    --set frontend.replicaCount=${replicas}
                ;;
            *)
                log_error "Unknown component: $component"
                exit 1
                ;;
        esac
    else
        # kubectl deployment
        kubectl scale deployment/${component} --replicas=${replicas} -n ${NAMESPACE}
    fi
    
    log_success "${component} scaled to ${replicas} replicas"
}

# Get deployment status
get_status() {
    log_info "Getting deployment status..."
    
    echo "Pods:"
    kubectl get pods -n ${NAMESPACE} -o wide
    
    echo -e "\nServices:"
    kubectl get services -n ${NAMESPACE}
    
    echo -e "\nIngress:"
    kubectl get ingress -n ${NAMESPACE}
    
    echo -e "\nHPA:"
    kubectl get hpa -n ${NAMESPACE} 2>/dev/null || echo "No HPA found"
    
    if command_exists helm && [[ -n $(helm list -n ${NAMESPACE} -q -f "^automata-app$") ]]; then
        echo -e "\nHelm Status:"
        helm status automata-app -n ${NAMESPACE}
    fi
}

# Clean up deployment
cleanup() {
    log_info "Cleaning up deployment..."
    
    if command_exists helm && [[ -n $(helm list -n ${NAMESPACE} -q -f "^automata-app$") ]]; then
        log_info "Uninstalling Helm release..."
        helm uninstall automata-app -n ${NAMESPACE}
    else
        log_info "Deleting kubectl resources..."
        kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
    fi
    
    log_success "Cleanup completed"
}

# Update secrets
update_secrets() {
    log_info "Updating secrets..."
    
    # Check if secrets file exists
    local secrets_file="secrets/${ENVIRONMENT}.env"
    if [[ ! -f "$secrets_file" ]]; then
        log_error "Secrets file not found: $secrets_file"
        log_info "Please create the secrets file with the required values."
        return 1
    fi
    
    # Create secret from file
    kubectl create secret generic automata-secrets \
        --from-env-file="$secrets_file" \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Restart deployments to pick up new secrets
    kubectl rollout restart deployment -n ${NAMESPACE}
    
    log_success "Secrets updated"
}

# Database migration
migrate_database() {
    log_info "Running database migration..."
    
    # Find backend pod
    local backend_pod=$(kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/component=backend -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -z "$backend_pod" ]]; then
        log_error "No backend pod found"
        exit 1
    fi
    
    # Run migration
    kubectl exec -n ${NAMESPACE} "$backend_pod" -- python -c "from app.database import init_db; init_db()"
    
    log_success "Database migration completed"
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build           Build and push Docker images
    deploy-kubectl  Deploy using kubectl manifests
    deploy-helm     Deploy using Helm chart
    scale          Scale deployment (requires component and replicas)
    status         Show deployment status
    cleanup        Clean up deployment
    update-secrets Update application secrets
    migrate        Run database migration

Options:
    -n, --namespace    Kubernetes namespace (default: automata-app)
    -e, --environment  Deployment environment (default: production)
    -r, --registry     Docker registry (default: docker.io)
    -t, --tag          Image tag (default: latest)
    -h, --help         Show this help message

Examples:
    $0 build
    $0 deploy-helm
    $0 scale backend 5
    $0 status
    $0 cleanup

Environment variables:
    NAMESPACE         Kubernetes namespace
    ENVIRONMENT       Deployment environment  
    DOCKER_REGISTRY   Docker registry URL
    IMAGE_TAG         Docker image tag
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            COMMAND="$1"
            shift
            break
            ;;
    esac
done

# Main execution
main() {
    case "${COMMAND:-}" in
        build)
            check_prerequisites
            build_images
            ;;
        deploy-kubectl)
            check_prerequisites
            deploy_kubectl
            ;;
        deploy-helm)
            check_prerequisites
            deploy_helm
            ;;
        scale)
            if [[ $# -lt 2 ]]; then
                log_error "Scale command requires component and replicas"
                log_error "Usage: $0 scale <component> <replicas>"
                exit 1
            fi
            check_prerequisites
            scale_deployment "$1" "$2"
            ;;
        status)
            get_status
            ;;
        cleanup)
            cleanup
            ;;
        update-secrets)
            update_secrets
            ;;
        migrate)
            migrate_database
            ;;
        *)
            log_error "Unknown command: ${COMMAND:-}"
            echo
            usage
            exit 1
            ;;
    esac
}

main "$@"