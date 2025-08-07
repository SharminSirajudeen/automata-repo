#!/bin/bash

# Backend Scaling Script for Automata Learning Platform
# Supports automatic scaling, rolling updates, and zero-downtime deployments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"
K8S_DIR="$PROJECT_ROOT/k8s"

# Default configuration
DEFAULT_MIN_REPLICAS=2
DEFAULT_MAX_REPLICAS=10
DEFAULT_TARGET_CPU=70
DEFAULT_TARGET_MEMORY=80
DEFAULT_SCALE_UP_THRESHOLD=80
DEFAULT_SCALE_DOWN_THRESHOLD=30
DEFAULT_COOLDOWN_PERIOD=300

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $*${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $*${NC}" >&2
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*${NC}" >&2
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*${NC}" >&2
}

# Help function
show_help() {
    cat << EOF
Backend Scaling Script for Automata Learning Platform

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    scale-up [REPLICAS]         Scale up backend instances
    scale-down [REPLICAS]       Scale down backend instances
    auto-scale                  Enable auto-scaling with HPA
    disable-auto-scale          Disable auto-scaling
    rolling-update [IMAGE]      Perform rolling update
    canary-deploy [IMAGE]       Deploy canary version
    rollback                    Rollback to previous version
    status                      Show current scaling status
    metrics                     Show scaling metrics
    health-check               Check health of all instances
    load-test                  Run load test to verify scaling
    cleanup                    Clean up old deployments

Options:
    --namespace NAMESPACE       Kubernetes namespace (default: default)
    --min-replicas NUM          Minimum replicas for auto-scaling (default: $DEFAULT_MIN_REPLICAS)
    --max-replicas NUM          Maximum replicas for auto-scaling (default: $DEFAULT_MAX_REPLICAS)
    --cpu-threshold NUM         CPU threshold for scaling (default: $DEFAULT_TARGET_CPU%)
    --memory-threshold NUM      Memory threshold for scaling (default: $DEFAULT_TARGET_MEMORY%)
    --image IMAGE               Docker image to deploy
    --timeout SECONDS           Operation timeout (default: 600)
    --dry-run                   Show what would be done without executing
    --force                     Force operation without confirmation
    --verbose                   Enable verbose output

Examples:
    $0 scale-up 5                                    # Scale to 5 replicas
    $0 auto-scale --min-replicas 3 --max-replicas 15 # Enable auto-scaling
    $0 rolling-update myregistry/automata-backend:v2.0.0
    $0 canary-deploy myregistry/automata-backend:v2.1.0
    $0 rollback                                      # Rollback deployment
    $0 load-test                                    # Run load test

EOF
}

# Parse command line arguments
parse_args() {
    COMMAND=""
    NAMESPACE="default"
    MIN_REPLICAS=$DEFAULT_MIN_REPLICAS
    MAX_REPLICAS=$DEFAULT_MAX_REPLICAS
    CPU_THRESHOLD=$DEFAULT_TARGET_CPU
    MEMORY_THRESHOLD=$DEFAULT_TARGET_MEMORY
    IMAGE=""
    TIMEOUT=600
    DRY_RUN=false
    FORCE=false
    VERBOSE=false
    REPLICAS=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            scale-up|scale-down|auto-scale|disable-auto-scale|rolling-update|canary-deploy|rollback|status|metrics|health-check|load-test|cleanup)
                COMMAND="$1"
                if [[ $# -gt 1 && ! $2 =~ ^-- ]]; then
                    if [[ $1 == "scale-up" || $1 == "scale-down" ]]; then
                        REPLICAS="$2"
                        shift
                    elif [[ $1 == "rolling-update" || $1 == "canary-deploy" ]]; then
                        IMAGE="$2"
                        shift
                    fi
                fi
                ;;
            --namespace)
                NAMESPACE="$2"
                shift
                ;;
            --min-replicas)
                MIN_REPLICAS="$2"
                shift
                ;;
            --max-replicas)
                MAX_REPLICAS="$2"
                shift
                ;;
            --cpu-threshold)
                CPU_THRESHOLD="$2"
                shift
                ;;
            --memory-threshold)
                MEMORY_THRESHOLD="$2"
                shift
                ;;
            --image)
                IMAGE="$2"
                shift
                ;;
            --timeout)
                TIMEOUT="$2"
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                ;;
            --force)
                FORCE=true
                ;;
            --verbose)
                VERBOSE=true
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
        shift
    done

    if [[ -z "$COMMAND" ]]; then
        error "No command specified"
        show_help
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    local missing_tools=()

    # Check for required tools
    command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    # Check kubectl connection
    if ! kubectl cluster-info >/dev/null 2>&1; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        warn "Namespace '$NAMESPACE' does not exist, creating it..."
        if [[ $DRY_RUN == false ]]; then
            kubectl create namespace "$NAMESPACE"
        fi
    fi
}

# Execute command with dry-run support
execute() {
    if [[ $VERBOSE == true ]]; then
        info "Executing: $*"
    fi

    if [[ $DRY_RUN == true ]]; then
        info "[DRY-RUN] Would execute: $*"
    else
        "$@"
    fi
}

# Get current deployment info
get_deployment_info() {
    local deployment_name="automata-backend"
    
    if kubectl get deployment "$deployment_name" -n "$NAMESPACE" >/dev/null 2>&1; then
        kubectl get deployment "$deployment_name" -n "$NAMESPACE" -o json
    else
        echo "{}"
    fi
}

# Get current replica count
get_current_replicas() {
    local deployment_info
    deployment_info=$(get_deployment_info)
    echo "$deployment_info" | jq -r '.spec.replicas // 0'
}

# Scale deployment
scale_deployment() {
    local target_replicas=$1
    local deployment_name="automata-backend"

    log "Scaling deployment '$deployment_name' to $target_replicas replicas"

    execute kubectl scale deployment "$deployment_name" \
        --replicas="$target_replicas" \
        --namespace="$NAMESPACE"

    if [[ $DRY_RUN == false ]]; then
        # Wait for rollout to complete
        log "Waiting for deployment to scale..."
        kubectl rollout status deployment "$deployment_name" \
            --namespace="$NAMESPACE" \
            --timeout="${TIMEOUT}s"
    fi
}

# Setup Horizontal Pod Autoscaler
setup_hpa() {
    local deployment_name="automata-backend"
    local hpa_name="automata-backend-hpa"

    log "Setting up HPA with min=$MIN_REPLICAS, max=$MAX_REPLICAS, CPU=$CPU_THRESHOLD%"

    # Create HPA configuration
    cat > /tmp/hpa.yaml << EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: $hpa_name
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: $deployment_name
  minReplicas: $MIN_REPLICAS
  maxReplicas: $MAX_REPLICAS
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: $CPU_THRESHOLD
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: $MEMORY_THRESHOLD
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      selectPolicy: Min
EOF

    execute kubectl apply -f /tmp/hpa.yaml
    rm -f /tmp/hpa.yaml

    if [[ $DRY_RUN == false ]]; then
        log "HPA created successfully"
        kubectl get hpa "$hpa_name" -n "$NAMESPACE"
    fi
}

# Disable auto-scaling
disable_hpa() {
    local hpa_name="automata-backend-hpa"

    if kubectl get hpa "$hpa_name" -n "$NAMESPACE" >/dev/null 2>&1; then
        log "Disabling auto-scaling by deleting HPA"
        execute kubectl delete hpa "$hpa_name" -n "$NAMESPACE"
    else
        warn "HPA '$hpa_name' not found"
    fi
}

# Perform rolling update
rolling_update() {
    local new_image=$1
    local deployment_name="automata-backend"

    if [[ -z "$new_image" ]]; then
        error "Image not specified for rolling update"
        exit 1
    fi

    log "Performing rolling update to image: $new_image"

    # Update deployment image
    execute kubectl set image deployment "$deployment_name" \
        automata-backend="$new_image" \
        --namespace="$NAMESPACE"

    if [[ $DRY_RUN == false ]]; then
        # Wait for rollout to complete
        log "Waiting for rolling update to complete..."
        kubectl rollout status deployment "$deployment_name" \
            --namespace="$NAMESPACE" \
            --timeout="${TIMEOUT}s"

        # Verify deployment
        local ready_replicas
        ready_replicas=$(kubectl get deployment "$deployment_name" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        local total_replicas
        total_replicas=$(kubectl get deployment "$deployment_name" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')

        if [[ "$ready_replicas" == "$total_replicas" ]]; then
            log "Rolling update completed successfully"
        else
            error "Rolling update failed - only $ready_replicas/$total_replicas replicas ready"
            exit 1
        fi
    fi
}

# Perform canary deployment
canary_deploy() {
    local new_image=$1
    local canary_name="automata-backend-canary"

    if [[ -z "$new_image" ]]; then
        error "Image not specified for canary deployment"
        exit 1
    fi

    log "Performing canary deployment with image: $new_image"

    # Create canary deployment (10% traffic)
    local canary_replicas=1
    local main_deployment_info
    main_deployment_info=$(get_deployment_info)
    
    # Get main deployment spec as template
    local main_spec
    main_spec=$(echo "$main_deployment_info" | jq '.spec')

    # Create canary deployment manifest
    cat > /tmp/canary.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $canary_name
  namespace: $NAMESPACE
  labels:
    app: automata-backend
    version: canary
spec:
  replicas: $canary_replicas
  selector:
    matchLabels:
      app: automata-backend
      version: canary
  template:
    metadata:
      labels:
        app: automata-backend
        version: canary
    spec:
$(echo "$main_spec" | jq -r '.template.spec' | sed 's/^/      /')
---
apiVersion: v1
kind: Service
metadata:
  name: automata-backend-canary
  namespace: $NAMESPACE
spec:
  selector:
    app: automata-backend
    version: canary
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
EOF

    # Update image in canary deployment
    sed -i.bak "s|image: .*automata-backend.*|image: $new_image|g" /tmp/canary.yaml
    rm -f /tmp/canary.yaml.bak

    execute kubectl apply -f /tmp/canary.yaml
    rm -f /tmp/canary.yaml

    if [[ $DRY_RUN == false ]]; then
        log "Waiting for canary deployment to be ready..."
        kubectl rollout status deployment "$canary_name" \
            --namespace="$NAMESPACE" \
            --timeout="${TIMEOUT}s"

        info "Canary deployment created successfully"
        info "To promote canary to production, run: $0 rolling-update $new_image"
        info "To rollback canary, run: kubectl delete deployment $canary_name -n $NAMESPACE"
    fi
}

# Rollback deployment
rollback_deployment() {
    local deployment_name="automata-backend"

    log "Rolling back deployment '$deployment_name'"

    # Get rollout history
    local previous_revision
    previous_revision=$(kubectl rollout history deployment "$deployment_name" -n "$NAMESPACE" --output=json | \
        jq -r '.items[-2].metadata.annotations."deployment.kubernetes.io/revision"' 2>/dev/null || echo "")

    if [[ -n "$previous_revision" ]]; then
        log "Rolling back to revision $previous_revision"
        execute kubectl rollout undo deployment "$deployment_name" \
            --to-revision="$previous_revision" \
            --namespace="$NAMESPACE"
    else
        log "Rolling back to previous revision"
        execute kubectl rollout undo deployment "$deployment_name" \
            --namespace="$NAMESPACE"
    fi

    if [[ $DRY_RUN == false ]]; then
        log "Waiting for rollback to complete..."
        kubectl rollout status deployment "$deployment_name" \
            --namespace="$NAMESPACE" \
            --timeout="${TIMEOUT}s"

        log "Rollback completed successfully"
    fi
}

# Show current status
show_status() {
    local deployment_name="automata-backend"
    local hpa_name="automata-backend-hpa"

    log "Current scaling status:"
    echo

    # Deployment status
    if kubectl get deployment "$deployment_name" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo "Deployment Status:"
        kubectl get deployment "$deployment_name" -n "$NAMESPACE"
        echo

        # Pod status
        echo "Pod Status:"
        kubectl get pods -l app=automata-backend -n "$NAMESPACE"
        echo
    else
        warn "Deployment '$deployment_name' not found"
    fi

    # HPA status
    if kubectl get hpa "$hpa_name" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo "Auto-scaling Status:"
        kubectl get hpa "$hpa_name" -n "$NAMESPACE"
        echo

        echo "HPA Details:"
        kubectl describe hpa "$hpa_name" -n "$NAMESPACE"
    else
        info "Auto-scaling is disabled"
    fi

    # Service status
    echo "Service Status:"
    kubectl get services -l app=automata-backend -n "$NAMESPACE"
}

# Show metrics
show_metrics() {
    local deployment_name="automata-backend"

    log "Current metrics:"
    echo

    # CPU and Memory usage
    if command -v kubectl-top >/dev/null 2>&1 || kubectl top pods >/dev/null 2>&1; then
        echo "Resource Usage:"
        kubectl top pods -l app=automata-backend -n "$NAMESPACE" 2>/dev/null || \
            info "Metrics server not available"
        echo
    fi

    # Request rate and response time (if monitoring is available)
    if kubectl get service prometheus -n monitoring >/dev/null 2>&1; then
        info "Prometheus metrics available at: http://prometheus.monitoring.svc.cluster.local:9090"
    fi

    # Load balancer metrics (if available)
    local pods
    pods=$(kubectl get pods -l app=automata-backend -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')
    
    for pod in $pods; do
        echo "Metrics for $pod:"
        kubectl exec "$pod" -n "$NAMESPACE" -- curl -s localhost:8000/metrics/performance 2>/dev/null | \
            jq . 2>/dev/null || info "Metrics not available for $pod"
        echo
    done
}

# Health check all instances
health_check() {
    local deployment_name="automata-backend"
    local pods
    local healthy=0
    local total=0

    log "Checking health of all backend instances"

    pods=$(kubectl get pods -l app=automata-backend -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")

    if [[ -z "$pods" ]]; then
        error "No backend pods found"
        exit 1
    fi

    for pod in $pods; do
        total=$((total + 1))
        echo -n "Checking $pod... "
        
        if kubectl exec "$pod" -n "$NAMESPACE" -- curl -sf localhost:8000/healthz >/dev/null 2>&1; then
            echo -e "${GREEN}HEALTHY${NC}"
            healthy=$((healthy + 1))
        else
            echo -e "${RED}UNHEALTHY${NC}"
        fi
    done

    echo
    if [[ $healthy -eq $total ]]; then
        log "All $total instances are healthy"
    else
        warn "$healthy/$total instances are healthy"
        if [[ $healthy -eq 0 ]]; then
            error "No healthy instances found!"
            exit 1
        fi
    fi
}

# Run load test
load_test() {
    log "Running load test to verify scaling behavior"

    # Check if locust is available
    if ! command -v locust >/dev/null 2>&1; then
        error "Locust not found. Please install it: pip install locust"
        exit 1
    fi

    local load_test_dir="$PROJECT_ROOT/load_tests"
    if [[ ! -f "$load_test_dir/locustfile.py" ]]; then
        error "Load test files not found in $load_test_dir"
        exit 1
    fi

    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get service automata-backend -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || \
                kubectl get service automata-backend -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')

    if [[ -z "$service_ip" ]]; then
        error "Cannot determine service endpoint"
        exit 1
    fi

    local service_port=8000
    local target_url="http://$service_ip:$service_port"

    info "Running load test against $target_url"
    info "This will test auto-scaling behavior under load"

    # Run load test
    cd "$load_test_dir"
    locust --headless \
        --users 50 \
        --spawn-rate 5 \
        --run-time 5m \
        --host "$target_url" \
        --html report.html

    log "Load test completed. Report saved to $load_test_dir/report.html"
    
    # Show scaling results
    info "Checking if auto-scaling occurred..."
    kubectl get hpa automata-backend-hpa -n "$NAMESPACE" 2>/dev/null || \
        info "Auto-scaling not enabled"
}

# Cleanup old deployments
cleanup_deployments() {
    local deployment_name="automata-backend"

    log "Cleaning up old deployments and resources"

    # Clean up old replica sets
    local old_rs
    old_rs=$(kubectl get rs -l app=automata-backend -n "$NAMESPACE" -o jsonpath='{.items[?(@.spec.replicas==0)].metadata.name}')

    for rs in $old_rs; do
        if [[ -n "$rs" ]]; then
            info "Deleting old replica set: $rs"
            execute kubectl delete rs "$rs" -n "$NAMESPACE"
        fi
    done

    # Clean up canary deployments
    if kubectl get deployment automata-backend-canary -n "$NAMESPACE" >/dev/null 2>&1; then
        info "Found canary deployment"
        if [[ $FORCE == true ]] || confirm "Delete canary deployment?"; then
            execute kubectl delete deployment automata-backend-canary -n "$NAMESPACE"
            execute kubectl delete service automata-backend-canary -n "$NAMESPACE" 2>/dev/null || true
        fi
    fi

    # Clean up completed pods
    local completed_pods
    completed_pods=$(kubectl get pods -l app=automata-backend -n "$NAMESPACE" --field-selector=status.phase=Succeeded -o jsonpath='{.items[*].metadata.name}')

    for pod in $completed_pods; do
        if [[ -n "$pod" ]]; then
            info "Deleting completed pod: $pod"
            execute kubectl delete pod "$pod" -n "$NAMESPACE"
        fi
    done

    log "Cleanup completed"
}

# Confirmation prompt
confirm() {
    local message=$1
    if [[ $FORCE == true ]]; then
        return 0
    fi

    echo -n "$message [y/N] "
    read -r response
    [[ "$response" =~ ^[Yy]$ ]]
}

# Main function
main() {
    parse_args "$@"
    check_prerequisites

    case "$COMMAND" in
        scale-up)
            if [[ -z "$REPLICAS" ]]; then
                error "Number of replicas not specified"
                exit 1
            fi
            scale_deployment "$REPLICAS"
            ;;
        scale-down)
            if [[ -z "$REPLICAS" ]]; then
                error "Number of replicas not specified"
                exit 1
            fi
            scale_deployment "$REPLICAS"
            ;;
        auto-scale)
            setup_hpa
            ;;
        disable-auto-scale)
            disable_hpa
            ;;
        rolling-update)
            rolling_update "$IMAGE"
            ;;
        canary-deploy)
            canary_deploy "$IMAGE"
            ;;
        rollback)
            if confirm "Are you sure you want to rollback the deployment?"; then
                rollback_deployment
            fi
            ;;
        status)
            show_status
            ;;
        metrics)
            show_metrics
            ;;
        health-check)
            health_check
            ;;
        load-test)
            load_test
            ;;
        cleanup)
            cleanup_deployments
            ;;
        *)
            error "Unknown command: $COMMAND"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"