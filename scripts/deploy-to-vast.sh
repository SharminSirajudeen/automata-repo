#!/bin/bash

# Deploy to vast.ai - Complete Production Deployment Script
# This script handles the entire deployment process to vast.ai

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}     Automata Learning Platform Deployment      ${NC}"
echo -e "${GREEN}             Deploying to vast.ai               ${NC}"
echo -e "${GREEN}================================================${NC}"

# Function to check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}kubectl is not installed. Please install kubectl first.${NC}"
        exit 1
    fi
    
    # Check environment files
    if [ ! -f ".env.production" ]; then
        echo -e "${RED}.env.production file not found. Creating from template...${NC}"
        cp frontend/.env.production .env.production
        echo -e "${YELLOW}Please edit .env.production with your actual values${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}All prerequisites satisfied!${NC}"
}

# Function to run security checks
run_security_checks() {
    echo -e "\n${YELLOW}Running security checks...${NC}"
    
    # Check for exposed secrets
    if grep -r "password\|secret\|key\|token" docker-compose.yml docker-compose.prod.yml 2>/dev/null | grep -v "changeme\|your-\|example"; then
        echo -e "${RED}WARNING: Potential exposed secrets found in Docker Compose files!${NC}"
        echo -e "${YELLOW}Please review and fix before continuing.${NC}"
        read -p "Continue anyway? (y/N): " confirm
        if [ "$confirm" != "y" ]; then
            exit 1
        fi
    fi
    
    # Generate secure passwords if needed
    if [ ! -f ".env.production.secrets" ]; then
        echo -e "${YELLOW}Generating secure secrets...${NC}"
        cat > .env.production.secrets <<EOF
# Auto-generated secure secrets - DO NOT COMMIT TO GIT
POSTGRES_PASSWORD=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 64)
REDIS_PASSWORD=$(openssl rand -base64 32)
SENTRY_DSN=your-sentry-dsn-here
EOF
        echo -e "${GREEN}Secure secrets generated in .env.production.secrets${NC}"
    fi
}

# Function to build Docker images
build_images() {
    echo -e "\n${YELLOW}Building Docker images...${NC}"
    
    # Build frontend
    echo -e "${YELLOW}Building frontend image...${NC}"
    docker build -f frontend/Dockerfile -t automata-frontend:prod ./frontend
    
    # Build backend
    echo -e "${YELLOW}Building backend image...${NC}"
    docker build -f backend/Dockerfile -t automata-backend:prod ./backend
    
    echo -e "${GREEN}Docker images built successfully!${NC}"
}

# Function to push images to registry
push_images() {
    echo -e "\n${YELLOW}Pushing images to registry...${NC}"
    
    read -p "Enter your Docker registry (e.g., docker.io/username): " REGISTRY
    
    # Tag images
    docker tag automata-frontend:prod $REGISTRY/automata-frontend:prod
    docker tag automata-backend:prod $REGISTRY/automata-backend:prod
    
    # Push images
    docker push $REGISTRY/automata-frontend:prod
    docker push $REGISTRY/automata-backend:prod
    
    echo -e "${GREEN}Images pushed to registry!${NC}"
}

# Function to setup vast.ai instance
setup_vast_instance() {
    echo -e "\n${YELLOW}Setting up vast.ai instance...${NC}"
    
    read -p "Enter your vast.ai instance IP: " VAST_IP
    read -p "Enter your SSH key path: " SSH_KEY
    
    # Copy deployment files
    echo -e "${YELLOW}Copying deployment files to vast.ai instance...${NC}"
    scp -i $SSH_KEY -r k8s/ root@$VAST_IP:/root/
    scp -i $SSH_KEY .env.production.secrets root@$VAST_IP:/root/
    
    # Setup Kubernetes on vast.ai
    echo -e "${YELLOW}Setting up Kubernetes on vast.ai...${NC}"
    ssh -i $SSH_KEY root@$VAST_IP 'bash -s' < scripts/k8s-setup.sh
    
    echo -e "${GREEN}vast.ai instance setup complete!${NC}"
}

# Function to deploy application
deploy_application() {
    echo -e "\n${YELLOW}Deploying application to Kubernetes...${NC}"
    
    # Create namespace
    kubectl create namespace automata-app --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    kubectl create secret generic app-secrets \
        --from-env-file=.env.production.secrets \
        --namespace=automata-app \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/ --namespace=automata-app
    
    # Wait for deployments
    echo -e "${YELLOW}Waiting for deployments to be ready...${NC}"
    kubectl wait --for=condition=available --timeout=300s \
        deployment/frontend-deployment deployment/backend-deployment \
        --namespace=automata-app
    
    echo -e "${GREEN}Application deployed successfully!${NC}"
}

# Function to setup monitoring
setup_monitoring() {
    echo -e "\n${YELLOW}Setting up monitoring...${NC}"
    
    # Deploy Prometheus and Grafana
    kubectl apply -f k8s/monitoring/ --namespace=automata-app
    
    # Get monitoring URLs
    FRONTEND_URL=$(kubectl get ingress -n automata-app -o jsonpath='{.items[0].spec.rules[0].host}')
    
    echo -e "${GREEN}Monitoring setup complete!${NC}"
    echo -e "${YELLOW}Access your application at: https://$FRONTEND_URL${NC}"
}

# Function to verify deployment
verify_deployment() {
    echo -e "\n${YELLOW}Verifying deployment...${NC}"
    
    # Check pod status
    kubectl get pods -n automata-app
    
    # Check services
    kubectl get svc -n automata-app
    
    # Run health checks
    BACKEND_POD=$(kubectl get pod -n automata-app -l app=backend -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n automata-app $BACKEND_POD -- curl -f http://localhost:8000/health
    
    echo -e "${GREEN}Deployment verification complete!${NC}"
}

# Main deployment flow
main() {
    echo -e "${YELLOW}Starting deployment process...${NC}"
    
    check_prerequisites
    run_security_checks
    build_images
    
    read -p "Push images to registry? (y/N): " push_confirm
    if [ "$push_confirm" = "y" ]; then
        push_images
    fi
    
    read -p "Setup vast.ai instance? (y/N): " vast_confirm
    if [ "$vast_confirm" = "y" ]; then
        setup_vast_instance
    fi
    
    deploy_application
    setup_monitoring
    verify_deployment
    
    echo -e "\n${GREEN}================================================${NC}"
    echo -e "${GREEN}       Deployment Complete Successfully!        ${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo -e "1. Configure DNS to point to your vast.ai instance"
    echo -e "2. Set up SSL certificates with Let's Encrypt"
    echo -e "3. Configure Sentry for error monitoring"
    echo -e "4. Set up backup strategies"
    echo -e "5. Monitor application performance"
}

# Run main function
main