# 🚀 Automata Platform Deployment Guide

## Overview

This document provides comprehensive instructions for deploying the Automata Theory Learning Platform on vast.ai GPU instances using Docker and Kubernetes orchestration.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Local Development](#local-development)
- [Production Deployment](#production-deployment)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)

## Prerequisites

### System Requirements

- **vast.ai GPU instance** with at least:
  - 8 vCPUs
  - 16 GB RAM
  - 100 GB SSD storage
  - 1x NVIDIA GPU (recommended: RTX 3090 or better)
- **Operating System**: Ubuntu 20.04+ LTS
- **Network**: Open ports 80, 443, 6443

### Required Tools

- Docker 24.0+
- Kubernetes 1.28+
- Helm 3.13+
- kubectl configured
- NVIDIA Docker runtime (for GPU support)

### Environment Variables

Create a `.env` file with the following secrets:

```bash
# Database
POSTGRES_PASSWORD="secure-postgres-password-256-chars"
REDIS_PASSWORD="secure-redis-password"

# Application Security
SECRET_KEY="super-secure-application-key-must-be-long-and-random"
JWT_SECRET="jwt-secret-key-for-token-signing-512-bits-recommended"

# AI Services
OPENAI_API_KEY="sk-proj-your-openai-api-key"
ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key"

# Domain Configuration
DOMAIN="automata.vast.ai"
API_URL="https://api.automata.vast.ai"
WS_URL="wss://api.automata.vast.ai/ws"

# Docker Registry (if using private registry)
DOCKER_USERNAME="your-registry-username"
DOCKER_PASSWORD="your-registry-password"

# vast.ai Configuration
VAST_AI_INSTANCE_ID="your-instance-id"
```

## Architecture Overview

### Container Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Database      │
│   (Nginx +      │────│   (FastAPI +    │────│  (PostgreSQL +  │
│    React)       │    │    Python)      │    │     Redis)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Kubernetes    │
                    │   Orchestration │
                    └─────────────────┘
```

### Kubernetes Components

- **Namespace**: `automata-app`
- **Deployments**: Frontend (Nginx), Backend (FastAPI), Database (PostgreSQL), Cache (Redis)
- **Services**: Load balancing and service discovery
- **Ingress**: HTTPS termination with Let's Encrypt
- **Storage**: Persistent volumes for data persistence
- **Monitoring**: Prometheus + Grafana stack

## Local Development

### Quick Start with Docker Compose

```bash
# Clone repository
git clone <repository-url>
cd automata-repo

# Set environment variables
cp .env.example .env
# Edit .env with your values

# Start local development environment
docker-compose up -d

# Check services
docker-compose ps
```

### Accessing Services

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Database**: localhost:5432
- **Redis**: localhost:6379

### Development Commands

```bash
# View logs
docker-compose logs -f

# Rebuild services
docker-compose up -d --build

# Run database migrations
docker-compose exec backend alembic upgrade head

# Access database
docker-compose exec postgres psql -U automata -d automata_db

# Redis CLI
docker-compose exec redis redis-cli
```

## Production Deployment

### Step 1: Prepare vast.ai Instance

```bash
# SSH into your vast.ai instance
ssh -p <port> root@<instance-ip>

# Download deployment scripts
wget https://raw.githubusercontent.com/<your-repo>/main/scripts/k8s-setup.sh
chmod +x k8s-setup.sh

# Setup Kubernetes cluster
./k8s-setup.sh
```

### Step 2: Configure Secrets

```bash
# Set required environment variables
export POSTGRES_PASSWORD="your-secure-password"
export SECRET_KEY="your-secret-key"
export JWT_SECRET="your-jwt-secret"
export OPENAI_API_KEY="sk-proj-your-key"
export DOMAIN="your-domain.vast.ai"

# Verify all required vars are set
echo "Required vars:"
echo "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:0:10}..."
echo "SECRET_KEY: ${SECRET_KEY:0:10}..."
echo "JWT_SECRET: ${JWT_SECRET:0:10}..."
echo "OPENAI_API_KEY: ${OPENAI_API_KEY:0:20}..."
```

### Step 3: Deploy Application

```bash
# Clone repository
git clone <your-repository-url>
cd automata-repo

# Make deployment script executable
chmod +x scripts/deploy-vast.sh

# Deploy to production
./scripts/deploy-vast.sh latest production
```

### Step 4: Configure DNS

Point your domain to the vast.ai instance IP:

```bash
# Get external IP
kubectl get svc -n ingress-nginx

# Update DNS A record
# your-domain.vast.ai -> <EXTERNAL-IP>
```

### Step 5: Verify Deployment

```bash
# Check pod status
kubectl get pods -n automata-app

# Check services
kubectl get svc -n automata-app

# Check ingress
kubectl get ingress -n automata-app

# Test application
curl -f https://your-domain.vast.ai/health
curl -f https://api.your-domain.vast.ai/health
```

## Monitoring & Maintenance

### Monitoring Dashboard

Access monitoring at: `https://your-domain.vast.ai:3001`

Default credentials:
- Username: admin
- Password: [check secrets]

### Key Metrics to Monitor

1. **Application Health**
   - HTTP response times
   - Error rates
   - Request volume

2. **Infrastructure Health**
   - CPU and memory usage
   - Disk space
   - Network I/O

3. **Database Performance**
   - Connection pool status
   - Query performance
   - Storage usage

4. **AI Processing**
   - GPU utilization
   - AI request queue depth
   - Model inference times

### Maintenance Commands

```bash
# Scale application
kubectl scale deployment automata-backend --replicas=4 -n automata-app

# Update application
kubectl set image deployment/automata-backend backend=registry.vast.ai/automata-backend:new-version -n automata-app

# Check resource usage
kubectl top nodes
kubectl top pods -n automata-app

# View logs
kubectl logs -f deployment/automata-backend -n automata-app

# Database backup
kubectl exec -n automata-app deployment/automata-postgres -- pg_dump -U automata automata_db > backup.sql

# Restart services
kubectl rollout restart deployment/automata-backend -n automata-app
```

### Auto-scaling Configuration

The platform includes Horizontal Pod Autoscaler (HPA):

- **Frontend**: Scales 2-10 replicas based on CPU (70%) and memory (80%)
- **Backend**: Scales 2-8 replicas based on CPU (60%), memory (75%), and request rate

Monitor scaling:
```bash
kubectl get hpa -n automata-app -w
```

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n automata-app

# Check logs
kubectl logs <pod-name> -n automata-app

# Common causes:
# - Resource limits too low
# - Missing secrets
# - Image pull failures
```

#### 2. Database Connection Issues

```bash
# Check database pod
kubectl get pods -n automata-app -l app=automata-postgres

# Test database connectivity
kubectl exec -n automata-app deployment/automata-postgres -- pg_isready -U automata

# Check database logs
kubectl logs -n automata-app deployment/automata-postgres
```

#### 3. SSL Certificate Issues

```bash
# Check cert-manager
kubectl get certificates -n automata-app
kubectl describe certificate automata-tls -n automata-app

# Check certificate issuer
kubectl get clusterissuer letsencrypt-prod

# Manual certificate renewal
kubectl delete certificate automata-tls -n automata-app
kubectl apply -f k8s/ingress.yaml
```

#### 4. High Memory Usage

```bash
# Check memory usage
kubectl top pods -n automata-app

# Adjust resource limits
kubectl patch deployment automata-backend -n automata-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"backend","resources":{"limits":{"memory":"4Gi"}}}]}}}}'

# Clear Redis cache if needed
kubectl exec -n automata-app deployment/automata-redis -- redis-cli FLUSHALL
```

### Debug Commands

```bash
# Get all resources
kubectl get all -n automata-app

# Describe problematic resource
kubectl describe deployment automata-backend -n automata-app

# Check events
kubectl get events -n automata-app --sort-by='.lastTimestamp'

# Port forward for local debugging
kubectl port-forward svc/automata-backend 8000:8000 -n automata-app

# Execute commands in pods
kubectl exec -it deployment/automata-backend -n automata-app -- /bin/bash
```

## Security Considerations

### 1. Secret Management

- Use external secret management (HashiCorp Vault, AWS Secrets Manager)
- Rotate secrets regularly
- Never commit secrets to version control

### 2. Network Security

- Network policies are configured to restrict pod-to-pod communication
- Ingress controller with rate limiting
- TLS termination with strong ciphers

### 3. Container Security

- Images run as non-root users
- Security contexts configured
- Regular vulnerability scanning with Trivy

### 4. Access Control

- RBAC configured for service accounts
- Least privilege principle
- Regular access review

### 5. Data Protection

- Database encryption at rest
- TLS for all communications
- Regular backups with encryption

## Backup & Recovery

### Automated Backups

Database backups run daily:

```bash
# Check backup status
kubectl get cronjobs -n automata-app

# Manual backup
kubectl create job --from=cronjob/postgres-backup manual-backup -n automata-app
```

### Recovery Procedures

```bash
# Restore from backup
kubectl exec -n automata-app deployment/automata-postgres -- psql -U automata -d automata_db < backup.sql

# Rollback deployment
kubectl rollout undo deployment/automata-backend -n automata-app

# Scale down for maintenance
kubectl scale deployment --all --replicas=0 -n automata-app
```

## Performance Tuning

### Database Optimization

- Connection pooling configured
- Query optimization with indexes
- Regular VACUUM and ANALYZE

### Application Optimization

- Redis caching for frequently accessed data
- Async processing for heavy operations
- CDN for static assets

### Infrastructure Optimization

- Resource limits tuned for workload
- Node affinity for optimal placement
- Persistent volumes with high IOPS

## CI/CD Pipeline

### GitHub Actions Workflows

1. **Test Pipeline** (`.github/workflows/test.yml`)
   - Frontend and backend tests
   - Security scanning
   - Performance testing

2. **Deployment Pipeline** (`.github/workflows/deploy.yml`)
   - Automated deployments to staging/production
   - Health checks and rollback capabilities

3. **Monitoring Pipeline** (`.github/workflows/monitoring.yml`)
   - Continuous health monitoring
   - Performance tracking
   - Alert management

### Deployment Environments

- **Development**: Local Docker Compose
- **Staging**: Kubernetes on vast.ai (staging.automata.vast.ai)
- **Production**: Kubernetes on vast.ai (automata.vast.ai)

## Support & Contributing

### Getting Help

1. Check the [troubleshooting section](#troubleshooting)
2. Review application logs
3. Search existing issues
4. Create a detailed issue report

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Monitoring Integration

The platform integrates with:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and alerting  
- **Jaeger**: Distributed tracing (optional)
- **ELK Stack**: Log aggregation (optional)

---

For additional support, please refer to the project documentation or create an issue in the GitHub repository.