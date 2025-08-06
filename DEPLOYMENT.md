# Deployment Guide

This document provides comprehensive instructions for deploying the Automata Theory Learning Application with performance and scale optimizations.

## Architecture Overview

The application consists of:
- **Frontend**: React/Vite application with CDN optimization and code splitting
- **Backend**: FastAPI with advanced database optimization and connection pooling
- **Database**: PostgreSQL with performance indexes and query optimization
- **Cache**: Redis for session management and query caching
- **CDN**: CloudFlare/AWS CloudFront integration for static assets

## Performance Optimizations Implemented

### 1. CDN Integration (`/backend/app/cdn_config.py`)
- CloudFlare and AWS CloudFront support
- Static asset management with versioning
- Automatic cache purging
- Gzip and Brotli compression

### 2. Frontend Optimizations (`/frontend/vite.config.ts`)
- Advanced code splitting with dynamic imports
- Bundle analysis and visualization
- Route-based lazy loading
- Asset optimization by type (images, fonts, CSS)
- Manual chunk splitting for better caching

### 3. Code Splitting (`/frontend/src/utils/lazy-loader.ts`, `/frontend/src/router/lazy-routes.tsx`)
- Retry logic for failed imports
- Intersection Observer lazy loading
- Preloading strategies (critical, high, medium, low priority)
- Error boundaries for failed components

### 4. Database Optimization (`/backend/app/database_optimization.py`)
- Connection pooling (SQLAlchemy + AsyncPG)
- Redis query caching
- Performance indexes on all tables
- Query monitoring and slow query detection
- Bulk operations support

### 5. Kubernetes Deployment
- Horizontal Pod Autoscaler (HPA)
- Vertical Pod Autoscaler (VPA) recommendations
- Network policies for security
- Pod Disruption Budgets (PDB)
- Resource quotas and limits
- Monitoring and alerting

## Deployment Options

### Option 1: Helm Deployment (Recommended)

```bash
# Install with default values
helm install automata-app ./helm/automata-app \
  --namespace automata-app \
  --create-namespace

# Install with custom values
helm install automata-app ./helm/automata-app \
  --namespace automata-app \
  --create-namespace \
  --values ./helm/automata-app/values-production.yaml
```

### Option 2: kubectl Deployment

```bash
# Apply all manifests
kubectl apply -f k8s/

# Or use the deployment script
./scripts/deploy.sh deploy-kubectl
```

### Option 3: Deployment Script

The deployment script provides various commands:

```bash
# Build and push images
./scripts/deploy.sh build

# Deploy with Helm
./scripts/deploy.sh deploy-helm

# Scale components
./scripts/deploy.sh scale backend 10
./scripts/deploy.sh scale frontend 5

# Check status
./scripts/deploy.sh status

# Clean up
./scripts/deploy.sh cleanup
```

## Configuration

### Environment Variables

Create environment-specific configuration files:

```bash
# Production secrets
cat > secrets/production.env << EOF
POSTGRES_PASSWORD=secure-production-password
SECRET_KEY=super-secret-jwt-key-change-in-production
OPENAI_API_KEY=sk-...
CDN_API_TOKEN=cloudflare-or-aws-token
EOF
```

### Helm Values Customization

Create custom values files for different environments:

```yaml
# helm/automata-app/values-production.yaml
backend:
  replicaCount: 5
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "4Gi"
      cpu: "2000m"

postgresql:
  primary:
    resources:
      requests:
        memory: "1Gi"
        cpu: "500m"
      limits:
        memory: "4Gi"
        cpu: "2000m"

ingress:
  hosts:
    - host: automata.example.com
      paths:
        - path: /
          service:
            name: frontend
            port: 80
```

## Performance Tuning

### Database Optimization

The database optimization module provides:

1. **Connection Pooling**:
   - SQLAlchemy pool: 20 base connections, 30 overflow
   - AsyncPG pool: 10-100 connections
   - Connection recycling every hour

2. **Query Caching**:
   - Redis-based query result caching
   - Automatic cache invalidation
   - Configurable TTL per query type

3. **Performance Monitoring**:
   - Query execution time tracking
   - Slow query logging (>1s threshold)
   - Connection pool metrics

### Frontend Performance

1. **Code Splitting Strategy**:
   - Critical components: preloaded immediately
   - High priority: preloaded on idle
   - Medium priority: preloaded on user interaction
   - Low priority: lazy loaded when needed

2. **Asset Optimization**:
   - Image optimization and WebP conversion
   - Font preloading and subsetting
   - CSS/JS minification and compression

### CDN Configuration

Configure CDN settings in `cdn_config.py`:

```python
config = CDNConfig(
    provider=CDNProvider.CLOUDFLARE,
    base_url="https://cdn.automata.example.com",
    zone_id="your-cloudflare-zone-id",
    api_token="your-api-token",
    browser_cache_ttl=31536000,  # 1 year
    edge_cache_ttl=2592000,      # 30 days
)
```

## Monitoring and Alerting

### Prometheus Metrics

The application exposes metrics at `/metrics` endpoint:
- HTTP request duration and count
- Database connection pool status
- Query execution times
- Cache hit/miss ratios

### Alerts Configuration

Key alerts configured in `k8s/monitoring.yaml`:
- High error rate (>10% 5xx responses)
- High response time (95th percentile >2s)
- Database connection usage >80%
- Memory/CPU usage >90%
- Pod crash looping

### Health Checks

Health check endpoints:
- `/health` - Application health
- `/health/db` - Database connectivity
- `/health/redis` - Redis connectivity

## Scaling

### Horizontal Scaling

HPA is configured for:
- Backend: 3-20 replicas based on CPU (70%) and memory (80%)
- Frontend: 2-10 replicas based on CPU (80%) and memory (85%)

### Vertical Scaling

VPA provides recommendations for:
- Resource requests/limits optimization
- Right-sizing based on actual usage

### Auto-scaling Triggers

Scaling policies:
- Scale up: 50% increase or 2 pods per minute
- Scale down: 10% decrease per minute with 5-minute stabilization

## Security

### Network Policies

Network policies restrict:
- Ingress to ingress controller only
- Internal communication within namespace
- Egress to DNS and HTTPS only

### Pod Security

Security contexts enforce:
- Non-root user execution
- Read-only root filesystem
- Dropped capabilities
- No privilege escalation

### Secrets Management

Secrets are stored in Kubernetes secrets:
- Database passwords
- API keys
- JWT signing keys
- CDN tokens

## Troubleshooting

### Common Issues

1. **Database Connection Issues**:
   ```bash
   kubectl logs deployment/backend -n automata-app
   kubectl exec -it backend-pod -n automata-app -- python -c "from app.database_optimization import check_database_health; print(check_database_health())"
   ```

2. **Performance Issues**:
   ```bash
   kubectl top pods -n automata-app
   kubectl get hpa -n automata-app
   ```

3. **CDN Issues**:
   ```bash
   kubectl logs deployment/backend -n automata-app | grep CDN
   ```

### Debugging Commands

```bash
# Check pod status
kubectl get pods -n automata-app -o wide

# View logs
kubectl logs -f deployment/backend -n automata-app
kubectl logs -f deployment/frontend -n automata-app

# Port forward for local access
kubectl port-forward svc/backend-service 8000:8000 -n automata-app
kubectl port-forward svc/frontend-service 3000:80 -n automata-app

# Database access
kubectl port-forward svc/postgres-service 5432:5432 -n automata-app

# Redis access
kubectl port-forward svc/redis-service 6379:6379 -n automata-app
```

## Backup and Recovery

### Database Backups

Configure automated backups:
```bash
# Create backup job
kubectl create job --from=cronjob/postgres-backup postgres-backup-manual -n automata-app
```

### Disaster Recovery

1. Database: Point-in-time recovery with WAL archiving
2. Application: Multi-region deployment with traffic routing
3. CDN: Origin failover to backup servers

## Performance Benchmarks

Expected performance with optimizations:
- Response time: <200ms (95th percentile)
- Throughput: >1000 RPS per backend pod
- Database: <50ms average query time
- CDN cache hit ratio: >95%
- Frontend bundle size: <500KB (main chunk)

## Cost Optimization

Optimization strategies:
- Right-sized resource requests/limits
- Efficient connection pooling
- CDN reduces origin server load
- Auto-scaling prevents over-provisioning
- Spot instances for non-critical workloads