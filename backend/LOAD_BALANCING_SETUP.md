# Load Balancing Setup for Automata Learning Platform

This document provides comprehensive configuration and setup instructions for load balancing the Automata Learning Platform backend services.

## Overview

The load balancing solution supports:
- **High Availability**: Multiple backend instances with automatic failover
- **Fault Tolerance**: Circuit breaker patterns and health monitoring
- **Auto-scaling**: Horizontal Pod Autoscaler (HPA) for Kubernetes
- **Traffic Management**: Advanced routing, rate limiting, and DDoS protection
- **Session Persistence**: Sticky sessions for WebSocket connections
- **Zero-downtime Deployments**: Rolling updates and canary deployments

## Architecture Components

### 1. NGINX Load Balancer
- **File**: `nginx/nginx.conf`
- **Features**: Layer 7 load balancing, SSL termination, rate limiting
- **Strategies**: Round-robin, least connections, IP hash
- **Capacity**: Optimized for 10,000+ concurrent connections

### 2. HAProxy Load Balancer
- **File**: `haproxy/haproxy.cfg`
- **Features**: Advanced health checks, circuit breaker, statistics dashboard
- **Capacity**: High-performance Layer 7 load balancing
- **Monitoring**: Built-in stats at `:8404/haproxy-stats`

### 3. Istio Service Mesh
- **File**: `k8s/service-mesh.yaml`
- **Features**: Traffic policies, canary deployments, A/B testing
- **Security**: mTLS, authorization policies, JWT validation
- **Observability**: Distributed tracing and telemetry

### 4. Application-level Load Balancer
- **File**: `backend/app/load_balancer.py`
- **Features**: Client-side load balancing, service discovery
- **Patterns**: Circuit breaker, health monitoring, graceful shutdown

## Quick Start

### Local Development with Docker Compose

1. **Start the load-balanced environment**:
   ```bash
   cd backend
   docker-compose -f docker-compose.loadbalancer.yml up -d
   ```

2. **Access services**:
   - NGINX Load Balancer: http://localhost (port 80)
   - HAProxy Load Balancer: http://localhost:8000
   - HAProxy Stats: http://localhost:8404/haproxy-stats
   - Grafana Dashboard: http://localhost:3000
   - Prometheus: http://localhost:9090

3. **Scale backend instances**:
   ```bash
   docker-compose -f docker-compose.loadbalancer.yml up -d --scale backend-1=2 --scale backend-2=2
   ```

### Kubernetes Deployment

1. **Apply service mesh configuration**:
   ```bash
   kubectl apply -f k8s/service-mesh.yaml
   ```

2. **Enable auto-scaling**:
   ```bash
   ./scripts/scale-backend.sh auto-scale --min-replicas 3 --max-replicas 15
   ```

3. **Deploy with load balancer**:
   ```bash
   kubectl apply -f k8s/
   ```

## Configuration Details

### NGINX Configuration

**Backend Pools**:
- `backend_round_robin`: Default round-robin distribution
- `backend_least_conn`: CPU-intensive operations
- `backend_ip_hash`: Session affinity
- `websocket_backend`: WebSocket connections with stickiness
- `ai_backend`: AI services with longer timeouts

**Rate Limiting**:
- API endpoints: 10 req/s per IP
- Auth endpoints: 5 req/s per IP  
- WebSocket: 20 req/s per IP
- Burst handling with nodelay option

**Security Features**:
- SSL/TLS termination with HTTP/2
- Security headers (HSTS, CSP, etc.)
- DDoS protection
- Real IP detection for proxy chains

### HAProxy Configuration

**Load Balancing Strategies**:
- Round-robin for general API traffic
- Least connections for auth endpoints
- Source hash for WebSocket persistence
- Health-based routing with circuit breaker

**Advanced Features**:
- Stick tables for rate limiting
- Circuit breaker with automatic recovery
- A/B testing support
- Canary deployment backends

**Monitoring**:
- Statistics dashboard at `/haproxy-stats`
- Prometheus metrics export
- Health check endpoints

### Istio Service Mesh

**Traffic Management**:
- Virtual services for advanced routing
- Destination rules for traffic policies
- Gateway configuration for external access
- Circuit breaker and retry policies

**Security**:
- Mutual TLS (mTLS) between services
- Authorization policies
- JWT token validation
- Request authentication

**Observability**:
- Distributed tracing with Jaeger
- Telemetry collection
- Access logging
- Custom metrics

## Scaling Operations

### Manual Scaling

```bash
# Scale up to 5 replicas
./scripts/scale-backend.sh scale-up 5

# Scale down to 2 replicas
./scripts/scale-backend.sh scale-down 2

# Check current status
./scripts/scale-backend.sh status
```

### Auto-scaling Setup

```bash
# Enable auto-scaling
./scripts/scale-backend.sh auto-scale \
  --min-replicas 2 \
  --max-replicas 10 \
  --cpu-threshold 70 \
  --memory-threshold 80

# Disable auto-scaling
./scripts/scale-backend.sh disable-auto-scale
```

### Rolling Updates

```bash
# Perform rolling update
./scripts/scale-backend.sh rolling-update myregistry/automata-backend:v2.0.0

# Canary deployment (10% traffic)
./scripts/scale-backend.sh canary-deploy myregistry/automata-backend:v2.1.0

# Rollback if needed
./scripts/scale-backend.sh rollback
```

## Health Monitoring

### Health Check Endpoints

- `/health`: Comprehensive health status
- `/healthz`: Kubernetes-style liveness probe
- `/health/detailed`: Detailed component status
- `/metrics`: Prometheus metrics
- `/metrics/performance`: Performance statistics

### Application Metrics

The application exposes custom metrics for load balancer monitoring:

```python
# Connection tracking
automata_active_connections
automata_max_connections

# Circuit breaker status  
automata_circuit_breaker_open

# WebSocket metrics
automata_websocket_connections
automata_websocket_messages_total

# Database connection pool
automata_db_connections_active
automata_db_connections_idle
```

### Grafana Dashboard

The load balancer dashboard (`monitoring/load-balancer-dashboard.json`) provides:

- Request rate by backend instance
- Response time distribution
- Error rate tracking
- Connection pool status
- Circuit breaker monitoring
- Auto-scaling status
- Resource utilization

Import the dashboard:
1. Open Grafana (http://localhost:3000)
2. Login with admin/admin
3. Import dashboard from `monitoring/load-balancer-dashboard.json`

## Load Testing

### Using Locust

1. **Install Locust**:
   ```bash
   pip install locust
   ```

2. **Run load test**:
   ```bash
   cd load_tests
   locust --host http://localhost --users 100 --spawn-rate 10
   ```

3. **Automated testing**:
   ```bash
   ./scripts/scale-backend.sh load-test
   ```

### Test Scenarios

The load tests include:
- Normal API usage patterns
- High-frequency WebSocket connections
- AI service stress testing
- Authentication load testing
- Mixed traffic scenarios

## Troubleshooting

### Common Issues

1. **Backend instances not responding**:
   ```bash
   ./scripts/scale-backend.sh health-check
   kubectl get pods -l app=automata-backend
   ```

2. **High error rates**:
   ```bash
   # Check logs
   kubectl logs -l app=automata-backend --tail=100
   
   # Check metrics
   ./scripts/scale-backend.sh metrics
   ```

3. **Circuit breaker triggered**:
   ```bash
   # Check dashboard for circuit breaker status
   # Wait for automatic recovery (5 minutes)
   # Or restart affected pods
   kubectl rollout restart deployment automata-backend
   ```

4. **WebSocket connection issues**:
   - Ensure session affinity is enabled
   - Check load balancer sticky session configuration
   - Verify WebSocket headers are properly forwarded

### Log Analysis

**NGINX Logs**:
```bash
# Access logs with response times
tail -f logs/nginx/access.log

# Error logs
tail -f logs/nginx/error.log
```

**HAProxy Logs**:
```bash
# HAProxy logs include backend selection and timings
tail -f logs/haproxy/haproxy.log
```

**Application Logs**:
```bash
# Backend logs
kubectl logs -f deployment/automata-backend

# Load balancer component logs
grep "load_balancer" logs/backend-*/app.log
```

## Performance Tuning

### NGINX Tuning

```nginx
# Worker processes (auto-detected)
worker_processes auto;

# Increase worker connections
events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

# Connection keep-alive
keepalive_requests 100;
keepalive_timeout 65;

# Buffer sizes
proxy_buffers 8 4k;
proxy_buffer_size 4k;
```

### HAProxy Tuning

```
# Global settings
maxconn 40000
nbthread 4

# Backend settings  
balance leastconn
option httpchk GET /healthz
timeout server 60s
```

### Application Tuning

```python
# Connection pool settings
max_connections = 100
pool_size = 20
pool_timeout = 30

# Circuit breaker thresholds
failure_threshold = 5
recovery_timeout = 300
```

## Security Considerations

### Rate Limiting

- Implement progressive rate limiting
- Use different limits for different endpoints
- Monitor for rate limit violations

### DDoS Protection

- Configure connection limits per IP
- Implement request size limits
- Use geographic blocking if needed

### SSL/TLS Configuration

```nginx
# Strong SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:...;
ssl_prefer_server_ciphers off;

# Security headers
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Frame-Options "SAMEORIGIN" always;
```

## Maintenance

### Regular Tasks

1. **Monitor metrics and alerts**
2. **Review load balancer logs**
3. **Update SSL certificates**
4. **Test failover scenarios**
5. **Optimize based on traffic patterns**

### Backup and Recovery

- Configuration files are version controlled
- Database backups are automated
- Load balancer state is stateless
- Recovery procedures documented

## Support

For issues or questions:
1. Check logs and metrics first
2. Review this documentation
3. Test with load testing tools
4. Check Kubernetes events and pod status
5. Consult the monitoring dashboard

## File Structure

```
backend/
├── nginx/
│   ├── nginx.conf              # Main NGINX configuration
│   └── proxy_params            # Common proxy settings
├── haproxy/
│   ├── haproxy.cfg            # HAProxy configuration
│   └── rate_limits.map        # Rate limiting rules
├── app/
│   └── load_balancer.py       # Application load balancer
├── docker-compose.loadbalancer.yml  # Local testing
└── LOAD_BALANCING_SETUP.md    # This documentation

k8s/
└── service-mesh.yaml          # Istio configuration

scripts/
└── scale-backend.sh           # Scaling operations

monitoring/
└── load-balancer-dashboard.json  # Grafana dashboard
```

This comprehensive load balancing setup ensures high availability, fault tolerance, and scalability for the Automata Learning Platform while supporting 10,000+ concurrent connections.