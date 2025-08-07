# Testing Guide for Automata Learning Platform

## 🚀 Quick Start Testing

### 1. Quick Smoke Test (2 minutes)
```bash
# Make the test script executable
chmod +x scripts/test-local.sh

# Run quick smoke test
./scripts/test-local.sh
# Select option 9 for quick smoke test
```

This will:
- Check all dependencies
- Build the frontend
- Start services with Docker Compose
- Test health endpoints
- Verify basic functionality

### 2. Manual Testing Without Docker

#### Frontend Testing
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Open browser to http://localhost:5173
```

**Test these features:**
- ✅ Create a DFA/NFA/PDA/Turing Machine
- ✅ Run animations on state transitions
- ✅ Test the onboarding flow (click "Get Started")
- ✅ Try code splitting (navigate between routes)
- ✅ Test dark mode toggle
- ✅ Test accessibility (Tab navigation)

#### Backend Testing
```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
uvicorn app.main:app --reload --port 8000

# Test API at http://localhost:8000/docs
```

**Test these endpoints:**
- `GET /health` - Health check
- `GET /health/detailed` - Detailed health with metrics
- `POST /api/automata/create` - Create automaton
- `POST /api/ai/solve` - Test AI solver

### 3. Full Docker Testing

```bash
# Build and run with Docker Compose
docker-compose up --build

# Services will be available at:
# - Frontend: http://localhost:3000
# - Backend: http://localhost:8000
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379
```

## 🧪 Comprehensive Testing

### Run All Tests
```bash
./scripts/test-local.sh
# Select option 1
```

This comprehensive test suite will:
1. **Dependency Check** - Verify Node.js, Python, Docker are installed
2. **Frontend Tests** - TypeScript check, linting, build verification
3. **Backend Tests** - Python linting, type checking, unit tests
4. **Docker Builds** - Build production images
5. **Integration Tests** - Test API endpoints
6. **Performance Tests** - Lighthouse audit, load testing
7. **Security Checks** - Scan for exposed secrets, vulnerabilities

## 📊 Testing Checklist

### Frontend Features
- [ ] **Animation System**
  - [ ] State transitions animate smoothly
  - [ ] Play/pause/speed controls work
  - [ ] Export to GIF/video (framework ready)
  - [ ] Mobile responsive

- [ ] **Onboarding Flow**
  - [ ] Multi-step tutorial loads
  - [ ] Interactive demos work
  - [ ] Progress tracking
  - [ ] Skip functionality

- [ ] **Code Splitting**
  - [ ] Routes lazy load
  - [ ] Bundle size < 50KB initial
  - [ ] No chunk loading errors
  - [ ] Proper loading states

- [ ] **Performance**
  - [ ] Page loads < 3 seconds
  - [ ] No memory leaks
  - [ ] Smooth scrolling
  - [ ] 60fps animations

### Backend Features
- [ ] **Health Checks**
  - [ ] `/health` returns 200
  - [ ] `/health/detailed` shows metrics
  - [ ] Database connectivity
  - [ ] Redis connectivity

- [ ] **AI Features**
  - [ ] Problem solving works
  - [ ] Intelligent routing
  - [ ] Learning system active
  - [ ] Knowledge extraction

- [ ] **Load Balancing**
  - [ ] Multiple instances can run
  - [ ] Session persistence
  - [ ] WebSocket connections
  - [ ] Rate limiting

### Security
- [ ] No exposed secrets in code
- [ ] Input validation working
- [ ] CORS configured properly
- [ ] Security headers present
- [ ] HTTPS in production mode

## 🔍 Testing Individual Components

### Test Animation System
```javascript
// In browser console at http://localhost:5173
// Create a test automaton and run animation
const testDFA = {
  states: ['q0', 'q1', 'q2'],
  alphabet: ['0', '1'],
  start_state: 'q0',
  accept_states: ['q2'],
  transitions: [
    {from: 'q0', to: 'q1', symbol: '0'},
    {from: 'q1', to: 'q2', symbol: '1'}
  ]
};
// Check if animation runs smoothly
```

### Test AI Solver
```bash
# Test the intelligent solver
curl -X POST http://localhost:8000/api/ai/solve \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "Create a DFA that accepts strings with even number of 0s",
    "type": "DFA_CONSTRUCTION"
  }'
```

### Test Load Balancer
```bash
# Start multiple backend instances
docker-compose up --scale backend=3

# Test load distribution
for i in {1..10}; do
  curl http://localhost:8000/health
done
```

## 🐛 Debugging Common Issues

### Frontend Won't Start
```bash
# Clear node_modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Backend Import Errors
```bash
cd backend
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Docker Build Fails
```bash
# Clean Docker cache
docker system prune -a
docker-compose build --no-cache
```

### Port Already in Use
```bash
# Find and kill process using port
lsof -i :3000  # or :8000 for backend
kill -9 <PID>
```

## 🚢 Production Testing

### Test Production Build
```bash
# Build production images
docker build -f frontend/Dockerfile -t automata-frontend:prod ./frontend
docker build -f backend/Dockerfile -t automata-backend:prod ./backend

# Run production containers
docker run -p 3000:80 automata-frontend:prod
docker run -p 8000:8000 automata-backend:prod
```

### Load Testing
```bash
# Install Apache Bench
apt-get install apache2-utils  # Ubuntu/Debian
brew install httpd  # macOS

# Run load test
ab -n 1000 -c 10 http://localhost:8000/health
```

### Performance Testing
```bash
# Install Lighthouse
npm install -g lighthouse

# Run performance audit
lighthouse http://localhost:3000 --view
```

## 📝 Test Results Interpretation

### Expected Metrics
- **Bundle Size**: < 50KB initial, < 500KB total
- **Lighthouse Score**: > 90 for Performance
- **Load Time**: < 3s on 3G network
- **API Response**: < 200ms average
- **Memory Usage**: < 100MB for frontend
- **CPU Usage**: < 50% under normal load

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:00:00Z",
  "version": "1.0.0",
  "services": {
    "database": "connected",
    "redis": "connected",
    "ai_service": "ready"
  },
  "metrics": {
    "memory_usage_mb": 45.2,
    "cpu_percent": 12.5,
    "active_connections": 5,
    "request_rate": 10.5
  }
}
```

## 🎯 Ready for Deployment?

If all tests pass:
1. ✅ All health checks return 200
2. ✅ No security vulnerabilities found
3. ✅ Performance metrics meet targets
4. ✅ Docker images build successfully
5. ✅ Integration tests pass

Then you're ready to deploy to vast.ai:
```bash
./scripts/deploy-to-vast.sh
```

## 💡 Tips

- Run tests in order: Unit → Integration → E2E → Performance
- Test on different browsers (Chrome, Firefox, Safari)
- Test on mobile devices or responsive mode
- Monitor memory usage during long sessions
- Check network tab for failed requests
- Review console for errors or warnings

## 📞 Getting Help

If tests fail:
1. Check the error logs in `docker-compose logs`
2. Review the TEST_GUIDE.md for specific component testing
3. Check GitHub Issues for known problems
4. Review the code with security scanner

Happy Testing! 🎉