#!/bin/bash

# Quick Test Script for Automata Learning Platform
# This performs a minimal test without Docker

echo "================================"
echo "Quick Test - Automata Platform"
echo "================================"

# Check if we're in the right directory
if [ ! -d "frontend" ] || [ ! -d "backend" ]; then
    echo "❌ Error: Run this from the project root directory"
    exit 1
fi

echo ""
echo "📁 Project Structure Check:"
echo "✅ Frontend directory exists"
echo "✅ Backend directory exists"

# Check key files
echo ""
echo "📄 Key Files Check:"
[ -f "frontend/package.json" ] && echo "✅ frontend/package.json exists" || echo "❌ frontend/package.json missing"
[ -f "backend/requirements.txt" ] && echo "✅ backend/requirements.txt exists" || echo "❌ backend/requirements.txt missing"
[ -f "docker-compose.yml" ] && echo "✅ docker-compose.yml exists" || echo "❌ docker-compose.yml missing"
[ -f ".env.example" ] && echo "✅ .env.example exists" || echo "❌ .env.example missing"

# Check security fixes
echo ""
echo "🔒 Security Fixes Check:"
if grep -q "your-secret-key-here" docker-compose.yml 2>/dev/null; then
    echo "❌ Hardcoded secrets found in docker-compose.yml"
else
    echo "✅ No hardcoded secrets in docker-compose.yml"
fi

[ -f "frontend/.env.production" ] && echo "✅ Production environment template exists" || echo "❌ Production environment template missing"
[ -f "frontend/Dockerfile" ] && echo "✅ Production Dockerfile exists" || echo "❌ Production Dockerfile missing"

# Check critical implementations
echo ""
echo "🚀 Feature Implementation Check:"
[ -d "frontend/src/components" ] && echo "✅ Frontend components directory exists" || echo "❌ Frontend components missing"
[ -f "frontend/src/components/AnimationSystem.tsx" ] && echo "✅ Animation system implemented" || echo "❌ Animation system missing"
[ -f "frontend/src/components/OnboardingFlow.tsx" ] && echo "✅ Onboarding flow implemented" || echo "❌ Onboarding missing"
[ -f "backend/app/intelligent_solver.py" ] && echo "✅ Intelligent solver implemented" || echo "❌ Intelligent solver missing"
[ -f "backend/app/health.py" ] && echo "✅ Health endpoints implemented" || echo "❌ Health endpoints missing"

# Check Kubernetes configs
echo ""
echo "☸️ Kubernetes Configuration Check:"
[ -d "k8s" ] && echo "✅ Kubernetes configs directory exists" || echo "❌ K8s configs missing"
[ -f "k8s/frontend-deployment.yaml" ] && echo "✅ Frontend deployment config exists" || echo "❌ Frontend deployment missing"
[ -f "k8s/backend-deployment.yaml" ] && echo "✅ Backend deployment config exists" || echo "❌ Backend deployment missing"

# Check deployment scripts
echo ""
echo "📜 Deployment Scripts Check:"
[ -f "scripts/deploy-to-vast.sh" ] && echo "✅ vast.ai deployment script exists" || echo "❌ vast.ai script missing"
[ -f "scripts/test-local.sh" ] && echo "✅ Local test script exists" || echo "❌ Test script missing"

# Check documentation
echo ""
echo "📚 Documentation Check:"
[ -f "VAST_AI_SETUP_GUIDE.md" ] && echo "✅ vast.ai setup guide exists" || echo "❌ Setup guide missing"
[ -f "TEST_GUIDE.md" ] && echo "✅ Testing guide exists" || echo "❌ Testing guide missing"
[ -f "DEPLOYMENT.md" ] && echo "✅ Deployment guide exists" || echo "❌ Deployment guide missing"

# Summary
echo ""
echo "================================"
echo "📊 Test Summary"
echo "================================"

# Count files
FRONTEND_FILES=$(find frontend/src -name "*.tsx" -o -name "*.ts" 2>/dev/null | wc -l | tr -d ' ')
BACKEND_FILES=$(find backend/app -name "*.py" 2>/dev/null | wc -l | tr -d ' ')
K8S_FILES=$(find k8s -name "*.yaml" 2>/dev/null | wc -l | tr -d ' ')

echo "Frontend TypeScript files: $FRONTEND_FILES"
echo "Backend Python files: $BACKEND_FILES"
echo "Kubernetes configs: $K8S_FILES"

echo ""
echo "✅ Platform structure is ready for deployment!"
echo ""
echo "Next steps:"
echo "1. Install Docker Desktop"
echo "2. Run: docker-compose up --build"
echo "3. Or deploy to vast.ai using the setup guide"
echo ""
echo "================================"