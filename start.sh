#!/bin/bash

# Automata-Repo Quick Start Script
echo "🚀 Starting Automata-Repo Development Environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f backend/.env ]; then
    echo "📝 Creating backend/.env file..."
    cp backend/.env.example backend/.env
    echo "⚠️  Please update backend/.env with your configuration"
fi

# Create frontend .env file if it doesn't exist
if [ ! -f frontend/.env ]; then
    echo "📝 Creating frontend/.env file..."
    cat > frontend/.env << EOF
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=http://localhost:9999
VITE_SUPABASE_ANON_KEY=your-anon-key-here
EOF
    echo "⚠️  Please update frontend/.env with your configuration"
fi

# Start Docker Compose services
echo "🐳 Starting Docker services..."
docker-compose up -d postgres redis

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 5

# Initialize the database
echo "🗄️  Initializing database..."
docker-compose run --rm backend python scripts/init_db.py

# Start all services
echo "🎯 Starting all services..."
docker-compose up -d

# Show service status
echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Access the application at:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo "   - API Documentation: http://localhost:8000/docs"
echo ""
echo "📋 Default credentials (development only):"
echo "   - PostgreSQL: automata/automata_password"
echo "   - Database: automata_db"
echo ""
echo "🛑 To stop all services: docker-compose down"
echo "📊 To view logs: docker-compose logs -f"
echo ""
echo "⚡ Make sure Ollama is running locally for AI features!"
echo "   Run: ollama serve"
echo ""