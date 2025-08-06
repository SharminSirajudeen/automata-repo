# 🎓 Automata Learning Platform

## The World's Most Comprehensive Theory of Computation Educational Tool

An intelligent, interactive learning platform that surpasses JFLAP with AI-powered tutoring, formal verification, and modern web technologies. Built for MIT/Oxford-level education with complete JFLAP feature parity plus advanced AI capabilities.

### 📊 Project Status: **50% Complete**

✅ **What's Ready:**
- Production-ready architecture with FastAPI + React
- Complete JFLAP algorithm implementations (20+ algorithms)
- AI-powered tutoring with LangChain & ChromaDB
- Elite educational features (proofs, verification, pumping lemma)
- Google Drive integration for user data
- Research paper integration (12 seminal papers)

🔄 **In Progress:**
- Phase 4: UI/UX Excellence
- LangGraph migration for stateful tutoring

## 🚀 Quick Start

### Local Development (No AI Required)
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend  
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to start learning!

### With AI Features (Optional)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull AI models
ollama pull codellama:34b      # For code generation
ollama pull deepseek-coder:33b  # For explanations
ollama pull llama3.1:8b         # Fallback model

# Start Ollama
ollama serve
```

## ✨ Key Features

### 🤖 AI-Powered Learning
- **Natural Language to Automaton**: Describe in English, get formal automaton
- **Intelligent Tutoring**: Personalized hints and step-by-step guidance
- **Proof Assistant**: AI helps construct formal proofs
- **Adaptive Learning**: Adjusts difficulty based on performance
- **RAG System**: Context-aware responses with source attribution

### 🎯 Complete JFLAP Feature Parity
- **All Automata Types**: DFA, NFA, PDA, CFG, Turing Machines
- **Conversion Algorithms**: NFA→DFA, DFA minimization, Regex↔Automaton
- **Parsing**: CYK, LL(1), LR(0), SLR parsers
- **Grammar Operations**: CNF conversion, epsilon/unit removal
- **Import/Export**: Full .jff file compatibility

### 📚 Elite Educational Features
- **Interactive Proof System**: Build proofs with validation
- **Formal Verification**: Check automata equivalence and correctness
- **Pumping Lemma Visualizer**: Interactive xyz decomposition
- **Complexity Theory Module**: P vs NP visualizations
- **Research Papers**: Integrated with Sipser, Hopcroft, Turing papers

### 🎨 Modern Interface
- **Visual Automata Editor**: Drag-drop state creation
- **Step-by-Step Simulation**: Watch execution with highlighting
- **Batch Testing**: Test multiple strings simultaneously
- **Course Structure**: MIT/Oxford-level curriculum
- **Progress Tracking**: Google Drive sync (<200 bytes storage!)

## 🏗️ Architecture

### Technology Stack
```
Frontend:  React + TypeScript + Vite + Tailwind + shadcn/ui
Backend:   FastAPI + Python 3.11+ + Pydantic
AI:        Ollama + LangChain + ChromaDB + RAG
Storage:   PostgreSQL + Google Drive + Redis
Algorithms: 20+ JFLAP implementations + AI enhancements
```

### Project Structure
```
automata-repo/
├── frontend/               # React application
│   └── src/components/    # 15+ UI components
├── backend/               # FastAPI server
│   └── app/
│       ├── jflap_complete.py    # All JFLAP algorithms (2,273 lines)
│       ├── jflap_simulator.py   # Simulation engine (726 lines)
│       ├── ai_proof_assistant.py # Proof generation
│       ├── semantic_search.py   # ChromaDB integration
│       └── rag_system.py        # LangChain RAG
└── MASTER_ROADMAP.md      # Complete project documentation
```

## 📋 API Endpoints (75+)

### Core Automata
- `POST /api/analyze-problem` - Natural language to automaton
- `POST /api/jflap/nfa-to-dfa` - Convert NFA to DFA
- `POST /api/jflap/minimize` - Minimize DFA
- `POST /api/jflap/simulate` - Step-by-step simulation

### AI-Powered
- `POST /api/ai/proof/generate` - Generate formal proofs
- `POST /api/ai/tutor/hint` - Get personalized hints
- `POST /api/ai/rag/query` - RAG-powered Q&A
- `GET /api/learning/recommendations` - Adaptive learning

### Educational
- `POST /api/verification/equivalence` - Check automata equivalence
- `POST /api/pumping/validate` - Pumping lemma validation
- `GET /api/papers/recommend` - Research paper suggestions
- `GET /api/complexity/analyze` - Complexity analysis

## 🎯 Roadmap

**See `MASTER_ROADMAP.md` for detailed progress tracking and development plans.**

### Completed (50%)
- ✅ Phase 1: Production Readiness
- ✅ Phase 1.5: Google Drive Integration  
- ✅ Phase 2: Elite Educational Features
- ✅ Phase 3: Advanced AI Integration
- ✅ JFLAP Feature Parity

### Upcoming
- 🔄 LangGraph Migration (3-4 days)
- ⏳ Phase 4: UI/UX Excellence
- ⏳ Phase 5: Performance & Scale
- ⏳ Phase 6: Advanced Features (Mobile, Voice, AR/VR)

## 🧪 Testing

```bash
# Run all tests
cd backend && pytest

# Test JFLAP algorithms
python test_jflap_complete.py

# Frontend tests
cd frontend && npm test
```

## 🤝 Contributing

1. Check `MASTER_ROADMAP.md` for current priorities
2. Fork and create feature branch
3. Follow existing code patterns
4. Add tests for new features
5. Submit PR with clear description

## 📚 Documentation

- **`MASTER_ROADMAP.md`** - Complete project documentation and progress
- **`jflap_extracted_content.json`** - JFLAP knowledge base
- **`JFLAP_IMPLEMENTATION_COMPLETE.md`** - Algorithm documentation
- **API Docs:** `http://localhost:8000/docs` (when running)

## 🌟 What Makes This Special

1. **Beyond JFLAP**: All JFLAP features + AI tutoring + modern web
2. **Production Ready**: Clean architecture, tested, documented
3. **Educational Focus**: Built by educators for educators
4. **Open Source**: MIT licensed, community-driven
5. **AI-First**: Integrated AI throughout, not bolted on

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- JFLAP (Rodger & Finley, 2006) for inspiration
- Sipser's "Introduction to Theory of Computation"
- MIT/Oxford CS theory courses
- Open source community

---

**Created by:** Sharmin Sirajudeen  
**Status:** Active Development  
**Version:** 1.0.0-beta