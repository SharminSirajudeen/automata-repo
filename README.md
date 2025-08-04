# Theory of Computation Tutor

An intelligent, interactive learning platform for Theory of Computation with AI-powered guidance using Ollama.

## Features

### 🤖 AI-Powered Learning
- **Real-time AI Guidance**: Step-by-step assistance powered by Ollama with uncensored mathematical models
- **Interactive Professor Mode**: AI acts like a professor, guiding students through automata construction
- **Personalized Feedback**: Detailed explanations and hints based on student progress
- **Smart Error Detection**: AI identifies common mistakes and provides targeted corrections

### 🎯 Interactive Automata Builder
- **Visual Canvas**: Drag-and-drop interface for building DFAs, NFAs, and more
- **Real-time Validation**: Instant feedback on automaton correctness
- **Step-by-step Simulation**: Watch your automaton process strings step by step
- **Multiple Problem Types**: DFA construction, language recognition, and more

### 📚 Educational Features
- **Progressive Difficulty**: Problems ranging from beginner to advanced
- **Comprehensive Test Cases**: Multiple test strings with detailed results
- **Hint System**: Built-in hints plus AI-generated personalized guidance
- **Progress Tracking**: Monitor learning progress and completion rates

### 🎨 Modern Interface
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Intuitive Controls**: Easy-to-use tools for state and transition creation
- **Visual Feedback**: Clear indicators for start states, accept states, and transitions
- **Accessibility**: Keyboard navigation and screen reader support

## Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **Ollama Integration**: AI-powered feedback and guidance
- **Pydantic**: Data validation and serialization
- **In-memory Database**: Fast prototyping with persistent storage planned

### Frontend
- **React 18**: Modern React with hooks and TypeScript
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **shadcn/ui**: Beautiful, accessible UI components
- **Lucide Icons**: Consistent iconography

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js 18+
- Ollama (for AI features)

### Backend Setup
```bash
cd backend
poetry install
poetry run fastapi dev app/main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Ollama Setup (for AI features)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a mathematical model (example)
ollama pull llama3.1:8b

# Start Ollama service
ollama serve
```

## API Endpoints

### Problems
- `GET /problems` - List all available problems
- `GET /problems/{id}` - Get specific problem details
- `POST /problems/{id}/validate` - Validate automaton solution
- `GET /problems/{id}/hint` - Get built-in hints

### AI Features
- `POST /problems/{id}/ai-hint` - Get personalized AI guidance
- `GET /ai/status` - Check AI service availability

## Problem Types

### Currently Supported
- **DFA Construction**: Build deterministic finite automata
- **Language Recognition**: Create automata for specific languages
- **Pattern Matching**: Automata for string patterns

### Planned Features
- **NFA Construction**: Non-deterministic finite automata
- **Regular Expressions**: Convert between regex and automata
- **Context-Free Grammars**: Pushdown automata construction
- **Turing Machines**: Basic Turing machine simulation

## AI Integration

The platform integrates with Ollama to provide intelligent tutoring:

1. **Real-time Guidance**: As students build automata, the AI provides contextual hints
2. **Error Analysis**: AI analyzes incorrect solutions and explains mistakes
3. **Step-by-step Teaching**: Progressive guidance through complex problems
4. **Adaptive Learning**: AI adjusts difficulty based on student performance

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by automata-tutor.live-lab.fi.muni.cz
- Built with modern web technologies for optimal learning experience
- AI integration powered by Ollama for intelligent tutoring

---

**Link to Devin run**: https://app.devin.ai/sessions/3c3d9d679cd345ac8045f6db1377d51e
**Requested by**: Sharmin Sirajudeen (@SharminSirajudeen)
