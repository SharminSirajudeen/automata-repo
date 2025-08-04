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

### AI Setup

1. Install Ollama:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. Pull the specialized models:
```bash
# For automaton generation and formal definitions
ollama pull codellama:34b

# For educational explanations and tutoring
ollama pull deepseek-coder:33b

# Optional: Keep the lightweight model for basic features
ollama pull llama3.1:8b
```

3. Start Ollama service:
```bash
ollama serve
```

The backend automatically orchestrates between models based on task type. The system works with any combination of available models and provides fallbacks when models are unavailable.

### API Endpoints

#### Core Endpoints
- `GET /problems` - List all available problems
- `GET /problems/{id}` - Get specific problem details
- `POST /problems/{id}/validate` - Validate automaton solution
- `GET /problems/{id}/hint` - Get built-in hints

#### AI-Powered Endpoints
- `POST /problems/{id}/ai-hint` - Get personalized guidance (deepseek-coder:33b)
- `POST /problems/{id}/generate-solution` - Generate complete automaton solution (codellama:34b)
- `POST /problems/{id}/explain-solution` - Get detailed explanation of user's solution (deepseek-coder:33b)
- `GET /ai/status` - Check AI service availability and model status

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

## 🤖 Multi-Model AI Features

The platform uses specialized AI models for different educational tasks:

### 🧠 **codellama:34b** - Automaton Generation
- Formal automaton definitions (states, transitions, alphabet)
- Complete Python implementations for simulation
- DOT graph code for visualization
- Comprehensive test case generation

### 📘 **deepseek-coder:33b** - Educational Explanations  
- Step-by-step reasoning and explanations
- Conceptual insights about automata theory
- Real-time tutoring and guidance
- Educational feedback on student solutions

### AI Capabilities

- **Smart Solution Generation**: Complete automaton solutions with formal definitions
- **Interactive Tutoring**: Real-time guidance as students build automata
- **Solution Analysis**: Detailed explanations of user-created automata
- **Personalized Hints**: Context-aware suggestions based on current progress
- **Educational Insights**: Deep explanations of automata theory concepts
- **Graceful Fallbacks**: Works without AI when models are offline

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
