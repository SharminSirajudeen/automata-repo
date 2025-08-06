import { useState, useEffect } from 'react';
import { ProblemSelector } from './components/ProblemSelector';
import ComprehensiveProblemView from './components/ComprehensiveProblemView';
import { ProblemInput } from './components/ProblemInput';
import { SkipNavigation } from './components/SkipNavigation';
import { AccessibilitySettings } from './components/AccessibilitySettings';
import { CollaborationProvider } from './components/CollaborationProvider';
import { CollaborativeWorkspace } from './components/CollaborativeWorkspace';
import { Problem } from './types/automata';
import { Button } from './components/ui/button';
import { ThemeProvider, useTheme } from './contexts/ThemeProvider';
import { AccessibilityProvider, useAccessibility } from './contexts/AccessibilityProvider';
import { BookOpen, Sparkles, Moon, Sun, Settings, Users } from 'lucide-react';
import './App.css';

// Main App Content Component
function AppContent() {
  const [selectedProblem, setSelectedProblem] = useState<Problem | null>(null);
  const [mode, setMode] = useState<'selector' | 'input' | 'collaborate'>('selector');
  const [isProcessingProblem, setIsProcessingProblem] = useState(false);
  
  const { resolvedTheme, toggleTheme } = useTheme();
  const { announceMessage, addShortcut } = useAccessibility();

  // Initialize theme toggle shortcut
  useEffect(() => {
    const handleToggleTheme = () => {
      toggleTheme();
      announceMessage(`Theme switched to ${resolvedTheme === 'dark' ? 'light' : 'dark'} mode`);
    };

    // Add theme toggle to accessibility shortcuts
    addShortcut({
      key: 't',
      altKey: true,
      action: 'toggle-theme',
      description: 'Toggle light/dark theme',
      handler: handleToggleTheme
    });

    // Listen for custom theme toggle events
    const handleCustomToggle = () => handleToggleTheme();
    window.addEventListener('toggle-theme', handleCustomToggle);
    
    return () => {
      window.removeEventListener('toggle-theme', handleCustomToggle);
    };
  }, [toggleTheme, resolvedTheme, announceMessage, addShortcut]);

  const handleSelectProblem = (problem: Problem) => {
    setSelectedProblem(problem);
    announceMessage(`Selected problem: ${problem.title}`);
  };

  const handleBackToProblems = () => {
    setSelectedProblem(null);
    setMode('selector');
    announceMessage('Returned to problem selection');
  };

  const handleProblemSubmit = async (problemText: string, type: 'text' | 'image') => {
    setIsProcessingProblem(true);
    announceMessage('Analyzing problem, please wait...');
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/analyze-problem`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem_text: problemText,
          type: type
        }),
      });

      const result = await response.json();
      
      if (result.is_toc_problem) {
        const aiGeneratedProblem: Problem = {
          id: `ai-${Date.now()}`,
          type: result.problem_type,
          title: `AI Problem: ${result.problem_type.toUpperCase()}`,
          description: result.problem_description,
          language_description: result.problem_description,
          alphabet: ['a', 'b'],
          test_strings: result.test_cases.accept.map((str: string) => ({ string: str, should_accept: true }))
            .concat(result.test_cases.reject.map((str: string) => ({ string: str, should_accept: false }))),
          hints: result.guided_steps || [],
          difficulty: result.difficulty,
          category: result.problem_type.toUpperCase()
        };
        
        setSelectedProblem(aiGeneratedProblem);
        announceMessage(`Problem analyzed successfully: ${result.problem_type} problem created`);
      } else {
        const errorMessage = result.message || 'This does not appear to be a Theory of Computation problem.';
        alert(errorMessage);
        announceMessage(errorMessage);
      }
    } catch (error) {
      console.error('Error analyzing problem:', error);
      const errorMessage = 'Error analyzing the problem. Please try again.';
      alert(errorMessage);
      announceMessage(errorMessage);
    } finally {
      setIsProcessingProblem(false);
    }
  };

  const handleModeChange = (newMode: 'selector' | 'input' | 'collaborate') => {
    setMode(newMode);
    const modeNames = { 
      selector: 'Practice Problems', 
      input: 'AI Problem Solver',
      collaborate: 'Collaborative Workspace'
    };
    announceMessage(`Switched to ${modeNames[newMode]} mode`);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <SkipNavigation />
      
      {selectedProblem ? (
        <ComprehensiveProblemView 
          problem={selectedProblem} 
          onBack={handleBackToProblems}
        />
      ) : (
        <div>
          {/* Header with navigation */}
          <header 
            id="header"
            className="bg-card shadow-sm border-b border-border"
            role="banner"
          >
            <div className="max-w-7xl mx-auto px-4 py-4">
              <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-foreground">
                  Theory of Computation Tutor
                </h1>
                
                {/* Navigation and Controls */}
                <nav 
                  id="navigation"
                  className="flex gap-2 items-center"
                  aria-label="Main navigation"
                >
                  {/* Mode Toggle Buttons */}
                  <div className="flex gap-2" role="group" aria-label="Mode selection">
                    <Button
                      variant={mode === 'selector' ? 'default' : 'outline'}
                      onClick={() => handleModeChange('selector')}
                      className="flex items-center gap-2"
                      aria-pressed={mode === 'selector'}
                      aria-describedby="practice-problems-desc"
                    >
                      <BookOpen className="h-4 w-4" aria-hidden="true" />
                      Practice Problems
                    </Button>
                    <div id="practice-problems-desc" className="sr-only">
                      Browse and solve predefined Theory of Computation problems
                    </div>
                    
                    <Button
                      variant={mode === 'input' ? 'default' : 'outline'}
                      onClick={() => handleModeChange('input')}
                      className="flex items-center gap-2"
                      aria-pressed={mode === 'input'}
                      aria-describedby="ai-solver-desc"
                    >
                      <Sparkles className="h-4 w-4" aria-hidden="true" />
                      AI Problem Solver
                    </Button>
                    <div id="ai-solver-desc" className="sr-only">
                      Input custom problems and get AI-powered solutions
                    </div>
                    
                    <Button
                      variant={mode === 'collaborate' ? 'default' : 'outline'}
                      onClick={() => handleModeChange('collaborate')}
                      className="flex items-center gap-2"
                      aria-pressed={mode === 'collaborate'}
                      aria-describedby="collaborate-desc"
                    >
                      <Users className="h-4 w-4" aria-hidden="true" />
                      Collaborate
                    </Button>
                    <div id="collaborate-desc" className="sr-only">
                      Real-time collaborative workspace with other users
                    </div>
                  </div>

                  {/* Theme Toggle */}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={toggleTheme}
                    className="gap-2"
                    aria-label={`Switch to ${resolvedTheme === 'dark' ? 'light' : 'dark'} theme`}
                    title={`Current theme: ${resolvedTheme}. Click to toggle.`}
                  >
                    {resolvedTheme === 'dark' ? (
                      <Sun className="h-4 w-4" aria-hidden="true" />
                    ) : (
                      <Moon className="h-4 w-4" aria-hidden="true" />
                    )}
                    <span className="sr-only">Toggle theme</span>
                  </Button>

                  {/* Accessibility Settings */}
                  <AccessibilitySettings 
                    trigger={
                      <Button
                        variant="ghost"
                        size="sm"
                        className="gap-2"
                        aria-label="Open accessibility settings"
                      >
                        <Settings className="h-4 w-4" aria-hidden="true" />
                        <span className="sr-only">Accessibility Settings</span>
                      </Button>
                    }
                  />
                </nav>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main 
            id="main-content"
            className="max-w-7xl mx-auto px-4 py-8"
            role="main"
            aria-label="Main content"
            tabIndex={-1}
          >
            {/* Live region for dynamic content announcements */}
            <div 
              aria-live="polite" 
              aria-atomic="true" 
              className="sr-only"
              id="content-status"
            >
              {mode === 'selector' && 'Practice problems loaded'}
              {mode === 'input' && 'AI problem solver ready'}
              {mode === 'collaborate' && 'Collaborative workspace ready'}
            </div>

            {mode === 'selector' && (
              <section aria-labelledby="practice-problems-heading">
                <h2 id="practice-problems-heading" className="sr-only">
                  Practice Problems Section
                </h2>
                <ProblemSelector onSelectProblem={handleSelectProblem} />
              </section>
            )}
            
            {mode === 'input' && (
              <section aria-labelledby="ai-solver-heading">
                <h2 id="ai-solver-heading" className="sr-only">
                  AI Problem Solver Section
                </h2>
                <ProblemInput 
                  onProblemSubmit={handleProblemSubmit}
                  isProcessing={isProcessingProblem}
                />
              </section>
            )}
            
            {mode === 'collaborate' && (
              <section aria-labelledby="collaborate-heading">
                <h2 id="collaborate-heading" className="sr-only">
                  Collaborative Workspace Section
                </h2>
                <CollaborativeWorkspace 
                  problem={selectedProblem}
                  onProblemChange={setSelectedProblem}
                />
              </section>
            )}
          </main>

          {/* Footer for additional navigation */}
          <footer 
            id="footer"
            className="mt-16 border-t border-border bg-muted/50"
            role="contentinfo"
          >
            <div className="max-w-7xl mx-auto px-4 py-6">
              <div className="text-center text-sm text-muted-foreground">
                <p>
                  Theory of Computation Tutor - 
                  <span className="ml-1">Accessible learning platform with WCAG AAA compliance</span>
                </p>
                <div className="mt-2 space-x-4">
                  <span>Press Alt+T to toggle theme</span>
                  <span>•</span>
                  <span>Press ? for keyboard shortcuts</span>
                  <span>•</span>
                  <span>Press S to skip to main content</span>
                </div>
              </div>
            </div>
          </footer>
        </div>
      )}
    </div>
  );
}

// Root App Component with Providers
function App() {
  return (
    <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
      <AccessibilityProvider>
        <CollaborationProvider
          userId={`user_${Date.now()}`} // In production, get from auth
          username="Anonymous User"
          autoConnect={true}
        >
          <AppContent />
        </CollaborationProvider>
      </AccessibilityProvider>
    </ThemeProvider>
  );
}

export default App;
