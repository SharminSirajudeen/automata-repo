import { useState } from 'react';
import { ProblemSelector } from './components/ProblemSelector';
import ComprehensiveProblemView from './components/ComprehensiveProblemView';
import { ProblemInput } from './components/ProblemInput';
import { Problem } from './types/automata';
import { Button } from './components/ui/button';
import { BookOpen, Sparkles } from 'lucide-react';
import './App.css';

function App() {
  const [selectedProblem, setSelectedProblem] = useState<Problem | null>(null);
  const [mode, setMode] = useState<'selector' | 'input'>('selector');
  const [isProcessingProblem, setIsProcessingProblem] = useState(false);

  const handleSelectProblem = (problem: Problem) => {
    setSelectedProblem(problem);
  };

  const handleBackToProblems = () => {
    setSelectedProblem(null);
    setMode('selector');
  };

  const handleProblemSubmit = async (problemText: string, type: 'text' | 'image') => {
    setIsProcessingProblem(true);
    
    try {
      const response = await fetch('http://localhost:8000/api/analyze-problem', {
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
      } else {
        alert(result.message || 'This does not appear to be a Theory of Computation problem.');
      }
    } catch (error) {
      console.error('Error analyzing problem:', error);
      alert('Error analyzing the problem. Please try again.');
    } finally {
      setIsProcessingProblem(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {selectedProblem ? (
        <ComprehensiveProblemView 
          problem={selectedProblem} 
          onBack={handleBackToProblems}
        />
      ) : (
        <div>
          <div className="bg-white shadow-sm border-b">
            <div className="max-w-7xl mx-auto px-4 py-4">
              <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-gray-900">
                  Theory of Computation Tutor
                </h1>
                <div className="flex gap-2">
                  <Button
                    variant={mode === 'selector' ? 'default' : 'outline'}
                    onClick={() => setMode('selector')}
                    className="flex items-center gap-2"
                  >
                    <BookOpen className="h-4 w-4" />
                    Practice Problems
                  </Button>
                  <Button
                    variant={mode === 'input' ? 'default' : 'outline'}
                    onClick={() => setMode('input')}
                    className="flex items-center gap-2"
                  >
                    <Sparkles className="h-4 w-4" />
                    AI Problem Solver
                  </Button>
                </div>
              </div>
            </div>
          </div>

          <div className="max-w-7xl mx-auto px-4 py-8">
            {mode === 'selector' ? (
              <ProblemSelector onSelectProblem={handleSelectProblem} />
            ) : (
              <ProblemInput 
                onProblemSubmit={handleProblemSubmit}
                isProcessing={isProcessingProblem}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
