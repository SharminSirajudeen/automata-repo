import { useState } from 'react';
import { ProblemSelector } from './components/ProblemSelector';
import { ComprehensiveProblemView } from './components/ComprehensiveProblemView';
import { Problem } from './types/automata';
import './App.css';

function App() {
  const [selectedProblem, setSelectedProblem] = useState<Problem | null>(null);

  const handleSelectProblem = (problem: Problem) => {
    setSelectedProblem(problem);
  };

  const handleBackToProblems = () => {
    setSelectedProblem(null);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {selectedProblem ? (
        <ComprehensiveProblemView 
          problem={selectedProblem} 
          onBack={handleBackToProblems}
        />
      ) : (
        <ProblemSelector onSelectProblem={handleSelectProblem} />
      )}
    </div>
  );
}

export default App;
