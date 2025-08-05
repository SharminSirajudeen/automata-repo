import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { ArrowLeft, Play, CheckCircle, AlertCircle, Brain, Code2, Zap, Search } from 'lucide-react';
import { Problem, ExtendedAutomaton, ValidationResult, AutomataType } from '../types/automata';
import { AutomataCanvas } from './AutomataCanvas';
import { AIAssistantPanel } from './AIAssistantPanel';
import { SimulationEngine } from './SimulationEngine';
import { CodeExporter } from './CodeExporter';
import { AutomataInspector } from './AutomataInspector';
import { LearningMode } from './LearningMode';
import { ProjectManager } from './ProjectManager';
import { ExampleGallery } from './ExampleGallery';
import { ProofAssistant } from './ProofAssistant';
const apiService = {
  validateSolution: async (problemId: string, solution: any) => {
    const response = await fetch(`/api/problems/${problemId}/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(solution)
    });
    return response.json();
  }
};

interface ComprehensiveProblemViewProps {
  problem: Problem;
  onBack: () => void;
}

const ComprehensiveProblemView: React.FC<ComprehensiveProblemViewProps> = ({
  problem,
  onBack
}) => {
  const [automaton, setAutomaton] = useState<ExtendedAutomaton>(() => {
    const baseAutomaton = {
      states: [],
      transitions: [],
      alphabet: problem.alphabet
    };

    switch (problem.type) {
      case 'pda':
        return {
          ...baseAutomaton,
          type: 'pda',
          stack_alphabet: ['Z', 'A', 'B'],
          start_stack_symbol: 'Z'
        } as ExtendedAutomaton;
      case 'cfg':
        return {
          type: 'cfg',
          terminals: problem.alphabet,
          non_terminals: ['S', 'A', 'B'],
          productions: [],
          start_symbol: 'S'
        } as ExtendedAutomaton;
      case 'tm':
        return {
          ...baseAutomaton,
          type: 'tm',
          tape_alphabet: [...problem.alphabet, '_'],
          blank_symbol: '_',
          num_tapes: 1
        } as ExtendedAutomaton;
      case 'regex':
        return {
          type: 'regex',
          pattern: '',
          alphabet: problem.alphabet
        } as ExtendedAutomaton;
      case 'pumping':
        return {
          type: 'pumping',
          language_type: 'regular',
          language_description: problem.language_description
        } as ExtendedAutomaton;
      default:
        return {
          ...baseAutomaton,
          type: problem.type as AutomataType
        } as ExtendedAutomaton;
    }
  });

  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [activeTab, setActiveTab] = useState('canvas');
  const [isGeneratingComplete, setIsGeneratingComplete] = useState(false);
  const [isGeneratingGuided, setIsGeneratingGuided] = useState(false);
  const [completeSolution, setCompleteSolution] = useState<any>(null);
  const [guidedSteps, setGuidedSteps] = useState<string[]>([]);

  const handleAutomatonChange = (newAutomaton: ExtendedAutomaton) => {
    console.log('Loading new automaton:', newAutomaton);
    setAutomaton(newAutomaton);
    setValidationResult(null);
  };

  const handleValidate = async () => {
    setIsValidating(true);
    try {
      const result = await apiService.validateSolution(problem.id, {
        problem_id: problem.id,
        automaton: automaton,
        user_id: 'user'
      });
      setValidationResult(result);
    } catch (error) {
      console.error('Validation error:', error);
    } finally {
      setIsValidating(false);
    }
  };

  const handleCompleteSolution = async () => {
    setIsGeneratingComplete(true);
    try {
      const response = await fetch('/api/complete-solution', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task: problem.description,
          problem_type: problem.type,
          problem_id: problem.id
        })
      });
      const result = await response.json();
      setCompleteSolution(result);
      if (result.automaton) {
        handleAutomatonChange(result.automaton);
      }
    } catch (error) {
      console.error('Complete solution error:', error);
    } finally {
      setIsGeneratingComplete(false);
    }
  };

  const handleGuidedApproach = async () => {
    setIsGeneratingGuided(true);
    try {
      const response = await fetch('/api/guided-approach', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task: problem.description,
          problem_type: problem.type,
          current_progress: automaton
        })
      });
      const result = await response.json();
      setGuidedSteps(result.steps || []);
    } catch (error) {
      console.error('Guided approach error:', error);
    } finally {
      setIsGeneratingGuided(false);
    }
  };

  const getTypeDisplayName = (type: AutomataType) => {
    const typeNames: { [key in AutomataType]: string } = {
      'dfa': 'DFA',
      'nfa': 'NFA',
      'enfa': 'ε-NFA',
      'pda': 'PDA',
      'cfg': 'CFG',
      'tm': 'Turing Machine',
      'regex': 'Regular Expression',
      'pumping': 'Pumping Lemma'
    };
    return typeNames[type];
  };

  const getDifficultyColor = (difficulty?: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-6">
          <Button
            onClick={onBack}
            variant="outline"
            className="mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Problems
          </Button>
          
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                {problem.title}
              </h1>
              <div className="flex gap-2 mb-4">
                <Badge variant="secondary">
                  {getTypeDisplayName(problem.type)}
                </Badge>
                {problem.difficulty && (
                  <Badge className={getDifficultyColor(problem.difficulty)}>
                    {problem.difficulty}
                  </Badge>
                )}
                {problem.category && (
                  <Badge variant="outline">
                    {problem.category}
                  </Badge>
                )}
              </div>
              <p className="text-gray-600 max-w-3xl">
                {problem.description}
              </p>
              <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <h3 className="font-medium text-blue-900 mb-2">Language Definition:</h3>
                <p className="text-blue-800">{problem.language_description}</p>
                <div className="mt-2">
                  <span className="text-sm font-medium text-blue-900">Alphabet: </span>
                  <span className="text-blue-800">{problem.alphabet.join(', ')}</span>
                </div>
              </div>
            </div>
            
            <div className="flex gap-2">
              <Button
                onClick={handleCompleteSolution}
                disabled={isGeneratingComplete}
                className="bg-blue-600 hover:bg-blue-700"
              >
                {isGeneratingComplete ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4 mr-2" />
                    Complete Solution
                  </>
                )}
              </Button>
              <Button
                onClick={handleGuidedApproach}
                disabled={isGeneratingGuided}
                variant="outline"
                className="border-blue-600 text-blue-600 hover:bg-blue-50"
              >
                {isGeneratingGuided ? (
                  <>
                    <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin mr-2" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4 mr-2" />
                    Guided Approach
                  </>
                )}
              </Button>
              <Button
                onClick={handleValidate}
                disabled={isValidating}
                className="bg-green-600 hover:bg-green-700"
              >
                {isValidating ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                    Validating...
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Validate Solution
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>

        {validationResult && (
          <div className={`mb-6 p-4 rounded-lg border ${
            validationResult.is_correct 
              ? 'bg-green-50 border-green-200' 
              : 'bg-red-50 border-red-200'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              {validationResult.is_correct ? (
                <CheckCircle className="w-5 h-5 text-green-600" />
              ) : (
                <AlertCircle className="w-5 h-5 text-red-600" />
              )}
              <span className={`font-medium ${
                validationResult.is_correct ? 'text-green-800' : 'text-red-800'
              }`}>
                {validationResult.is_correct ? 'Correct!' : 'Needs Improvement'}
              </span>
              <Badge variant="outline" className="ml-auto">
                Score: {Math.round(validationResult.score * 100)}%
              </Badge>
            </div>
            <p className={`text-sm ${
              validationResult.is_correct ? 'text-green-700' : 'text-red-700'
            }`}>
              {validationResult.feedback}
            </p>
            {validationResult.ai_explanation && (
              <div className="mt-3 p-3 bg-white rounded border">
                <h4 className="font-medium text-gray-900 mb-1">AI Explanation:</h4>
                <p className="text-sm text-gray-700">{validationResult.ai_explanation}</p>
              </div>
            )}
          </div>
        )}

        {completeSolution && (
          <div className="mb-6 p-4 rounded-lg border bg-blue-50 border-blue-200">
            <div className="flex items-center gap-2 mb-2">
              <Brain className="w-5 h-5 text-blue-600" />
              <span className="font-medium text-blue-800">Complete Solution Generated</span>
            </div>
            {completeSolution.explanation && (
              <div className="mt-3 p-3 bg-white rounded border">
                <h4 className="font-medium text-gray-900 mb-1">AI Explanation:</h4>
                <p className="text-sm text-gray-700">{completeSolution.explanation}</p>
              </div>
            )}
            {completeSolution.formal_definition && (
              <div className="mt-3 p-3 bg-white rounded border">
                <h4 className="font-medium text-gray-900 mb-1">Formal Definition:</h4>
                <pre className="text-sm text-gray-700 whitespace-pre-wrap">{completeSolution.formal_definition}</pre>
              </div>
            )}
          </div>
        )}

        {guidedSteps.length > 0 && (
          <div className="mb-6 p-4 rounded-lg border bg-yellow-50 border-yellow-200">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-5 h-5 text-yellow-600" />
              <span className="font-medium text-yellow-800">Guided Approach</span>
            </div>
            <div className="space-y-2">
              {guidedSteps.map((step, index) => (
                <div key={index} className="p-2 bg-white rounded border">
                  <span className="text-sm font-medium text-gray-900">Step {index + 1}:</span>
                  <p className="text-sm text-gray-700 mt-1">{step}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full">
              <TabsList className="grid w-full grid-cols-6">
                <TabsTrigger value="canvas" className="flex items-center gap-1">
                  <Play className="w-3 h-3" />
                  Canvas
                </TabsTrigger>
                <TabsTrigger value="simulate" className="flex items-center gap-1">
                  <Zap className="w-3 h-3" />
                  Simulate
                </TabsTrigger>
                <TabsTrigger value="export" className="flex items-center gap-1">
                  <Code2 className="w-3 h-3" />
                  Export
                </TabsTrigger>
                <TabsTrigger value="inspect" className="flex items-center gap-1">
                  <Search className="w-3 h-3" />
                  Inspect
                </TabsTrigger>
                <TabsTrigger value="learn" className="flex items-center gap-1">
                  <Brain className="w-3 h-3" />
                  Learn
                </TabsTrigger>
                <TabsTrigger value="assistant" className="flex items-center gap-1">
                  <Brain className="w-3 h-3" />
                  AI
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="canvas" className="mt-4">
                <Card className="h-[600px]">
                  <CardHeader>
                    <CardTitle>Automaton Builder</CardTitle>
                  </CardHeader>
                  <CardContent className="h-full">
                    <AutomataCanvas
                      automaton={automaton as any}
                      onAutomatonChange={handleAutomatonChange as any}
                    />
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="simulate" className="mt-4">
                <div className="h-[600px]">
                  <SimulationEngine
                    automaton={automaton}
                    automatonType={problem.type}
                  />
                </div>
              </TabsContent>
              
              <TabsContent value="export" className="mt-4">
                <div className="h-[600px]">
                  <CodeExporter
                    automaton={automaton}
                    automatonType={problem.type}
                  />
                </div>
              </TabsContent>
              
              <TabsContent value="inspect" className="mt-4">
                <div className="h-[600px]">
                  <AutomataInspector
                    automaton={automaton}
                    automatonType={problem.type}
                  />
                </div>
              </TabsContent>
              
              <TabsContent value="learn" className="mt-4">
                <div className="h-[600px]">
                  <LearningMode
                    automaton={automaton}
                    automatonType={problem.type}
                    onAutomatonChange={handleAutomatonChange}
                    currentProblem={problem}
                  />
                </div>
              </TabsContent>
              
              <TabsContent value="assistant" className="mt-4">
                <div className="h-[600px] grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <AIAssistantPanel
                    automatonType={problem.type}
                    currentAutomaton={automaton}
                    onAutomatonGenerated={handleAutomatonChange}
                  />
                  <ProofAssistant
                    automaton={automaton}
                  />
                </div>
              </TabsContent>
            </Tabs>
          </div>
          
          <div className="lg:col-span-1 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Test Cases</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {problem.test_strings.map((testCase, index) => (
                    <div
                      key={index}
                      className={`p-2 rounded border text-sm ${
                        validationResult?.test_results[index]?.correct === true
                          ? 'bg-green-50 border-green-200'
                          : validationResult?.test_results[index]?.correct === false
                          ? 'bg-red-50 border-red-200'
                          : 'bg-gray-50 border-gray-200'
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <span className="font-mono">
                          "{testCase.string || 'ε'}"
                        </span>
                        <Badge variant={testCase.should_accept ? "default" : "secondary"}>
                          {testCase.should_accept ? 'Accept' : 'Reject'}
                        </Badge>
                      </div>
                      {validationResult?.test_results[index] && (
                        <div className="mt-1 text-xs text-gray-600">
                          Result: {validationResult.test_results[index].actual ? 'Accept' : 'Reject'}
                          {validationResult.test_results[index].correct ? ' ✓' : ' ✗'}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <ProjectManager
              currentAutomaton={automaton}
              currentProblem={problem}
              onLoadProject={(project) => {
                setAutomaton(project.automaton);
                setValidationResult(null);
              }}
              onSaveProject={(name) => {
                console.log(`Project "${name}" saved successfully`);
              }}
            />

            <ExampleGallery
              onLoadExample={handleAutomatonChange}
            />
            
            {problem.hints && problem.hints.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Hints</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 max-h-32 overflow-y-auto">
                    {problem.hints.map((hint, index) => (
                      <div key={index} className="p-2 bg-yellow-50 rounded border border-yellow-200">
                        <p className="text-sm text-yellow-800">{hint}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export { ComprehensiveProblemView };
