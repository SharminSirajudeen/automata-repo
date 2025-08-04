import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { CheckCircle, XCircle, Play, RotateCcw, Lightbulb } from 'lucide-react';
import { Problem, Automaton, ValidationResult } from '../types/automata';
import { AutomataCanvas } from './AutomataCanvas';
import { AITutor } from './AITutor';
import { apiService } from '../services/api';

interface ProblemViewProps {
  problem: Problem;
  onBack: () => void;
}

export const ProblemView: React.FC<ProblemViewProps> = ({ problem, onBack }) => {
  const [automaton, setAutomaton] = useState<Automaton>({
    states: [],
    transitions: [],
    alphabet: problem.alphabet,
  });
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [currentHintIndex, setCurrentHintIndex] = useState(0);
  const [currentHint, setCurrentHint] = useState<string>('');
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationString, setSimulationString] = useState('');
  const [simulationPath, setSimulationPath] = useState<string[]>([]);
  const [currentSimulationStep, setCurrentSimulationStep] = useState(0);

  const validateSolution = async () => {
    setIsValidating(true);
    try {
      const result = await apiService.validateSolution(problem.id, automaton);
      setValidationResult(result);
    } catch (error) {
      console.error('Validation failed:', error);
    } finally {
      setIsValidating(false);
    }
  };

  const getHint = async () => {
    try {
      const response = await apiService.getHint(problem.id, currentHintIndex);
      setCurrentHint(response.hint);
      setCurrentHintIndex((prev) => (prev + 1) % (problem.hints?.length || 1));
    } catch (error) {
      console.error('Failed to get hint:', error);
    }
  };

  const simulateString = (testString: string) => {
    const startState = automaton.states.find(s => s.is_start);
    if (!startState) return;

    const path = [startState.id];
    let currentState = startState.id;

    for (const symbol of testString) {
      const transition = automaton.transitions.find(
        t => t.from_state === currentState && t.symbol === symbol
      );
      if (!transition) break;
      currentState = transition.to_state;
      path.push(currentState);
    }

    setSimulationPath(path);
    setSimulationString(testString);
    setCurrentSimulationStep(0);
    setIsSimulating(true);
  };

  const stepSimulation = () => {
    if (currentSimulationStep < simulationPath.length - 1) {
      setCurrentSimulationStep(prev => prev + 1);
    }
  };

  const resetSimulation = () => {
    setIsSimulating(false);
    setSimulationPath([]);
    setCurrentSimulationStep(0);
    setSimulationString('');
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Button onClick={onBack} variant="outline" className="mb-4">
            ← Back to Problems
          </Button>
          <h1 className="text-3xl font-bold">{problem.title}</h1>
          <p className="text-gray-600 mt-2">{problem.description}</p>
        </div>
        <Badge variant="secondary" className="text-lg px-4 py-2">
          {problem.type.toUpperCase()}
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Automaton Builder</CardTitle>
            </CardHeader>
            <CardContent>
              <AutomataCanvas
                automaton={automaton}
                onAutomatonChange={setAutomaton}
                onRequestAIGuidance={getHint}
                isSimulating={isSimulating}
                simulationPath={simulationPath}
                currentSimulationStep={currentSimulationStep}
                showInteractiveOverlay={false}
                stepExplanations={{}}
                onStateHover={(stateId) => console.log('State hovered:', stateId)}
                onTransitionHover={(index) => console.log('Transition hovered:', index)}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Validation & Testing</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Button
                  onClick={validateSolution}
                  disabled={isValidating || automaton.states.length === 0}
                  className="bg-green-600 hover:bg-green-700"
                >
                  {isValidating ? 'Validating...' : 'Validate Solution'}
                </Button>
                
                {isSimulating ? (
                  <div className="flex gap-2">
                    <Button onClick={stepSimulation} size="sm">
                      <Play className="w-4 h-4 mr-1" />
                      Step ({currentSimulationStep + 1}/{simulationPath.length})
                    </Button>
                    <Button onClick={resetSimulation} variant="outline" size="sm">
                      <RotateCcw className="w-4 h-4 mr-1" />
                      Reset
                    </Button>
                  </div>
                ) : null}
              </div>

              {isSimulating && (
                <div className="bg-blue-50 p-3 rounded-lg">
                  <p className="text-sm font-medium">
                    Simulating: "{simulationString}"
                  </p>
                  <p className="text-sm text-gray-600">
                    Current state: {simulationPath[currentSimulationStep]}
                  </p>
                  <p className="text-sm text-gray-600">
                    Step {currentSimulationStep + 1} of {simulationPath.length}
                  </p>
                </div>
              )}

              {validationResult && (
                <div className="space-y-4">
                  <div className={`p-4 rounded-lg ${
                    validationResult.is_correct ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      {validationResult.is_correct ? (
                        <CheckCircle className="w-5 h-5 text-green-600" />
                      ) : (
                        <XCircle className="w-5 h-5 text-red-600" />
                      )}
                      <span className="font-semibold">
                        Score: {Math.round(validationResult.score * 100)}%
                      </span>
                    </div>
                    <p className="text-sm">{validationResult.feedback}</p>
                  </div>

                  <Tabs defaultValue="test-results" className="w-full">
                    <TabsList>
                      <TabsTrigger value="test-results">Test Results</TabsTrigger>
                      <TabsTrigger value="mistakes">Issues</TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="test-results" className="space-y-2">
                      {validationResult.test_results.map((result, index) => (
                        <div
                          key={index}
                          className={`p-3 rounded border cursor-pointer hover:bg-gray-50 ${
                            result.correct ? 'border-green-200' : 'border-red-200'
                          }`}
                          onClick={() => simulateString(result.string)}
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-mono">"{result.string}"</span>
                            <div className="flex items-center gap-2">
                              {result.correct ? (
                                <CheckCircle className="w-4 h-4 text-green-600" />
                              ) : (
                                <XCircle className="w-4 h-4 text-red-600" />
                              )}
                              <Button size="sm" variant="outline">
                                <Play className="w-3 h-3 mr-1" />
                                Simulate
                              </Button>
                            </div>
                          </div>
                          <div className="text-xs text-gray-600 mt-1">
                            Expected: {result.expected ? 'Accept' : 'Reject'} | 
                            Got: {result.actual ? 'Accept' : 'Reject'}
                          </div>
                        </div>
                      ))}
                    </TabsContent>
                    
                    <TabsContent value="mistakes">
                      {validationResult.mistakes.length > 0 ? (
                        <ul className="space-y-2">
                          {validationResult.mistakes.map((mistake, index) => (
                            <li key={index} className="text-sm text-red-700 bg-red-50 p-2 rounded">
                              • {mistake}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-sm text-gray-600">No structural issues found.</p>
                      )}
                    </TabsContent>
                  </Tabs>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <AITutor
            problem={problem}
            automaton={automaton}
            validationResult={validationResult || undefined}
            onGetHint={getHint}
          />

          <Card>
            <CardHeader>
              <CardTitle>Problem Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-semibold text-sm mb-2">Language</h4>
                <p className="text-sm text-gray-700">{problem.language_description}</p>
              </div>
              
              <div>
                <h4 className="font-semibold text-sm mb-2">Alphabet</h4>
                <div className="flex gap-1">
                  {problem.alphabet.map(symbol => (
                    <Badge key={symbol} variant="outline">{symbol}</Badge>
                  ))}
                </div>
              </div>

              {currentHint && (
                <div>
                  <h4 className="font-semibold text-sm mb-2 flex items-center gap-1">
                    <Lightbulb className="w-4 h-4 text-yellow-500" />
                    Hint
                  </h4>
                  <p className="text-sm text-gray-700 bg-yellow-50 p-3 rounded">{currentHint}</p>
                </div>
              )}

              <div>
                <h4 className="font-semibold text-sm mb-2">Test Cases</h4>
                <div className="space-y-1 max-h-40 overflow-y-auto">
                  {problem.test_strings.slice(0, 5).map((test, index) => (
                    <div key={index} className="text-xs flex justify-between items-center p-2 bg-gray-50 rounded">
                      <span className="font-mono">"{test.string}"</span>
                      <Badge variant={test.should_accept ? "default" : "secondary"} className="text-xs">
                        {test.should_accept ? 'Accept' : 'Reject'}
                      </Badge>
                    </div>
                  ))}
                  {problem.test_strings.length > 5 && (
                    <p className="text-xs text-gray-500 text-center">
                      +{problem.test_strings.length - 5} more test cases
                    </p>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
