import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Loader2, Brain, Lightbulb, MessageCircle, Code, BookOpen, Target } from 'lucide-react';
import { apiService } from '../services/api';
import { Problem, Automaton, ValidationResult, Solution } from '../types/automata';

interface AITutorProps {
  problem: Problem;
  automaton: Automaton;
  validationResult?: ValidationResult;
  onGetHint: () => void;
}

export const AITutor: React.FC<AITutorProps> = ({
  problem,
  automaton,
  validationResult,
  onGetHint,
}) => {
  const [aiStatus, setAiStatus] = useState<{ available: boolean; models?: string[] }>({ available: false });
  const [isLoadingHint, setIsLoadingHint] = useState(false);
  const [currentHint, setCurrentHint] = useState<string>('');
  const [stepByStepGuidance, setStepByStepGuidance] = useState<string[]>([]);
  const [generatedSolution, setGeneratedSolution] = useState<any>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [solutionExplanation, setSolutionExplanation] = useState<any>(null);
  const [isExplaining, setIsExplaining] = useState(false);
  const [guidedMode, setGuidedMode] = useState(false);
  const [currentGuidedStep, setCurrentGuidedStep] = useState<string>('');
  const [isGettingGuidedStep, setIsGettingGuidedStep] = useState(false);

  useEffect(() => {
    checkAIStatus();
  }, []);

  const checkAIStatus = async () => {
    try {
      const status = await apiService.checkAIStatus();
      setAiStatus(status);
    } catch (error) {
      console.error('Failed to check AI status:', error);
    }
  };

  const getAIGuidance = async () => {
    if (!aiStatus.available) return;

    setIsLoadingHint(true);
    try {
      const response = await apiService.getAIHint(problem.id, {
        problem_description: problem.description,
        user_automaton: automaton,
        test_results: validationResult?.test_results || [],
        mistakes: validationResult?.mistakes || [],
        automata_type: problem.type,
      });
      setCurrentHint(response.ai_hint);
      
      generateStepByStepGuidance();
    } catch (error) {
      console.error('Failed to get AI guidance:', error);
      setCurrentHint('AI guidance is temporarily unavailable. Please try the built-in hints.');
    } finally {
      setIsLoadingHint(false);
    }
  };

  const generateReferenceSolution = async () => {
    if (!aiStatus.available) return;

    setIsGenerating(true);
    try {
      const response = await apiService.generateSolution(problem.id);
      setGeneratedSolution(response);
    } catch (error) {
      console.error('Failed to generate solution:', error);
      setGeneratedSolution({
        note: 'Solution generation temporarily unavailable. Try building the automaton step by step.',
        generated_automaton: { formal_definition: 'Not available' }
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const explainCurrentSolution = async () => {
    if (!aiStatus.available || automaton.states.length === 0) return;

    setIsExplaining(true);
    try {
      const solution: Solution = {
        problem_id: problem.id,
        automaton: automaton,
        user_id: 'anonymous'
      };
      const response = await apiService.explainSolution(problem.id, solution);
      setSolutionExplanation(response);
    } catch (error) {
      console.error('Failed to explain solution:', error);
      setSolutionExplanation({
        explanation: 'Solution explanation temporarily unavailable. Review your automaton against the test cases.',
        key_concepts: ['states', 'transitions', 'acceptance'],
        next_steps: ['Test with examples', 'Check state transitions']
      });
    } finally {
      setIsExplaining(false);
    }
  };

  const getGuidedStep = async () => {
    if (!aiStatus.available) return;

    setIsGettingGuidedStep(true);
    try {
      const response = await apiService.getGuidedStep(problem.id, {
        problem_description: problem.description,
        user_automaton: automaton,
        test_results: validationResult?.test_results || [],
        mistakes: validationResult?.mistakes || [],
        automata_type: problem.type,
      });
      setCurrentGuidedStep(response.guided_step);
    } catch (error) {
      console.error('Failed to get guided step:', error);
      setCurrentGuidedStep('Try adding your first state by clicking on the canvas.');
    } finally {
      setIsGettingGuidedStep(false);
    }
  };

  const generateStepByStepGuidance = () => {
    const guidance: string[] = [];
    
    if (automaton.states.length === 0) {
      guidance.push("Start by creating your first state - click anywhere on the canvas");
    } else {
      const startStates = automaton.states.filter(s => s.is_start);
      if (startStates.length === 0) {
        guidance.push("Select a state and mark it as the start state using the 'Start State' button");
      } else if (startStates.length > 1) {
        guidance.push("A DFA should have exactly one start state - you currently have multiple");
      }
      
      const acceptStates = automaton.states.filter(s => s.is_accept);
      if (acceptStates.length === 0) {
        guidance.push("Consider which states should be accepting states based on the problem requirements");
      }
      
      if (automaton.transitions.length === 0) {
        guidance.push("Add transitions between states - select a state and click 'Add Transition'");
      } else {
        const missingTransitions: string[] = [];
        automaton.states.forEach(state => {
          problem.alphabet.forEach(symbol => {
            const hasTransition = automaton.transitions.some(
              t => t.from_state === state.id && t.symbol === symbol
            );
            if (!hasTransition) {
              missingTransitions.push(`State ${state.id} needs a transition for symbol '${symbol}'`);
            }
          });
        });
        
        if (missingTransitions.length > 0) {
          guidance.push("Complete the transition function - every state needs transitions for all alphabet symbols");
        }
      }
    }
    
    setStepByStepGuidance(guidance);
  };

  useEffect(() => {
    generateStepByStepGuidance();
  }, [automaton, problem]);

  const getProgressAnalysis = () => {
    const totalStates = automaton.states.length;
    const totalTransitions = automaton.transitions.length;
    const startStates = automaton.states.filter(s => s.is_start).length;
    const acceptStates = automaton.states.filter(s => s.is_accept).length;
    
    const expectedTransitions = totalStates * problem.alphabet.length;
    const completeness = totalTransitions / Math.max(expectedTransitions, 1);
    
    return {
      totalStates,
      totalTransitions,
      startStates,
      acceptStates,
      completeness: Math.min(completeness, 1),
      isStructurallyComplete: startStates === 1 && acceptStates > 0 && completeness >= 1,
    };
  };

  const progress = getProgressAnalysis();

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-600" />
          AI Tutor
          {aiStatus.available ? (
            <Badge variant="secondary" className="bg-green-100 text-green-800">
              AI Online
            </Badge>
          ) : (
            <Badge variant="secondary" className="bg-red-100 text-red-800">
              AI Offline
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <h4 className="font-semibold text-sm">Progress Analysis</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>States: {progress.totalStates}</div>
            <div>Transitions: {progress.totalTransitions}</div>
            <div>Start States: {progress.startStates}</div>
            <div>Accept States: {progress.acceptStates}</div>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress.completeness * 100}%` }}
            />
          </div>
          <p className="text-xs text-gray-600">
            Transition completeness: {Math.round(progress.completeness * 100)}%
          </p>
        </div>

        {stepByStepGuidance.length > 0 && (
          <div className="space-y-2">
            <h4 className="font-semibold text-sm flex items-center gap-1">
              <Lightbulb className="w-4 h-4 text-yellow-500" />
              Next Steps
            </h4>
            <ul className="space-y-1">
              {stepByStepGuidance.map((step, index) => (
                <li key={index} className="text-sm text-gray-700 flex items-start gap-2">
                  <span className="text-blue-500 font-bold">{index + 1}.</span>
                  {step}
                </li>
              ))}
            </ul>
          </div>
        )}

        {validationResult?.ai_explanation && (
          <div className="space-y-2">
            <h4 className="font-semibold text-sm flex items-center gap-1">
              <MessageCircle className="w-4 h-4 text-blue-500" />
              AI Feedback
            </h4>
            <div className="bg-blue-50 p-3 rounded-lg text-sm">
              {validationResult.ai_explanation}
            </div>
          </div>
        )}

        {validationResult?.ai_hints && validationResult.ai_hints.length > 0 && (
          <div className="space-y-2">
            <h4 className="font-semibold text-sm">AI Hints</h4>
            <ul className="space-y-1">
              {validationResult.ai_hints.map((hint, index) => (
                <li key={index} className="text-sm text-gray-700 bg-yellow-50 p-2 rounded">
                  💡 {hint}
                </li>
              ))}
            </ul>
          </div>
        )}

        {currentHint && (
          <div className="space-y-2">
            <h4 className="font-semibold text-sm">Personalized Guidance</h4>
            <div className="bg-purple-50 p-3 rounded-lg text-sm">
              🤖 {currentHint}
            </div>
          </div>
        )}


        {generatedSolution && (
          <div className="space-y-2">
            <h4 className="font-semibold text-sm flex items-center gap-1">
              <Code className="w-4 h-4 text-green-500" />
              Reference Solution
            </h4>
            <div className="bg-green-50 p-3 rounded-lg text-sm space-y-2">
              <p className="font-medium text-green-800">{generatedSolution.note}</p>
              {generatedSolution.generated_automaton?.formal_definition && (
                <div>
                  <p className="font-medium">Formal Definition:</p>
                  <pre className="text-xs bg-white p-2 rounded border overflow-x-auto">
                    {typeof generatedSolution.generated_automaton.formal_definition === 'string' 
                      ? generatedSolution.generated_automaton.formal_definition 
                      : JSON.stringify(generatedSolution.generated_automaton.formal_definition, null, 2)}
                  </pre>
                </div>
              )}
              {generatedSolution.explanation?.explanation && (
                <div>
                  <p className="font-medium">AI Explanation:</p>
                  <p className="text-gray-700">{generatedSolution.explanation.explanation.substring(0, 200)}...</p>
                </div>
              )}
            </div>
          </div>
        )}

        {solutionExplanation && (
          <div className="space-y-2">
            <h4 className="font-semibold text-sm flex items-center gap-1">
              <BookOpen className="w-4 h-4 text-blue-500" />
              Your Solution Analysis
            </h4>
            <div className="bg-blue-50 p-3 rounded-lg text-sm space-y-2">
              <p className="text-gray-700">{solutionExplanation.explanation?.substring(0, 300)}...</p>
              {solutionExplanation.key_concepts && (
                <div>
                  <p className="font-medium">Key Concepts:</p>
                  <div className="flex flex-wrap gap-1">
                    {solutionExplanation.key_concepts.slice(0, 3).map((concept: string, index: number) => (
                      <Badge key={index} variant="secondary" className="text-xs">
                        {concept}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          <Button
            onClick={() => setGuidedMode(!guidedMode)}
            variant={guidedMode ? "default" : "outline"}
            size="sm"
            className={guidedMode ? "bg-gradient-to-r from-green-500 to-blue-500 text-white" : "border-green-500 text-green-600 hover:bg-green-50"}
          >
            <Target className="w-4 h-4 mr-1" />
            {guidedMode ? "Exit Guided Mode" : "Guided Mode"}
          </Button>
          
          {guidedMode && (
            <Button
              onClick={getGuidedStep}
              disabled={!aiStatus.available || isGettingGuidedStep}
              size="sm"
              variant="outline"
              className="border-green-500 text-green-600 hover:bg-green-50"
            >
              {isGettingGuidedStep ? (
                <Loader2 className="w-4 h-4 animate-spin mr-1" />
              ) : (
                <Lightbulb className="w-4 h-4 mr-1" />
              )}
              Next Step
            </Button>
          )}
          
          <Button
            onClick={getAIGuidance}
            disabled={!aiStatus.available || isLoadingHint}
            size="sm"
            className="bg-gradient-to-r from-purple-500 to-blue-500 text-white"
          >
            {isLoadingHint ? (
              <Loader2 className="w-4 h-4 animate-spin mr-1" />
            ) : (
              <Brain className="w-4 h-4 mr-1" />
            )}
            Get AI Guidance
          </Button>
          
          <Button
            onClick={generateReferenceSolution}
            disabled={!aiStatus.available || isGenerating}
            size="sm"
            variant="outline"
            className="border-green-500 text-green-600 hover:bg-green-50"
          >
            {isGenerating ? (
              <Loader2 className="w-4 h-4 animate-spin mr-1" />
            ) : (
              <Code className="w-4 h-4 mr-1" />
            )}
            Generate Solution
          </Button>

          <Button
            onClick={explainCurrentSolution}
            disabled={!aiStatus.available || isExplaining || automaton.states.length === 0}
            size="sm"
            variant="outline"
            className="border-blue-500 text-blue-600 hover:bg-blue-50"
          >
            {isExplaining ? (
              <Loader2 className="w-4 h-4 animate-spin mr-1" />
            ) : (
              <BookOpen className="w-4 h-4 mr-1" />
            )}
            Explain My Solution
          </Button>
          
          <Button onClick={onGetHint} variant="outline" size="sm">
            <Lightbulb className="w-4 h-4 mr-1" />
            Built-in Hint
          </Button>
        </div>

        {guidedMode && currentGuidedStep && (
          <div className="space-y-2">
            <h4 className="font-semibold text-sm flex items-center gap-1">
              <Lightbulb className="w-4 h-4 text-green-500" />
              Guided Step
            </h4>
            <div className="bg-green-50 p-3 rounded-lg text-sm border-l-4 border-green-500">
              🎯 {currentGuidedStep}
            </div>
          </div>
        )}

        {!aiStatus.available && (
          <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
            💡 To enable multi-model AI tutoring, install: ollama pull codellama:34b && ollama pull deepseek-coder:33b
          </div>
        )}
      </CardContent>
    </Card>
  );
};
