import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Play, Pause, SkipForward, SkipBack, RotateCcw, Zap } from 'lucide-react';
import { ExtendedAutomaton, AutomataType, SimulationResult, SimulationStep } from '../types/automata';

interface SimulationEngineProps {
  automaton: ExtendedAutomaton;
  automatonType: AutomataType;
  onStateHighlight?: (stateId: string) => void;
  onTransitionHighlight?: (transitionIndex: number) => void;
}

export const SimulationEngine: React.FC<SimulationEngineProps> = ({
  automaton,
  automatonType,
  onStateHighlight,
  onTransitionHighlight
}) => {
  const highlightTransition = onTransitionHighlight || (() => {});
  const [inputString, setInputString] = useState('');
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000);
  const [stackContents, setStackContents] = useState<string[]>([]);
  const [tapeContents, setTapeContents] = useState<string[]>([]);
  const [headPosition, setHeadPosition] = useState(0);

  useEffect(() => {
    let interval: any;
    if (isPlaying && simulationResult && currentStep < simulationResult.steps.length) {
      interval = setInterval(() => {
        setCurrentStep(prev => {
          const nextStep = prev + 1;
          if (nextStep >= simulationResult.steps.length) {
            setIsPlaying(false);
            return prev;
          }
          return nextStep;
        });
      }, playbackSpeed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, simulationResult, currentStep, playbackSpeed]);

  useEffect(() => {
    if (simulationResult && simulationResult.steps[currentStep]) {
      const step = simulationResult.steps[currentStep];
      onStateHighlight?.(step.current_state);
      highlightTransition(currentStep);
      
      if (step.stack_contents) {
        setStackContents(step.stack_contents);
      }
      if (step.tape_contents) {
        setTapeContents(step.tape_contents);
        setHeadPosition(step.head_position || 0);
      }
    }
  }, [currentStep, simulationResult, onStateHighlight]);

  const handleSimulate = async () => {
    if (!inputString.trim()) return;

    try {
      const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          automaton,
          input_string: inputString,
          type: automatonType
        })
      });

      if (response.ok) {
        const result: SimulationResult = await response.json();
        setSimulationResult(result);
        setCurrentStep(0);
        setIsPlaying(false);
      }
    } catch (error) {
      console.error('Simulation error:', error);
    }
  };

  const handlePlay = () => {
    if (!simulationResult) return;
    setIsPlaying(!isPlaying);
  };

  const handleStepForward = () => {
    if (!simulationResult) return;
    setCurrentStep(prev => Math.min(prev + 1, simulationResult.steps.length - 1));
  };

  const handleStepBack = () => {
    setCurrentStep(prev => Math.max(prev - 1, 0));
  };

  const handleReset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
    setStackContents([]);
    setTapeContents([]);
    setHeadPosition(0);
  };

  const getCurrentStep = (): SimulationStep | null => {
    return simulationResult?.steps[currentStep] || null;
  };

  const renderInputString = () => {
    const step = getCurrentStep();
    if (!step) return inputString;

    return (
      <div className="font-mono text-lg flex">
        {inputString.split('').map((char, index) => (
          <span
            key={index}
            className={`px-1 ${
              index < step.input_position
                ? 'bg-green-200 text-green-800'
                : index === step.input_position
                ? 'bg-blue-200 text-blue-800 border-b-2 border-blue-600'
                : 'text-gray-600'
            }`}
          >
            {char}
          </span>
        ))}
        {step.input_position >= inputString.length && (
          <span className="px-1 bg-blue-200 text-blue-800 border-b-2 border-blue-600">$</span>
        )}
      </div>
    );
  };

  const renderStackVisualization = () => {
    if (automatonType !== 'pda' || !stackContents.length) return null;

    return (
      <Card className="mt-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Stack Contents</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col-reverse gap-1 max-h-32 overflow-y-auto">
            {stackContents.map((symbol, index) => (
              <div
                key={index}
                className="bg-purple-100 text-purple-800 px-2 py-1 rounded text-center font-mono text-sm border"
              >
                {symbol}
              </div>
            ))}
            {stackContents.length === 0 && (
              <div className="text-gray-500 text-center text-sm italic">Empty Stack</div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderTapeVisualization = () => {
    if (automatonType !== 'tm' || !tapeContents.length) return null;

    return (
      <Card className="mt-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Tape Contents</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-1 overflow-x-auto pb-2">
            {tapeContents.map((symbol, index) => (
              <div
                key={index}
                className={`min-w-[2rem] h-8 border-2 flex items-center justify-center font-mono text-sm ${
                  index === headPosition
                    ? 'border-red-500 bg-red-100 text-red-800'
                    : 'border-gray-300 bg-white'
                }`}
              >
                {symbol === ' ' ? '□' : symbol}
              </div>
            ))}
          </div>
          <div className="flex gap-1 mt-1">
            {tapeContents.map((_, index) => (
              <div
                key={index}
                className={`min-w-[2rem] h-4 flex items-center justify-center text-xs ${
                  index === headPosition ? 'text-red-600' : 'text-transparent'
                }`}
              >
                ↑
              </div>
            ))}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Head Position: {headPosition}
          </div>
        </CardContent>
      </Card>
    );
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

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-green-600" />
          Simulation Engine
          <Badge variant="outline" className="ml-auto">
            {getTypeDisplayName(automatonType)}
          </Badge>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Input
            value={inputString}
            onChange={(e) => setInputString(e.target.value)}
            placeholder="Enter input string to simulate..."
            className="flex-1"
          />
          <Button onClick={handleSimulate} disabled={!inputString.trim()}>
            <Play className="w-4 h-4 mr-1" />
            Simulate
          </Button>
        </div>

        {simulationResult && (
          <>
            <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50">
              <div className="flex items-center gap-2">
                <Badge variant={simulationResult.accepted ? "default" : "destructive"}>
                  {simulationResult.accepted ? "ACCEPTED" : "REJECTED"}
                </Badge>
                <span className="text-sm text-gray-600">
                  Step {currentStep + 1} of {simulationResult.steps.length}
                </span>
              </div>
              
              <div className="flex items-center gap-1">
                <Button size="sm" variant="outline" onClick={handleReset}>
                  <RotateCcw className="w-3 h-3" />
                </Button>
                <Button size="sm" variant="outline" onClick={handleStepBack} disabled={currentStep === 0}>
                  <SkipBack className="w-3 h-3" />
                </Button>
                <Button size="sm" variant="outline" onClick={handlePlay}>
                  {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                </Button>
                <Button 
                  size="sm" 
                  variant="outline" 
                  onClick={handleStepForward} 
                  disabled={currentStep >= simulationResult.steps.length - 1}
                >
                  <SkipForward className="w-3 h-3" />
                </Button>
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium text-gray-700 block mb-1">Input String:</label>
                {renderInputString()}
              </div>

              {getCurrentStep() && (
                <div>
                  <label className="text-sm font-medium text-gray-700 block mb-1">Current State:</label>
                  <Badge variant="outline" className="font-mono">
                    {getCurrentStep()!.current_state}
                  </Badge>
                </div>
              )}

              {getCurrentStep() && (
                <div>
                  <label className="text-sm font-medium text-gray-700 block mb-1">Action:</label>
                  <p className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
                    {getCurrentStep()!.action_description}
                  </p>
                </div>
              )}

              <div className="flex gap-2 items-center">
                <label className="text-sm font-medium text-gray-700">Speed:</label>
                <input
                  type="range"
                  min="100"
                  max="2000"
                  step="100"
                  value={playbackSpeed}
                  onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                  className="flex-1"
                />
                <span className="text-xs text-gray-500">{playbackSpeed}ms</span>
              </div>
            </div>

            {renderStackVisualization()}
            {renderTapeVisualization()}

            {simulationResult.error_message && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-800">
                  <strong>Error:</strong> {simulationResult.error_message}
                </p>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
};
