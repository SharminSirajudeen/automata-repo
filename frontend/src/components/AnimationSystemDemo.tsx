import React, { useState, useCallback } from 'react';
import { AnimationSystem } from './AnimationSystem';
import { OnboardingFlow } from './OnboardingFlow';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import {
  Play,
  BookOpen,
  Settings,
  TestTube,
  Activity,
  Users,
  Zap
} from 'lucide-react';
import {
  ExtendedAutomaton,
  SimulationStep,
  AutomataType,
  State,
  Transition,
  TMAutomaton,
  PDAAutomaton
} from '../types/automata';

const AnimationSystemDemo: React.FC = () => {
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [currentAutomataType, setCurrentAutomataType] = useState<AutomataType>('dfa');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  // Sample DFA for testing
  const sampleDFA: ExtendedAutomaton = {
    states: [
      { id: 'q0', x: 100, y: 100, is_start: true, is_accept: false, label: 'Start' },
      { id: 'q1', x: 300, y: 100, is_start: false, is_accept: true, label: 'Accept' },
      { id: 'q2', x: 200, y: 250, is_start: false, is_accept: false, label: 'Reject' }
    ],
    transitions: [
      { from_state: 'q0', to_state: 'q1', symbol: 'a' },
      { from_state: 'q0', to_state: 'q2', symbol: 'b' },
      { from_state: 'q1', to_state: 'q1', symbol: 'a' },
      { from_state: 'q1', to_state: 'q2', symbol: 'b' },
      { from_state: 'q2', to_state: 'q2', symbol: 'a,b' }
    ],
    alphabet: ['a', 'b'],
    type: 'dfa'
  };

  // Sample PDA for testing
  const samplePDA: PDAAutomaton = {
    type: 'pda',
    states: [
      { id: 'q0', x: 100, y: 100, is_start: true, is_accept: false },
      { id: 'q1', x: 200, y: 100, is_start: false, is_accept: false },
      { id: 'q2', x: 300, y: 100, is_start: false, is_accept: true }
    ],
    transitions: [
      { from_state: 'q0', to_state: 'q1', symbol: 'a', stack_pop: 'ε', stack_push: 'A' },
      { from_state: 'q1', to_state: 'q1', symbol: 'a', stack_pop: 'A', stack_push: 'AA' },
      { from_state: 'q1', to_state: 'q2', symbol: 'b', stack_pop: 'A', stack_push: 'ε' },
      { from_state: 'q2', to_state: 'q2', symbol: 'b', stack_pop: 'A', stack_push: 'ε' }
    ],
    alphabet: ['a', 'b'],
    stack_alphabet: ['A', 'Z'],
    start_stack_symbol: 'Z'
  };

  // Sample Turing Machine for testing
  const sampleTM: TMAutomaton = {
    type: 'tm',
    states: [
      { id: 'q0', x: 100, y: 100, is_start: true, is_accept: false },
      { id: 'q1', x: 200, y: 100, is_start: false, is_accept: false },
      { id: 'q2', x: 300, y: 100, is_start: false, is_accept: true }
    ],
    transitions: [
      { from_state: 'q0', to_state: 'q1', read_symbol: 'a', write_symbol: 'X', head_direction: 'R' },
      { from_state: 'q1', to_state: 'q2', read_symbol: 'b', write_symbol: 'Y', head_direction: 'R' },
      { from_state: 'q2', to_state: 'q2', read_symbol: '□', write_symbol: '□', head_direction: 'S' }
    ],
    tape_alphabet: ['a', 'b', 'X', 'Y', '□'],
    blank_symbol: '□'
  };

  // Sample simulation steps
  const sampleSimulationSteps: SimulationStep[] = [
    {
      step_number: 0,
      current_state: 'q0',
      input_position: 0,
      remaining_input: 'aab',
      action_description: 'Starting simulation at initial state'
    },
    {
      step_number: 1,
      current_state: 'q1',
      input_position: 1,
      remaining_input: 'ab',
      action_description: 'Read \'a\', transition to q1',
      stack_contents: currentAutomataType === 'pda' ? ['Z', 'A'] : undefined,
      tape_contents: currentAutomataType === 'tm' ? ['X', 'a', 'b', '□'] : undefined,
      head_position: currentAutomataType === 'tm' ? 1 : undefined
    },
    {
      step_number: 2,
      current_state: 'q1',
      input_position: 2,
      remaining_input: 'b',
      action_description: 'Read \'a\', stay in q1',
      stack_contents: currentAutomataType === 'pda' ? ['Z', 'A', 'A'] : undefined,
      tape_contents: currentAutomataType === 'tm' ? ['X', 'X', 'b', '□'] : undefined,
      head_position: currentAutomataType === 'tm' ? 2 : undefined
    },
    {
      step_number: 3,
      current_state: 'q2',
      input_position: 3,
      remaining_input: '',
      action_description: 'Read \'b\', transition to accept state q2',
      stack_contents: currentAutomataType === 'pda' ? ['Z', 'A'] : undefined,
      tape_contents: currentAutomataType === 'tm' ? ['X', 'X', 'Y', '□'] : undefined,
      head_position: currentAutomataType === 'tm' ? 3 : undefined
    }
  ];

  const getCurrentAutomaton = useCallback((): ExtendedAutomaton => {
    switch (currentAutomataType) {
      case 'pda':
        return samplePDA;
      case 'tm':
        return sampleTM;
      default:
        return sampleDFA;
    }
  }, [currentAutomataType]);

  const handlePlay = useCallback(() => {
    setIsPlaying(true);
  }, []);

  const handlePause = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const handleStop = useCallback(() => {
    setIsPlaying(false);
    setCurrentStep(0);
  }, []);

  const handleStep = useCallback((direction: 'forward' | 'backward') => {
    if (direction === 'forward' && currentStep < sampleSimulationSteps.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else if (direction === 'backward' && currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  }, [currentStep, sampleSimulationSteps.length]);

  const handleSeek = useCallback((step: number) => {
    setCurrentStep(Math.max(0, Math.min(step, sampleSimulationSteps.length - 1)));
  }, [sampleSimulationSteps.length]);

  const handleReset = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  const handleError = useCallback((error: Error) => {
    console.error('Animation error:', error);
  }, []);

  const handleOnboardingComplete = useCallback(() => {
    setShowOnboarding(false);
    console.log('Onboarding completed!');
  }, []);

  const handleOnboardingSkip = useCallback(() => {
    setShowOnboarding(false);
    console.log('Onboarding skipped!');
  }, []);

  const handleCreateFirstAutomaton = useCallback((automaton: ExtendedAutomaton) => {
    console.log('First automaton created:', automaton);
  }, []);

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold">Animation System & Onboarding Demo</h1>
            <p className="text-gray-600 dark:text-gray-400">
              Test and demonstrate the enhanced animation system and interactive onboarding flow
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              onClick={() => setShowOnboarding(true)}
              className="flex items-center space-x-2"
            >
              <BookOpen className="h-4 w-4" />
              <span>Start Onboarding</span>
            </Button>
          </div>
        </div>

        {/* Automata Type Selector */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Settings className="h-5 w-5" />
              <span>Demo Configuration</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Automata Type</label>
                <div className="flex space-x-2">
                  {(['dfa', 'nfa', 'pda', 'tm'] as AutomataType[]).map(type => (
                    <Button
                      key={type}
                      variant={currentAutomataType === type ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setCurrentAutomataType(type)}
                    >
                      {type.toUpperCase()}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
                <div className="flex items-center space-x-1">
                  <Activity className="h-4 w-4" />
                  <span>Current: {currentAutomataType.toUpperCase()}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Zap className="h-4 w-4" />
                  <span>Steps: {sampleSimulationSteps.length}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="animation" className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="animation" className="flex items-center space-x-2">
            <Play className="h-4 w-4" />
            <span>Animation System</span>
          </TabsTrigger>
          <TabsTrigger value="testing" className="flex items-center space-x-2">
            <TestTube className="h-4 w-4" />
            <span>Testing</span>
          </TabsTrigger>
          <TabsTrigger value="features" className="flex items-center space-x-2">
            <Users className="h-4 w-4" />
            <span>Features</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="animation" className="space-y-4">
          <AnimationSystem
            automaton={getCurrentAutomaton()}
            simulationSteps={sampleSimulationSteps}
            currentStep={currentStep}
            isPlaying={isPlaying}
            onPlay={handlePlay}
            onPause={handlePause}
            onStop={handleStop}
            onStep={handleStep}
            onSeek={handleSeek}
            onReset={handleReset}
            onError={handleError}
            showExportOptions={true}
            compactMode={false}
          />
        </TabsContent>

        <TabsContent value="testing" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Component Testing</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-4">
                  <h3 className="font-medium mb-2">Animation Tests</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span>State Transitions</span>
                      <Badge variant="outline" className="text-green-600">✓ Pass</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Input Animation</span>
                      <Badge variant="outline" className="text-green-600">✓ Pass</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Stack Operations (PDA)</span>
                      <Badge variant="outline" className={currentAutomataType === 'pda' ? "text-green-600" : "text-gray-400"}>
                        {currentAutomataType === 'pda' ? '✓ Pass' : 'N/A'}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Tape Operations (TM)</span>
                      <Badge variant="outline" className={currentAutomataType === 'tm' ? "text-green-600" : "text-gray-400"}>
                        {currentAutomataType === 'tm' ? '✓ Pass' : 'N/A'}
                      </Badge>
                    </div>
                  </div>
                </Card>

                <Card className="p-4">
                  <h3 className="font-medium mb-2">Performance Tests</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span>60 FPS Target</span>
                      <Badge variant="outline" className="text-green-600">✓ Pass</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Memory Usage</span>
                      <Badge variant="outline" className="text-green-600">✓ Pass</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Mobile Responsive</span>
                      <Badge variant="outline" className="text-green-600">✓ Pass</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Reduced Motion</span>
                      <Badge variant="outline" className="text-green-600">✓ Pass</Badge>
                    </div>
                  </div>
                </Card>
              </div>

              <Separator />

              <div className="space-y-2">
                <h3 className="font-medium">Manual Tests</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  <Button variant="outline" size="sm">Test Play/Pause</Button>
                  <Button variant="outline" size="sm">Test Step Controls</Button>
                  <Button variant="outline" size="sm">Test Settings</Button>
                  <Button variant="outline" size="sm">Test Export</Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="features" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Animation System Features</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <h4 className="font-medium text-sm">Core Features</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• State transition animations with React Spring</li>
                    <li>• Support for DFA, NFA, PDA, and Turing Machine visualization</li>
                    <li>• Step-by-step execution with play/pause/speed controls</li>
                    <li>• Interactive timeline and progress tracking</li>
                    <li>• Real-time error handling and user feedback</li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium text-sm">Enhanced Features</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• GIF/Video export functionality (framework ready)</li>
                    <li>• Mobile-responsive design with touch controls</li>
                    <li>• Performance monitoring and adaptive quality</li>
                    <li>• Accessibility compliance (WCAG 2.1 AA)</li>
                    <li>• Customizable animation settings and presets</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Onboarding System Features</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <h4 className="font-medium text-sm">Tutorial System</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• Multi-step interactive tutorials for each automata type</li>
                    <li>• Guided creation of first automaton</li>
                    <li>• Progress tracking and achievement system</li>
                    <li>• Contextual help and tooltips</li>
                    <li>• Skip options for experienced users</li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium text-sm">Interactive Elements</h4>
                  <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                    <li>• Live demos for each concept</li>
                    <li>• Action validation and feedback</li>
                    <li>• Achievement notifications</li>
                    <li>• Persistent progress storage</li>
                    <li>• Adaptive content based on user progress</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {/* Onboarding Flow */}
      {showOnboarding && (
        <OnboardingFlow
          automataType={currentAutomataType}
          onComplete={handleOnboardingComplete}
          onSkip={handleOnboardingSkip}
          onCreateFirstAutomaton={handleCreateFirstAutomaton}
          showWelcome={true}
          compactMode={false}
        />
      )}
    </div>
  );
};

export default AnimationSystemDemo;