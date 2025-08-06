import React, { useState, useCallback } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import { Play, BookOpen, Zap } from 'lucide-react';
import AnimationSystem from './AnimationSystem';
import InteractiveOnboarding from './InteractiveOnboarding';
import { useAnimationSystem } from '../hooks/useAnimationSystem';
import { useOnboarding } from '../hooks/useOnboarding';
import { 
  AutomataType, 
  ExtendedAutomaton, 
  SimulationStep 
} from '../types/automata';

interface AnimationDemoProps {
  automataType: AutomataType;
  className?: string;
}

// Sample simulation data for demonstration
const SAMPLE_SIMULATION_STEPS: SimulationStep[] = [
  {
    step_number: 0,
    current_state: 'q0',
    input_position: 0,
    remaining_input: 'aabb',
    action_description: 'Start state with input "aabb"'
  },
  {
    step_number: 1,
    current_state: 'q1',
    input_position: 1,
    remaining_input: 'abb',
    action_description: 'Read "a", transition to q1'
  },
  {
    step_number: 2,
    current_state: 'q1',
    input_position: 2,
    remaining_input: 'bb',
    action_description: 'Read "a", stay in q1'
  },
  {
    step_number: 3,
    current_state: 'q2',
    input_position: 3,
    remaining_input: 'b',
    action_description: 'Read "b", transition to q2'
  },
  {
    step_number: 4,
    current_state: 'q2',
    input_position: 4,
    remaining_input: '',
    action_description: 'Read "b", accept the string'
  }
];

export const AnimationDemo: React.FC<AnimationDemoProps> = ({
  automataType,
  className
}) => {
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [sampleAutomaton, setSampleAutomaton] = useState<ExtendedAutomaton | null>(null);

  // Animation system hook
  const {
    currentStep,
    isPlaying,
    play,
    pause,
    stop,
    reset,
    stepForward,
    stepBackward,
    seekTo,
    updateConfig,
    updateSettings,
    metrics
  } = useAnimationSystem({
    simulationSteps: SAMPLE_SIMULATION_STEPS,
    onStepChange: (step) => {
      console.log('Animation step changed to:', step);
    },
    onAnimationEvent: (event) => {
      console.log('Animation event:', event);
    }
  });

  // Onboarding system hook
  const {
    shouldShowOnboarding,
    startOnboarding,
    completeOnboarding,
    unlockAchievement,
    achievements,
    userProgress
  } = useOnboarding({
    automataType,
    onAchievementUnlocked: (achievement) => {
      console.log('Achievement unlocked:', achievement);
    },
    onProgressUpdate: (progress) => {
      console.log('Progress updated:', progress);
    },
    persistProgress: true
  });

  const handleStartTutorial = useCallback(() => {
    setShowOnboarding(true);
    startOnboarding();
  }, [startOnboarding]);

  const handleLoadExample = useCallback((example: ExtendedAutomaton) => {
    setSampleAutomaton(example);
    unlockAchievement('automaton_creator');
  }, [unlockAchievement]);

  const handleRunSimulation = useCallback(() => {
    if (!isPlaying) {
      play();
      unlockAchievement('simulation_novice');
    } else {
      pause();
    }
  }, [isPlaying, play, pause, unlockAchievement]);

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Zap className="h-5 w-5" />
              <span>Animation & Onboarding Demo</span>
              <Badge variant="secondary">{automataType.toUpperCase()}</Badge>
            </div>
            <div className="flex space-x-2">
              {shouldShowOnboarding() && (
                <Button
                  variant="outline"
                  onClick={handleStartTutorial}
                >
                  <BookOpen className="h-4 w-4 mr-2" />
                  Start Tutorial
                </Button>
              )}
              <Button
                onClick={handleRunSimulation}
                disabled={SAMPLE_SIMULATION_STEPS.length === 0}
              >
                <Play className="h-4 w-4 mr-2" />
                {isPlaying ? 'Pause' : 'Start'} Demo
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{achievements.filter(a => a.unlocked).length}</div>
              <div className="text-sm text-gray-600">Achievements Unlocked</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{userProgress.totalAutomataCreated}</div>
              <div className="text-sm text-gray-600">Automata Created</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{userProgress.successfulSimulations}</div>
              <div className="text-sm text-gray-600">Successful Simulations</div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Separator />

      {/* Animation System */}
      <AnimationSystem
        automaton={sampleAutomaton || {
          type: automataType,
          states: [],
          transitions: [],
          alphabet: []
        } as ExtendedAutomaton}
        simulationSteps={SAMPLE_SIMULATION_STEPS}
        currentStep={currentStep}
        isPlaying={isPlaying}
        onPlay={play}
        onPause={pause}
        onStop={stop}
        onStep={(direction) => direction === 'forward' ? stepForward() : stepBackward()}
        onSeek={seekTo}
        onReset={reset}
      />

      {/* Metrics Display */}
      <Card>
        <CardHeader>
          <CardTitle>Animation Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="font-medium">Current Step</div>
              <div className="text-gray-600">{metrics.currentStep + 1} / {metrics.totalSteps}</div>
            </div>
            <div>
              <div className="font-medium">Completion</div>
              <div className="text-gray-600">{metrics.completionRate.toFixed(1)}%</div>
            </div>
            <div>
              <div className="font-medium">User Interactions</div>
              <div className="text-gray-600">{metrics.userInteractions}</div>
            </div>
            <div>
              <div className="font-medium">Total Time</div>
              <div className="text-gray-600">{(metrics.totalAnimationTime / 1000).toFixed(1)}s</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Interactive Onboarding */}
      <InteractiveOnboarding
        automataType={automataType}
        isVisible={showOnboarding}
        onClose={() => {
          setShowOnboarding(false);
          completeOnboarding();
        }}
        onExampleLoad={handleLoadExample}
        currentAutomaton={sampleAutomaton}
      />

      {/* Demo Instructions */}
      <Card>
        <CardHeader>
          <CardTitle>Demo Instructions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <h4 className="font-medium">Animation System Features:</h4>
            <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
              <li>Play/pause/step through simulation animations</li>
              <li>Configurable animation speed and easing</li>
              <li>Visual highlighting of current states and transitions</li>
              <li>Automata-specific animations (tape, stack, parse tree)</li>
              <li>Real-time metrics and progress tracking</li>
            </ul>
          </div>

          <Separator />

          <div className="space-y-2">
            <h4 className="font-medium">Interactive Onboarding Features:</h4>
            <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
              <li>Step-by-step tutorial with automata-specific content</li>
              <li>Achievement system with progress tracking</li>
              <li>Interactive tooltips and guided examples</li>
              <li>Persistent progress storage</li>
              <li>Keyboard navigation (arrow keys, Esc)</li>
            </ul>
          </div>

          <div className="bg-blue-50 dark:bg-blue-950 p-4 rounded-lg">
            <p className="text-sm text-blue-700 dark:text-blue-300">
              <strong>Try it:</strong> Click "Start Tutorial" to experience the onboarding flow, 
              or "Start Demo" to see the animation system in action with sample simulation data.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AnimationDemo;