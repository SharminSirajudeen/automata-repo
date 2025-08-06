import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Separator } from './ui/separator';
import { 
  Play, 
  Pause, 
  Square, 
  Settings, 
  BookOpen, 
  Zap,
  Activity,
  Award,
  HelpCircle
} from 'lucide-react';
import { AutomataCanvas } from './AutomataCanvas';
import AnimationSystem from './AnimationSystem';
import InteractiveOnboarding from './InteractiveOnboarding';
import { useAnimationSystem } from '../hooks/useAnimationSystem';
import { useOnboarding } from '../hooks/useOnboarding';
import { 
  AutomataType, 
  ExtendedAutomaton, 
  SimulationStep,
  SimulationResult,
  ValidationResult 
} from '../types/automata';
import { cn } from '../lib/utils';

interface EnhancedAutomataInterfaceProps {
  automataType: AutomataType;
  initialAutomaton?: ExtendedAutomaton;
  onAutomatonChange?: (automaton: ExtendedAutomaton) => void;
  onValidationResult?: (result: ValidationResult) => void;
  onSimulationResult?: (result: SimulationResult) => void;
  readOnly?: boolean;
  showOnboarding?: boolean;
  className?: string;
}

export const EnhancedAutomataInterface: React.FC<EnhancedAutomataInterfaceProps> = ({
  automataType,
  initialAutomaton,
  onAutomatonChange,
  onValidationResult,
  onSimulationResult,
  readOnly = false,
  showOnboarding = false,
  className
}) => {
  // Component state
  const [automaton, setAutomaton] = useState<ExtendedAutomaton>(
    initialAutomaton || {
      type: automataType,
      states: [],
      transitions: [],
      alphabet: []
    } as ExtendedAutomaton
  );
  
  const [simulationSteps, setSimulationSteps] = useState<SimulationStep[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [activeTab, setActiveTab] = useState('canvas');
  const [showSettings, setShowSettings] = useState(false);
  const [testInput, setTestInput] = useState('');

  // Animation system integration
  const {
    currentStep,
    isPlaying: isAnimationPlaying,
    play: playAnimation,
    pause: pauseAnimation,
    stop: stopAnimation,
    reset: resetAnimation,
    stepForward,
    stepBackward,
    seekTo,
    updateConfig: updateAnimationConfig,
    metrics: animationMetrics
  } = useAnimationSystem({
    simulationSteps,
    onStepChange: (step) => {
      // Sync animation step with simulation highlighting
      console.log('Animation step:', step);
    },
    onAnimationEvent: (event) => {
      console.log('Animation event:', event);
    }
  });

  // Onboarding system integration
  const {
    isOnboardingActive,
    shouldShowOnboarding,
    startOnboarding,
    completeOnboarding,
    achievements,
    userProgress,
    unlockAchievement
  } = useOnboarding({
    automataType,
    onAchievementUnlocked: (achievement) => {
      console.log('Achievement unlocked:', achievement.title);
      // Could show a toast notification here
    },
    onProgressUpdate: (progress) => {
      console.log('User progress updated:', progress);
    }
  });

  // Handle automaton changes
  const handleAutomatonChange = useCallback((newAutomaton: ExtendedAutomaton) => {
    setAutomaton(newAutomaton);
    onAutomatonChange?.(newAutomaton);
    
    // Track progress
    if (!userProgress.hasCreatedAutomaton) {
      unlockAchievement('automaton_creator');
    }
  }, [onAutomatonChange, userProgress.hasCreatedAutomaton, unlockAchievement]);

  // Simulate automaton execution
  const handleSimulation = useCallback(async (input: string) => {
    if (!input.trim()) return;

    setIsSimulating(true);
    setTestInput(input);

    // Mock simulation - in real implementation, this would call the backend
    const mockSteps: SimulationStep[] = [
      {
        step_number: 0,
        current_state: automaton.states[0]?.id || 'q0',
        input_position: 0,
        remaining_input: input,
        action_description: `Starting simulation with input "${input}"`
      }
    ];

    // Add steps based on input length (mock)
    for (let i = 0; i < input.length; i++) {
      mockSteps.push({
        step_number: i + 1,
        current_state: automaton.states[Math.min(i + 1, automaton.states.length - 1)]?.id || 'q1',
        input_position: i + 1,
        remaining_input: input.slice(i + 1),
        action_description: `Read "${input[i]}", moved to next state`
      });
    }

    setSimulationSteps(mockSteps);
    setIsSimulating(false);

    // Mock simulation result
    const simulationResult: SimulationResult = {
      accepted: true,
      steps: mockSteps,
      final_state: mockSteps[mockSteps.length - 1]?.current_state || 'qf',
      execution_path: mockSteps.map(step => step.current_state)
    };

    onSimulationResult?.(simulationResult);

    // Update progress
    if (!userProgress.hasRunSimulation) {
      unlockAchievement('simulation_novice');
    }
  }, [automaton.states, onSimulationResult, userProgress.hasRunSimulation, unlockAchievement]);

  // Start tutorial if needed
  useEffect(() => {
    if (showOnboarding && shouldShowOnboarding()) {
      startOnboarding();
    }
  }, [showOnboarding, shouldShowOnboarding, startOnboarding]);

  return (
    <div className={cn("w-full space-y-4", className)}>
      {/* Header with controls */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>{automataType.toUpperCase()} Interface</span>
              <Badge variant="secondary">
                {automaton.states.length} states, {automaton.transitions.length} transitions
              </Badge>
            </CardTitle>
            
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setActiveTab('achievements')}
                className="relative"
              >
                <Award className="h-4 w-4" />
                {achievements.filter(a => a.unlocked).length > 0 && (
                  <Badge 
                    variant="destructive" 
                    className="absolute -top-2 -right-2 h-5 w-5 p-0 text-xs"
                  >
                    {achievements.filter(a => a.unlocked).length}
                  </Badge>
                )}
              </Button>
              
              {shouldShowOnboarding() && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={startOnboarding}
                >
                  <HelpCircle className="h-4 w-4 mr-1" />
                  Tutorial
                </Button>
              )}
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowSettings(!showSettings)}
              >
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main interface */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="canvas">Canvas</TabsTrigger>
          <TabsTrigger value="animation">Animation</TabsTrigger>
          <TabsTrigger value="simulation">Simulation</TabsTrigger>
          <TabsTrigger value="achievements">
            Achievements
            {achievements.filter(a => a.unlocked).length > 0 && (
              <Badge variant="secondary" className="ml-2">
                {achievements.filter(a => a.unlocked).length}
              </Badge>
            )}
          </TabsTrigger>
        </TabsList>

        {/* Canvas Tab */}
        <TabsContent value="canvas" className="space-y-4">
          <AutomataCanvas
            automaton={automaton}
            onAutomatonChange={handleAutomatonChange}
            isSimulating={isSimulating}
            simulationPath={simulationSteps.map(step => step.current_state)}
            currentSimulationStep={currentStep}
            readOnly={readOnly}
            automatonType={automataType}
          />
        </TabsContent>

        {/* Animation Tab */}
        <TabsContent value="animation" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2">
              <AnimationSystem
                automaton={automaton}
                simulationSteps={simulationSteps}
                currentStep={currentStep}
                isPlaying={isAnimationPlaying}
                onPlay={playAnimation}
                onPause={pauseAnimation}
                onStop={stopAnimation}
                onStep={(direction) => direction === 'forward' ? stepForward() : stepBackward()}
                onSeek={seekTo}
                onReset={resetAnimation}
              />
            </div>
            
            {/* Animation controls sidebar */}
            <Card>
              <CardHeader>
                <CardTitle>Animation Controls</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Button
                    onClick={() => handleSimulation('aabb')}
                    disabled={isSimulating}
                    className="w-full"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Test Input: "aabb"
                  </Button>
                  
                  <Button
                    onClick={() => handleSimulation('abab')}
                    disabled={isSimulating}
                    className="w-full"
                    variant="outline"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Test Input: "abab"
                  </Button>
                </div>
                
                <Separator />
                
                <div className="space-y-2">
                  <div className="text-sm font-medium">Animation Metrics</div>
                  <div className="space-y-1 text-xs text-gray-600">
                    <div>Steps: {animationMetrics.currentStep + 1} / {animationMetrics.totalSteps}</div>
                    <div>Completion: {animationMetrics.completionRate.toFixed(1)}%</div>
                    <div>Interactions: {animationMetrics.userInteractions}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Simulation Tab */}
        <TabsContent value="simulation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Simulation Results</CardTitle>
            </CardHeader>
            <CardContent>
              {simulationSteps.length > 0 ? (
                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <Badge variant={simulationSteps.length > 0 ? "default" : "secondary"}>
                      {simulationSteps.length > 0 ? "Completed" : "No simulation"}
                    </Badge>
                    <span className="text-sm">Input: "{testInput}"</span>
                  </div>
                  
                  <div className="space-y-2">
                    {simulationSteps.map((step, index) => (
                      <div
                        key={index}
                        className={cn(
                          "p-2 rounded border text-sm",
                          index === currentStep 
                            ? "border-blue-500 bg-blue-50 dark:bg-blue-950" 
                            : "border-gray-200 dark:border-gray-700"
                        )}
                      >
                        <div className="font-medium">Step {step.step_number}</div>
                        <div className="text-gray-600 dark:text-gray-400">
                          {step.action_description}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No simulation results yet. Run a simulation to see step-by-step execution.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Achievements Tab */}
        <TabsContent value="achievements" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Achievements & Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {achievements.map(achievement => (
                  <div
                    key={achievement.id}
                    className={cn(
                      "p-4 rounded-lg border",
                      achievement.unlocked
                        ? "border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950"
                        : "border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900"
                    )}
                  >
                    <div className="flex items-start space-x-3">
                      <div className={cn(
                        "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center",
                        achievement.unlocked ? "bg-green-500" : "bg-gray-400"
                      )}>
                        {typeof achievement.icon === 'string' ? achievement.icon : achievement.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium">{achievement.title}</div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {achievement.description}
                        </div>
                        <div className="mt-2 text-xs">
                          Progress: {achievement.progress} / {achievement.maxProgress}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Interactive Onboarding */}
      <InteractiveOnboarding
        automataType={automataType}
        isVisible={isOnboardingActive}
        onClose={completeOnboarding}
        onExampleLoad={(example) => {
          setAutomaton(example);
          handleAutomatonChange(example);
        }}
        currentAutomaton={automaton}
      />
    </div>
  );
};

export default EnhancedAutomataInterface;