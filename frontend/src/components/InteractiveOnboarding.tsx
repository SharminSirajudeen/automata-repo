import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useSpring, animated, useTransition, config } from '@react-spring/web';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Separator } from './ui/separator';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { 
  ChevronLeft, 
  ChevronRight, 
  X, 
  Target, 
  Award, 
  Star, 
  CheckCircle2,
  Circle,
  Lightbulb,
  BookOpen,
  Zap,
  Trophy,
  ArrowRight,
  Play,
  HelpCircle,
  Sparkles
} from 'lucide-react';
import { AutomataType, ExtendedAutomaton, State, Transition, Problem } from '../types/automata';
import { cn } from '../lib/utils';

export interface TutorialStep {
  id: string;
  title: string;
  description: string;
  target?: string; // CSS selector for highlighting
  content: React.ReactNode;
  position: 'top' | 'bottom' | 'left' | 'right' | 'center';
  canSkip: boolean;
  actionRequired?: boolean;
  validationFn?: () => boolean;
}

export interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  unlocked: boolean;
  progress: number;
  maxProgress: number;
  category: 'tutorial' | 'creation' | 'simulation' | 'advanced';
}

export interface OnboardingState {
  currentStep: number;
  completedSteps: Set<string>;
  achievements: Achievement[];
  userProgress: {
    hasCreatedAutomaton: boolean;
    hasRunSimulation: boolean;
    hasUsedAI: boolean;
    totalAutomataCreated: number;
    successfulSimulations: number;
  };
  isFirstVisit: boolean;
}

interface InteractiveOnboardingProps {
  automataType: AutomataType;
  isVisible: boolean;
  onClose: () => void;
  onAutomatonCreate?: (automaton: ExtendedAutomaton) => void;
  onExampleLoad?: (example: ExtendedAutomaton) => void;
  currentAutomaton?: ExtendedAutomaton;
  className?: string;
}

const EXAMPLE_AUTOMATA: { [key in AutomataType]: ExtendedAutomaton } = {
  dfa: {
    type: 'dfa',
    states: [
      { id: 'q0', x: 100, y: 100, is_start: true, is_accept: false, label: 'q0' },
      { id: 'q1', x: 250, y: 100, is_start: false, is_accept: true, label: 'q1' }
    ],
    transitions: [
      { from_state: 'q0', to_state: 'q1', symbol: 'a' },
      { from_state: 'q1', to_state: 'q1', symbol: 'a,b' }
    ],
    alphabet: ['a', 'b']
  } as ExtendedAutomaton,
  nfa: {
    type: 'nfa',
    states: [
      { id: 'q0', x: 100, y: 100, is_start: true, is_accept: false },
      { id: 'q1', x: 200, y: 100, is_start: false, is_accept: false },
      { id: 'q2', x: 300, y: 100, is_start: false, is_accept: true }
    ],
    transitions: [
      { from_state: 'q0', to_state: 'q1', symbol: 'a' },
      { from_state: 'q0', to_state: 'q2', symbol: 'a' },
      { from_state: 'q1', to_state: 'q2', symbol: 'b' }
    ],
    alphabet: ['a', 'b']
  } as ExtendedAutomaton,
  enfa: {
    type: 'enfa', 
    states: [
      { id: 'q0', x: 100, y: 100, is_start: true, is_accept: false },
      { id: 'q1', x: 250, y: 100, is_start: false, is_accept: true }
    ],
    transitions: [
      { from_state: 'q0', to_state: 'q1', symbol: 'ε' },
      { from_state: 'q1', to_state: 'q1', symbol: 'a' }
    ],
    alphabet: ['a']
  } as ExtendedAutomaton,
  pda: {
    type: 'pda',
    states: [
      { id: 'q0', x: 100, y: 100, is_start: true, is_accept: false },
      { id: 'q1', x: 250, y: 100, is_start: false, is_accept: true }
    ],
    transitions: [
      { from_state: 'q0', to_state: 'q1', symbol: 'a', stack_pop: 'Z', stack_push: 'aZ' },
      { from_state: 'q1', to_state: 'q1', symbol: 'b', stack_pop: 'a', stack_push: '' }
    ],
    alphabet: ['a', 'b'],
    stack_alphabet: ['a', 'Z'],
    start_stack_symbol: 'Z'
  } as ExtendedAutomaton,
  cfg: {
    type: 'cfg',
    terminals: ['a', 'b'],
    non_terminals: ['S'],
    productions: [
      { id: 'p1', left_side: 'S', right_side: 'aSb' },
      { id: 'p2', left_side: 'S', right_side: 'ε' }
    ],
    start_symbol: 'S'
  } as ExtendedAutomaton,
  tm: {
    type: 'tm',
    states: [
      { id: 'q0', x: 100, y: 100, is_start: true, is_accept: false },
      { id: 'q1', x: 250, y: 100, is_start: false, is_accept: true }
    ],
    transitions: [
      { from_state: 'q0', to_state: 'q1', read_symbol: 'a', write_symbol: 'b', head_direction: 'R' }
    ],
    tape_alphabet: ['a', 'b', '□'],
    blank_symbol: '□'
  } as ExtendedAutomaton,
  regex: {
    type: 'regex',
    pattern: 'a*b+',
    alphabet: ['a', 'b']
  } as ExtendedAutomaton,
  pumping: {
    type: 'pumping',
    language_type: 'regular',
    language_description: 'The language of all strings with equal number of a\'s and b\'s'
  } as ExtendedAutomaton
};

export const InteractiveOnboarding: React.FC<InteractiveOnboardingProps> = ({
  automataType,
  isVisible,
  onClose,
  onAutomatonCreate,
  onExampleLoad,
  currentAutomaton,
  className
}) => {
  const [onboardingState, setOnboardingState] = useState<OnboardingState>({
    currentStep: 0,
    completedSteps: new Set(),
    achievements: [],
    userProgress: {
      hasCreatedAutomaton: false,
      hasRunSimulation: false,
      hasUsedAI: false,
      totalAutomataCreated: 0,
      successfulSimulations: 0
    },
    isFirstVisit: true
  });

  const [showAchievements, setShowAchievements] = useState(false);
  const [highlightedElement, setHighlightedElement] = useState<string | null>(null);
  const [tooltipStep, setTooltipStep] = useState<string | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  // Initialize achievements
  useEffect(() => {
    const achievements: Achievement[] = [
      {
        id: 'first_steps',
        title: 'First Steps',
        description: 'Complete the tutorial introduction',
        icon: <Star className="h-4 w-4" />,
        unlocked: false,
        progress: 0,
        maxProgress: 1,
        category: 'tutorial'
      },
      {
        id: 'automaton_creator',
        title: 'Automaton Creator',
        description: 'Create your first automaton',
        icon: <Target className="h-4 w-4" />,
        unlocked: false,
        progress: 0,
        maxProgress: 1,
        category: 'creation'
      },
      {
        id: 'simulation_master',
        title: 'Simulation Master',
        description: 'Run 5 successful simulations',
        icon: <Zap className="h-4 w-4" />,
        unlocked: false,
        progress: 0,
        maxProgress: 5,
        category: 'simulation'
      },
      {
        id: 'ai_explorer',
        title: 'AI Explorer',
        description: 'Use AI assistance for guidance',
        icon: <Sparkles className="h-4 w-4" />,
        unlocked: false,
        progress: 0,
        maxProgress: 1,
        category: 'advanced'
      }
    ];

    setOnboardingState(prev => ({ ...prev, achievements }));
  }, []);

  // Tutorial steps for different automata types
  const getTutorialSteps = useCallback((): TutorialStep[] => {
    const baseSteps: TutorialStep[] = [
      {
        id: 'welcome',
        title: 'Welcome to Automata Theory!',
        description: 'Let\'s get you started with creating and simulating automata',
        content: (
          <div className="space-y-4">
            <div className="text-center">
              <div className="mx-auto w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mb-4">
                <BookOpen className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Welcome to Automata Theory!</h3>
              <p className="text-gray-600 dark:text-gray-400">
                This interactive tutorial will guide you through the basics of creating and simulating {automataType.toUpperCase()}s.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                <Target className="h-6 w-6 mx-auto mb-2 text-blue-600" />
                <div className="text-sm font-medium">Create</div>
                <div className="text-xs text-gray-500">Build your automaton</div>
              </div>
              <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                <Play className="h-6 w-6 mx-auto mb-2 text-green-600" />
                <div className="text-sm font-medium">Simulate</div>
                <div className="text-xs text-gray-500">Test with inputs</div>
              </div>
            </div>
          </div>
        ),
        position: 'center',
        canSkip: false,
        actionRequired: false
      },
      {
        id: 'interface_overview',
        title: 'Interface Overview',
        description: 'Let\'s explore the main components of the interface',
        content: (
          <div className="space-y-4">
            <h3 className="font-semibold">Main Interface Components</h3>
            <div className="space-y-3">
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-sm">Canvas: Where you create your automaton</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-sm">Controls: Play, pause, and step through simulations</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                <span className="text-sm">Inspector: View and modify automaton details</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                <span className="text-sm">AI Assistant: Get help and suggestions</span>
              </div>
            </div>
          </div>
        ),
        position: 'center',
        canSkip: true,
        actionRequired: false
      }
    ];

    // Add automata-specific steps
    const automatonSpecificSteps: TutorialStep[] = [];

    switch (automataType) {
      case 'dfa':
      case 'nfa':
        automatonSpecificSteps.push({
          id: 'finite_automaton_basics',
          title: 'Finite Automaton Basics',
          description: 'Learn about states and transitions',
          content: (
            <div className="space-y-4">
              <h3 className="font-semibold">Finite Automaton Components</h3>
              <div className="space-y-3">
                <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <div className="font-medium text-sm">States</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    Circles representing the current position in computation
                  </div>
                </div>
                <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                  <div className="font-medium text-sm">Transitions</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    Arrows showing how to move between states on input
                  </div>
                </div>
                <div className="p-3 bg-purple-50 dark:bg-purple-950 rounded-lg">
                  <div className="font-medium text-sm">Acceptance</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    Double circles indicate accepting states
                  </div>
                </div>
              </div>
            </div>
          ),
          position: 'center',
          canSkip: false,
          actionRequired: false
        });
        break;

      case 'pda':
        automatonSpecificSteps.push({
          id: 'pda_basics',
          title: 'Pushdown Automaton Basics',
          description: 'Understanding stack operations',
          content: (
            <div className="space-y-4">
              <h3 className="font-semibold">PDA Components</h3>
              <div className="space-y-3">
                <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <div className="font-medium text-sm">Stack</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    LIFO memory for storing symbols
                  </div>
                </div>
                <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                  <div className="font-medium text-sm">Push/Pop Operations</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    Transitions can push to or pop from stack
                  </div>
                </div>
              </div>
            </div>
          ),
          position: 'center',
          canSkip: false,
          actionRequired: false
        });
        break;

      case 'tm':
        automatonSpecificSteps.push({
          id: 'tm_basics',
          title: 'Turing Machine Basics',
          description: 'Understanding tape operations',
          content: (
            <div className="space-y-4">
              <h3 className="font-semibold">Turing Machine Components</h3>
              <div className="space-y-3">
                <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <div className="font-medium text-sm">Tape</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    Infinite memory with read/write head
                  </div>
                </div>
                <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                  <div className="font-medium text-sm">Head Movement</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    Can move left (L), right (R), or stay (S)
                  </div>
                </div>
              </div>
            </div>
          ),
          position: 'center',
          canSkip: false,
          actionRequired: false
        });
        break;
    }

    const finalSteps: TutorialStep[] = [
      {
        id: 'try_example',
        title: 'Try an Example',
        description: 'Load a sample automaton to get started',
        content: (
          <div className="space-y-4">
            <h3 className="font-semibold">Ready to Practice?</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Let's load an example {automataType.toUpperCase()} to see how it works.
            </p>
            <Button 
              onClick={() => handleLoadExample()}
              className="w-full"
            >
              Load Example {automataType.toUpperCase()}
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        ),
        position: 'center',
        canSkip: true,
        actionRequired: true
      },
      {
        id: 'completion',
        title: 'Tutorial Complete!',
        description: 'You\'re ready to create your own automata',
        content: (
          <div className="space-y-4 text-center">
            <div className="mx-auto w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-600 rounded-full flex items-center justify-center mb-4">
              <Trophy className="h-8 w-8 text-white" />
            </div>
            <h3 className="text-lg font-semibold">Congratulations!</h3>
            <p className="text-gray-600 dark:text-gray-400">
              You've completed the tutorial. You're now ready to create and simulate your own automata!
            </p>
            <div className="flex space-x-2 justify-center">
              <Badge variant="secondary">
                <Star className="h-3 w-3 mr-1" />
                Tutorial Complete
              </Badge>
            </div>
          </div>
        ),
        position: 'center',
        canSkip: false,
        actionRequired: false
      }
    ];

    return [...baseSteps, ...automatonSpecificSteps, ...finalSteps];
  }, [automataType]);

  const tutorialSteps = getTutorialSteps();
  const currentTutorialStep = tutorialSteps[onboardingState.currentStep];

  // Animation springs
  const modalSpring = useSpring({
    opacity: isVisible ? 1 : 0,
    transform: isVisible ? 'scale(1)' : 'scale(0.9)',
    config: config.gentle
  });

  const stepTransitions = useTransition(
    currentTutorialStep ? [currentTutorialStep] : [],
    {
      from: { opacity: 0, transform: 'translateX(50px)' },
      enter: { opacity: 1, transform: 'translateX(0px)' },
      leave: { opacity: 0, transform: 'translateX(-50px)' },
      keys: (item) => item.id
    }
  );

  const achievementTransitions = useTransition(
    onboardingState.achievements.filter(a => a.unlocked),
    {
      from: { opacity: 0, scale: 0.8, transform: 'translateY(-10px)' },
      enter: { opacity: 1, scale: 1, transform: 'translateY(0px)' },
      leave: { opacity: 0, scale: 0.8, transform: 'translateY(10px)' },
      keys: (item) => item.id,
      trail: 100
    }
  );

  // Tutorial navigation
  const handleNext = useCallback(() => {
    if (currentTutorialStep?.validationFn && !currentTutorialStep.validationFn()) {
      return; // Don't advance if validation fails
    }

    const nextStep = onboardingState.currentStep + 1;
    if (nextStep < tutorialSteps.length) {
      setOnboardingState(prev => ({
        ...prev,
        currentStep: nextStep,
        completedSteps: new Set([...prev.completedSteps, currentTutorialStep.id])
      }));
      unlockAchievement('first_steps');
    } else {
      handleComplete();
    }
  }, [onboardingState.currentStep, tutorialSteps.length, currentTutorialStep]);

  const handlePrevious = useCallback(() => {
    if (onboardingState.currentStep > 0) {
      setOnboardingState(prev => ({
        ...prev,
        currentStep: prev.currentStep - 1
      }));
    }
  }, [onboardingState.currentStep]);

  const handleSkip = useCallback(() => {
    if (currentTutorialStep?.canSkip) {
      handleNext();
    }
  }, [currentTutorialStep, handleNext]);

  const handleComplete = useCallback(() => {
    unlockAchievement('first_steps');
    onClose();
  }, [onClose]);

  const handleLoadExample = useCallback(() => {
    const example = EXAMPLE_AUTOMATA[automataType];
    if (example && onExampleLoad) {
      onExampleLoad(example);
      unlockAchievement('automaton_creator');
      setOnboardingState(prev => ({
        ...prev,
        userProgress: {
          ...prev.userProgress,
          hasCreatedAutomaton: true,
          totalAutomataCreated: prev.userProgress.totalAutomataCreated + 1
        }
      }));
    }
  }, [automataType, onExampleLoad]);

  const unlockAchievement = useCallback((achievementId: string) => {
    setOnboardingState(prev => {
      const newAchievements = prev.achievements.map(achievement => {
        if (achievement.id === achievementId && !achievement.unlocked) {
          return { ...achievement, unlocked: true, progress: achievement.maxProgress };
        }
        return achievement;
      });

      // Show achievement notification if newly unlocked
      const wasUnlocked = prev.achievements.find(a => a.id === achievementId)?.unlocked;
      if (!wasUnlocked) {
        setShowAchievements(true);
      }

      return { ...prev, achievements: newAchievements };
    });
  }, []);

  // Keyboard navigation
  useEffect(() => {
    if (!isVisible) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowLeft':
          if (onboardingState.currentStep > 0) handlePrevious();
          break;
        case 'ArrowRight':
          if (onboardingState.currentStep < tutorialSteps.length - 1) handleNext();
          break;
        case 'Escape':
          onClose();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isVisible, onboardingState.currentStep, tutorialSteps.length, handleNext, handlePrevious, onClose]);

  if (!isVisible) return null;

  return (
    <TooltipProvider>
      <Dialog open={isVisible} onOpenChange={onClose}>
        <DialogContent className={cn("max-w-4xl h-[80vh] flex flex-col", className)}>
          <DialogHeader className="flex-shrink-0">
            <div className="flex items-center justify-between">
              <DialogTitle className="flex items-center space-x-2">
                <BookOpen className="h-5 w-5" />
                <span>Interactive Tutorial - {automataType.toUpperCase()}</span>
              </DialogTitle>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowAchievements(!showAchievements)}
                >
                  <Award className="h-4 w-4" />
                </Button>
                <Button variant="ghost" size="sm" onClick={onClose}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
            
            {/* Progress indicator */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
                <span>Tutorial Progress</span>
                <span>{onboardingState.currentStep + 1} / {tutorialSteps.length}</span>
              </div>
              <Progress 
                value={((onboardingState.currentStep + 1) / tutorialSteps.length) * 100} 
                className="h-2"
              />
            </div>
          </DialogHeader>

          <div className="flex-1 flex">
            {/* Main tutorial content */}
            <div className="flex-1 overflow-y-auto pr-4">
              {stepTransitions((style, step) => (
                <animated.div style={style} className="h-full">
                  <Card className="h-full">
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2">
                        <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
                          {onboardingState.currentStep + 1}
                        </div>
                        <span>{step.title}</span>
                      </CardTitle>
                      {step.description && (
                        <p className="text-gray-600 dark:text-gray-400">{step.description}</p>
                      )}
                    </CardHeader>
                    <CardContent className="flex-1">
                      {step.content}
                    </CardContent>
                  </Card>
                </animated.div>
              ))}
            </div>

            {/* Achievements sidebar */}
            {showAchievements && (
              <div className="w-80 ml-4">
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Trophy className="h-4 w-4" />
                      <span>Achievements</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {onboardingState.achievements.map(achievement => (
                      <div
                        key={achievement.id}
                        className={cn(
                          "p-3 rounded-lg border",
                          achievement.unlocked
                            ? "bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800"
                            : "bg-gray-50 border-gray-200 dark:bg-gray-900 dark:border-gray-700"
                        )}
                      >
                        <div className="flex items-start space-x-3">
                          <div className={cn(
                            "flex-shrink-0 p-1 rounded",
                            achievement.unlocked
                              ? "bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-400"
                              : "bg-gray-100 text-gray-400 dark:bg-gray-800"
                          )}>
                            {achievement.unlocked ? <CheckCircle2 className="h-4 w-4" /> : achievement.icon}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-sm">{achievement.title}</div>
                            <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                              {achievement.description}
                            </div>
                            <div className="mt-2">
                              <Progress 
                                value={(achievement.progress / achievement.maxProgress) * 100}
                                className="h-1"
                              />
                              <div className="text-xs text-gray-500 mt-1">
                                {achievement.progress} / {achievement.maxProgress}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>
            )}
          </div>

          {/* Navigation controls */}
          <div className="flex-shrink-0 border-t pt-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {onboardingState.currentStep > 0 && (
                  <Button variant="outline" onClick={handlePrevious}>
                    <ChevronLeft className="h-4 w-4 mr-1" />
                    Previous
                  </Button>
                )}
              </div>

              <div className="flex items-center space-x-2">
                {currentTutorialStep?.canSkip && (
                  <Button variant="ghost" onClick={handleSkip}>
                    Skip
                  </Button>
                )}
                
                {onboardingState.currentStep < tutorialSteps.length - 1 ? (
                  <Button 
                    onClick={handleNext}
                    disabled={currentTutorialStep?.actionRequired && currentTutorialStep?.validationFn && !currentTutorialStep.validationFn()}
                  >
                    Next
                    <ChevronRight className="h-4 w-4 ml-1" />
                  </Button>
                ) : (
                  <Button onClick={handleComplete}>
                    Complete Tutorial
                    <CheckCircle2 className="h-4 w-4 ml-1" />
                  </Button>
                )}
              </div>
            </div>

            {/* Keyboard shortcuts hint */}
            <div className="text-xs text-gray-500 text-center mt-2">
              Use arrow keys to navigate • Press Esc to close
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  );
};

export default InteractiveOnboarding;