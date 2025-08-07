import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useOnboarding } from '../hooks/useOnboarding';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Separator } from './ui/separator';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import {
  Play,
  ChevronLeft,
  ChevronRight,
  X,
  Check,
  BookOpen,
  Lightbulb,
  Target,
  Award,
  Clock,
  Users,
  Zap,
  Brain,
  Code,
  Eye,
  Sparkles,
  ArrowRight,
  Coffee,
  Rocket,
  Star,
  Trophy,
  Gift
} from 'lucide-react';
import { AutomataType, ExtendedAutomaton, State, Transition } from '../types/automata';
import { OnboardingStep, Achievement } from '../types/animation';
import { cn } from '../lib/utils';

interface OnboardingFlowProps {
  automataType: AutomataType;
  onComplete: () => void;
  onSkip: () => void;
  onCreateFirstAutomaton?: (automaton: ExtendedAutomaton) => void;
  className?: string;
  showWelcome?: boolean;
  compactMode?: boolean;
}

interface StepContent {
  title: string;
  description: string;
  interactiveDemo?: React.ReactNode;
  actionRequired?: boolean;
  validationFn?: () => boolean;
  tips?: string[];
}

export const OnboardingFlow: React.FC<OnboardingFlowProps> = ({
  automataType,
  onComplete,
  onSkip,
  onCreateFirstAutomaton,
  className,
  showWelcome = true,
  compactMode = false
}) => {
  const {
    isOnboardingActive,
    currentStep,
    tutorialState,
    userProgress,
    achievements,
    startOnboarding,
    completeOnboarding,
    skipOnboarding,
    nextStep,
    previousStep,
    goToStep,
    markStepComplete,
    unlockAchievement,
    getCompletionRate
  } = useOnboarding({
    automataType,
    onAchievementUnlocked: (achievement) => {
      // Show achievement notification
      setNewAchievement(achievement);
    },
    onProgressUpdate: (progress) => {
      // Could trigger external updates if needed
    }
  });

  const [isVisible, setIsVisible] = useState(showWelcome);
  const [newAchievement, setNewAchievement] = useState<Achievement | null>(null);
  const [demoAutomaton, setDemoAutomaton] = useState<ExtendedAutomaton | null>(null);
  const [hasInteracted, setHasInteracted] = useState(false);

  // Memoize interactive components to prevent re-creation
  const BasicConceptDemoMemo = useMemo(() => <BasicConceptDemo automataType={automataType} />, [automataType]);
  const StateCreationDemoMemo = useMemo(() => <StateCreationDemo onInteract={() => setHasInteracted(true)} />, []);
  const TransitionDemoMemo = useMemo(() => <TransitionDemo automataType={automataType} />, [automataType]);
  const StackDemoMemo = useMemo(() => <StackDemo />, []);
  const TapeDemoMemo = useMemo(() => <TapeDemo />, []);
  
  // Define onboarding steps based on automata type
  const onboardingSteps: StepContent[] = useMemo(() => {
    const baseSteps = [
      {
        title: 'Welcome to Automata Learning',
        description: `Let's start your journey with ${automataType.toUpperCase()}s! This tutorial will guide you through the basics and help you create your first automaton.`,
        tips: [
          'Take your time - learning theory takes practice',
          'Interactive demos help you understand concepts better',
          'Don\'t hesitate to revisit steps if needed'
        ]
      },
      {
        title: 'Understanding the Basics',
        description: getBasicDescription(automataType),
        interactiveDemo: BasicConceptDemoMemo,
        tips: [
          'Pay attention to the visual representation',
          'Each component has a specific role',
          'Understanding these basics is crucial for building automata'
        ]
      },
      {
        title: 'Creating Your First State',
        description: 'States are the building blocks of automata. Let\'s create your first state and understand its properties.',
        actionRequired: true,
        validationFn: () => hasInteracted,
        interactiveDemo: StateCreationDemoMemo,
        tips: [
          'Click on the canvas to create a state',
          'States can be start states, accept states, or both',
          'Each state needs a unique identifier'
        ]
      },
      {
        title: 'Adding Transitions',
        description: 'Transitions connect states and define how the automaton processes input. Let\'s add some transitions.',
        actionRequired: true,
        interactiveDemo: TransitionDemoMemo,
        tips: [
          'Drag from one state to another to create a transition',
          'Each transition needs an input symbol',
          'Self-loops are transitions from a state to itself'
        ]
      }
    ];

    // Add type-specific steps
    switch (automataType) {
      case 'dfa':
        baseSteps.push({
          title: 'DFA Completeness',
          description: 'A DFA must have exactly one transition for each input symbol from every state. Let\'s ensure completeness.',
          actionRequired: true,
          tips: [
            'Every state must have outgoing transitions for all input symbols',
            'No state can have multiple transitions for the same symbol',
            'Missing transitions mean the automaton rejects the input'
          ]
        });
        break;
      case 'nfa':
        baseSteps.push({
          title: 'NFA Non-determinism',
          description: 'NFAs can have multiple transitions for the same symbol, or ε-transitions. Let\'s explore this flexibility.',
          tips: [
            'Multiple transitions for the same symbol create non-determinism',
            'ε-transitions don\'t consume input',
            'NFAs accept if any path leads to an accept state'
          ]
        });
        break;
      case 'pda':
        baseSteps.push({
          title: 'Understanding the Stack',
          description: 'PDAs use a stack for additional memory. Let\'s learn how stack operations work.',
          interactiveDemo: StackDemoMemo,
          tips: [
            'The stack provides unlimited memory',
            'Transitions can push, pop, or ignore the stack',
            'Stack operations are written as pop,push format'
          ]
        });
        break;
      case 'tm':
        baseSteps.push({
          title: 'The Turing Machine Tape',
          description: 'Turing machines have an infinite tape that can be read and written. Let\'s explore tape operations.',
          interactiveDemo: TapeDemoMemo,
          tips: [
            'The tape is infinite in both directions',
            'The head can move left, right, or stay',
            'Each transition reads, writes, and moves'
          ]
        });
        break;
    }

    baseSteps.push({
      title: 'Testing Your Automaton',
      description: 'Now let\'s test your automaton with some input strings to see how it behaves.',
      actionRequired: true,
      tips: [
        'Start with simple test strings',
        'Watch the step-by-step execution',
        'Understand why strings are accepted or rejected'
      ]
    });

    baseSteps.push({
      title: 'Congratulations!',
      description: 'You\'ve successfully completed the tutorial! You now know the basics of creating and testing automata.',
      tips: [
        'Practice with more complex examples',
        'Explore the AI assistant for help',
        'Check out the example gallery for inspiration'
      ]
    });

    return baseSteps;
  }, [automataType, hasInteracted, BasicConceptDemoMemo, StateCreationDemoMemo, TransitionDemoMemo, StackDemoMemo, TapeDemoMemo]);

  // Memoize current step content and completion rate to prevent unnecessary re-renders
  const currentStepContent = useMemo(() => onboardingSteps[currentStep] || onboardingSteps[0], [onboardingSteps, currentStep]);
  const completionRate = useMemo(() => getCompletionRate(), [getCompletionRate]);
  const progressPercentage = useMemo(() => Math.round((currentStep / (onboardingSteps.length - 1)) * 100), [currentStep, onboardingSteps.length]);

  const handleStart = useCallback(() => {
    setIsVisible(true);
    startOnboarding();
  }, [startOnboarding]);

  const handleSkip = useCallback(() => {
    skipOnboarding();
    onSkip();
    setIsVisible(false);
  }, [skipOnboarding, onSkip]);

  const handleComplete = useCallback(() => {
    completeOnboarding();
    onComplete();
    setIsVisible(false);
    unlockAchievement('first_steps');
  }, [completeOnboarding, onComplete, unlockAchievement]);

  const handleNext = useCallback(() => {
    if (currentStepContent.actionRequired && currentStepContent.validationFn) {
      if (!currentStepContent.validationFn()) {
        return; // Don't proceed if validation fails
      }
    }

    markStepComplete(`step-${currentStep}`);

    if (currentStep >= onboardingSteps.length - 1) {
      handleComplete();
    } else {
      nextStep();
    }
  }, [currentStep, currentStepContent, markStepComplete, handleComplete, nextStep, onboardingSteps.length]);

  const handlePrevious = useCallback(() => {
    if (currentStep > 0) {
      previousStep();
    }
  }, [currentStep, previousStep]);

  // Achievement notification effect
  useEffect(() => {
    if (newAchievement) {
      const timer = setTimeout(() => setNewAchievement(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [newAchievement]);

  if (!isVisible) return null;

  return (
    <TooltipProvider>
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4"
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className={cn(
              "w-full max-w-4xl max-h-[90vh] overflow-hidden",
              compactMode && "max-w-2xl"
            )}
          >
            <Card className="w-full h-full">
              <CardHeader className={cn("pb-4", compactMode && "pb-3 px-4")}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center space-x-2">
                      <BookOpen className="h-6 w-6 text-primary" />
                      <CardTitle className={cn("text-xl", compactMode && "text-lg")}>
                        {automataType.toUpperCase()} Tutorial
                      </CardTitle>
                    </div>
                    <Badge variant="secondary">
                      Step {currentStep + 1} of {onboardingSteps.length}
                    </Badge>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={handleSkip}
                          className="text-gray-500 hover:text-gray-700"
                        >
                          Skip Tutorial
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Skip and go straight to the editor</TooltipContent>
                    </Tooltip>
                    
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleSkip}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                
                {/* Progress Bar */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm text-gray-600">
                    <span>Progress</span>
                    <span>{progressPercentage}%</span>
                  </div>
                  <Progress value={progressPercentage} />
                </div>
              </CardHeader>

              <CardContent className={cn("flex-1 overflow-y-auto", compactMode && "px-4")}>
                <AnimatePresence mode="wait">
                  <motion.div
                    key={currentStep}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-6"
                  >
                    {/* Step Header */}
                    <div className="space-y-3">
                      <div className="flex items-center space-x-2">
                        {currentStep === 0 && <Rocket className="h-5 w-5 text-blue-500" />}
                        {currentStep === 1 && <Brain className="h-5 w-5 text-purple-500" />}
                        {currentStep === 2 && <Target className="h-5 w-5 text-green-500" />}
                        {currentStep === 3 && <Zap className="h-5 w-5 text-yellow-500" />}
                        {currentStep >= 4 && <Star className="h-5 w-5 text-orange-500" />}
                        <h2 className="text-2xl font-bold">{currentStepContent.title}</h2>
                      </div>
                      <p className="text-gray-600 dark:text-gray-300 text-lg leading-relaxed">
                        {currentStepContent.description}
                      </p>
                    </div>

                    {/* Interactive Demo */}
                    {currentStepContent.interactiveDemo && (
                      <div className="space-y-3">
                        <div className="flex items-center space-x-2">
                          <Eye className="h-4 w-4 text-blue-500" />
                          <h3 className="font-medium">Interactive Demo</h3>
                        </div>
                        <div className="p-6 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg border-2 border-dashed border-blue-200 dark:border-blue-700">
                          {currentStepContent.interactiveDemo}
                        </div>
                      </div>
                    )}

                    {/* Tips */}
                    {currentStepContent.tips && currentStepContent.tips.length > 0 && (
                      <div className="space-y-3">
                        <div className="flex items-center space-x-2">
                          <Lightbulb className="h-4 w-4 text-yellow-500" />
                          <h3 className="font-medium">Tips</h3>
                        </div>
                        <div className="grid gap-3">
                          {currentStepContent.tips.map((tip, index) => (
                            <motion.div
                              key={index}
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: index * 0.1 }}
                              className="flex items-start space-x-3 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800"
                            >
                              <Sparkles className="h-4 w-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                              <span className="text-sm">{tip}</span>
                            </motion.div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Action Required Notice */}
                    {currentStepContent.actionRequired && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="p-4 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 rounded-r-lg"
                      >
                        <div className="flex items-center space-x-2">
                          <Target className="h-5 w-5 text-blue-500" />
                          <span className="font-medium text-blue-800 dark:text-blue-200">
                            Action Required
                          </span>
                        </div>
                        <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                          Complete the interactive demo above to continue.
                        </p>
                      </motion.div>
                    )}
                  </motion.div>
                </AnimatePresence>

                {/* Navigation */}
                <Separator className="my-6" />
                <div className="flex items-center justify-between">
                  <Button
                    variant="outline"
                    onClick={handlePrevious}
                    disabled={currentStep === 0}
                    className="flex items-center space-x-2"
                  >
                    <ChevronLeft className="h-4 w-4" />
                    <span>Previous</span>
                  </Button>

                  <div className="flex items-center space-x-4">
                    {/* Step indicators */}
                    <div className="flex space-x-2">
                      {onboardingSteps.map((_, index) => (
                        <button
                          key={index}
                          onClick={() => goToStep(index)}
                          className={cn(
                            "w-2 h-2 rounded-full transition-colors",
                            index === currentStep
                              ? "bg-primary"
                              : index < currentStep
                              ? "bg-green-500"
                              : "bg-gray-300 dark:bg-gray-600"
                          )}
                        />
                      ))}
                    </div>
                  </div>

                  {currentStep === onboardingSteps.length - 1 ? (
                    <Button onClick={handleComplete} className="flex items-center space-x-2">
                      <Trophy className="h-4 w-4" />
                      <span>Complete Tutorial</span>
                    </Button>
                  ) : (
                    <Button 
                      onClick={handleNext}
                      disabled={
                        currentStepContent.actionRequired && 
                        currentStepContent.validationFn && 
                        !currentStepContent.validationFn()
                      }
                      className="flex items-center space-x-2"
                    >
                      <span>Next</span>
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>
      </AnimatePresence>

      {/* Achievement Notification */}
      <AnimatePresence>
        {newAchievement && (
          <motion.div
            initial={{ opacity: 0, y: -100, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -100, scale: 0.9 }}
            className="fixed top-4 right-4 z-[60] w-80"
          >
            <Card className="bg-gradient-to-r from-yellow-50 to-orange-50 border-yellow-200 shadow-lg">
              <CardContent className="p-4">
                <div className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    <div className="w-10 h-10 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-full flex items-center justify-center">
                      <Award className="h-5 w-5 text-white" />
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900">
                      Achievement Unlocked!
                    </p>
                    <p className="text-sm text-gray-600">{newAchievement.title}</p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setNewAchievement(null)}
                    className="flex-shrink-0"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </TooltipProvider>
  );
};

// Helper function to get basic description based on automata type
function getBasicDescription(type: AutomataType): string {
  switch (type) {
    case 'dfa':
      return 'A Deterministic Finite Automaton (DFA) is a computational model that processes input strings character by character. It has states, transitions, and exactly one path for each input symbol from every state.';
    case 'nfa':
      return 'A Non-deterministic Finite Automaton (NFA) is similar to a DFA but allows multiple transitions for the same input symbol and ε-transitions that don\'t consume input.';
    case 'pda':
      return 'A Pushdown Automaton (PDA) extends finite automata with a stack memory, allowing it to recognize context-free languages like balanced parentheses.';
    case 'tm':
      return 'A Turing Machine (TM) is the most powerful computational model, with an infinite tape that can be read from and written to, capable of computing anything that is computable.';
    case 'cfg':
      return 'A Context-Free Grammar (CFG) defines a context-free language through production rules that generate strings by replacing non-terminals with terminals and other non-terminals.';
    default:
      return 'This automaton model is a formal computational system used to recognize patterns in strings and solve computational problems.';
  }
}

// Demo Components (simplified for brevity - these would be full interactive components)
const BasicConceptDemo: React.FC<{ automataType: AutomataType }> = ({ automataType }) => (
  <div className="space-y-4">
    <div className="flex items-center justify-center p-8 bg-white dark:bg-gray-800 rounded-lg border">
      <div className="text-center space-y-2">
        <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center mx-auto">
          <span className="text-white font-bold">q0</span>
        </div>
        <p className="text-sm text-gray-600">Start State</p>
      </div>
      <ArrowRight className="h-6 w-6 mx-4 text-gray-400" />
      <div className="text-center space-y-2">
        <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto border-4 border-green-300">
          <span className="text-white font-bold">q1</span>
        </div>
        <p className="text-sm text-gray-600">Accept State</p>
      </div>
    </div>
  </div>
);

const StateCreationDemo: React.FC<{ onInteract: () => void }> = ({ onInteract }) => (
  <div className="space-y-4">
    <p className="text-sm text-gray-600 text-center">Click anywhere below to create a state</p>
    <div 
      className="h-40 bg-gray-50 dark:bg-gray-800 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 flex items-center justify-center cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
      onClick={onInteract}
    >
      <div className="text-center text-gray-500">
        <Target className="h-8 w-8 mx-auto mb-2" />
        <p className="text-sm">Click to create your first state</p>
      </div>
    </div>
  </div>
);

const TransitionDemo: React.FC<{ automataType: AutomataType }> = ({ automataType }) => (
  <div className="space-y-4">
    <div className="flex items-center justify-center p-6 bg-white dark:bg-gray-800 rounded-lg border">
      <div className="flex items-center space-x-8">
        <div className="text-center">
          <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center">
            <span className="text-white text-sm font-bold">q0</span>
          </div>
        </div>
        <div className="flex flex-col items-center">
          <ArrowRight className="h-6 w-6 text-gray-400" />
          <Badge variant="outline" className="mt-1 text-xs">a</Badge>
        </div>
        <div className="text-center">
          <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center border-2 border-green-300">
            <span className="text-white text-sm font-bold">q1</span>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const StackDemo: React.FC = () => (
  <div className="space-y-4">
    <div className="flex items-center justify-center space-x-8">
      <div className="space-y-2">
        <h4 className="text-sm font-medium text-center">Stack</h4>
        <div className="w-16 border-2 border-gray-300 rounded">
          {['c', 'b', 'a'].map((symbol, index) => (
            <div key={index} className="h-8 border-b border-gray-200 flex items-center justify-center bg-purple-100 dark:bg-purple-900/20">
              <span className="text-sm font-mono">{symbol}</span>
            </div>
          ))}
        </div>
        <p className="text-xs text-center text-gray-500">Top</p>
      </div>
      <div className="space-y-2 text-center">
        <Coffee className="h-8 w-8 mx-auto text-purple-500" />
        <p className="text-sm">Stack operations:<br/>Push, Pop, Peek</p>
      </div>
    </div>
  </div>
);

const TapeDemo: React.FC = () => (
  <div className="space-y-4">
    <div className="space-y-2">
      <h4 className="text-sm font-medium text-center">Turing Machine Tape</h4>
      <div className="flex justify-center">
        <div className="flex space-x-1">
          {['...', 'a', 'b', 'c', '□', '...'].map((symbol, index) => (
            <div 
              key={index} 
              className={cn(
                "w-10 h-10 border-2 rounded flex items-center justify-center font-mono text-sm",
                index === 3 ? "border-red-500 bg-red-100 dark:bg-red-900/20" : "border-gray-300"
              )}
            >
              {symbol}
            </div>
          ))}
        </div>
      </div>
      <div className="text-center">
        <div className="inline-flex items-center space-x-1 text-xs text-red-600">
          <span>↑</span>
          <span>Read/Write Head</span>
        </div>
      </div>
    </div>
  </div>
);

export default OnboardingFlow;