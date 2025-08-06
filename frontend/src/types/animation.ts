import { config } from '@react-spring/web';

export interface AnimationConfig {
  duration: number;
  easing: keyof typeof config;
  stagger: number;
  showTrails: boolean;
  highlightIntensity: number;
}

export interface AnimationStep {
  id: string;
  type: 'state_transition' | 'input_read' | 'stack_operation' | 'tape_operation' | 'production_apply';
  timestamp: number;
  data: any;
  description: string;
}

export interface TapeCell {
  symbol: string;
  index: number;
  isHead: boolean;
  isHighlighted: boolean;
}

export interface StackFrame {
  symbol: string;
  index: number;
  isTop: boolean;
  operation: 'push' | 'pop' | 'none';
}

export interface ParseTreeNode {
  id: string;
  symbol: string;
  parent?: string;
  children: string[];
  level: number;
  isActive: boolean;
  production?: string;
}

export interface StateHighlight {
  stateId: string;
  intensity: number;
  color: 'blue' | 'green' | 'red' | 'yellow' | 'purple';
  animation: 'pulse' | 'glow' | 'bounce' | 'none';
}

export interface TransitionAnimation {
  fromState: string;
  toState: string;
  symbol: string;
  progress: number;
  isActive: boolean;
}

export interface AnimationEvent {
  type: 'start' | 'step' | 'pause' | 'stop' | 'reset';
  timestamp: number;
  step?: number;
  data?: any;
}

export interface AnimationSettings {
  autoPlay: boolean;
  loop: boolean;
  showDetails: boolean;
  highlightPath: boolean;
  animationSpeed: number;
  pauseOnError: boolean;
}

export interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  target?: string;
  position: 'top' | 'bottom' | 'left' | 'right' | 'center';
  canSkip: boolean;
  actionRequired?: boolean;
  validationFn?: () => boolean;
  content: React.ReactNode;
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
  unlockedAt?: Date;
}

export interface UserProgress {
  hasCreatedAutomaton: boolean;
  hasRunSimulation: boolean;
  hasUsedAI: boolean;
  totalAutomataCreated: number;
  successfulSimulations: number;
  completedTutorials: string[];
  achievementsUnlocked: string[];
  totalTimeSpent: number;
  lastActiveDate: Date;
}

export interface TutorialState {
  isActive: boolean;
  currentStep: number;
  completedSteps: Set<string>;
  skippedSteps: Set<string>;
  startTime: Date;
  totalDuration: number;
}

export interface InteractiveElement {
  id: string;
  type: 'tooltip' | 'highlight' | 'modal' | 'popover';
  target: string;
  content: string | React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
  trigger?: 'hover' | 'click' | 'focus';
  delay?: number;
  duration?: number;
}

export interface AnimationPreset {
  name: string;
  description: string;
  config: AnimationConfig;
  settings: AnimationSettings;
}

export const DEFAULT_ANIMATION_PRESETS: AnimationPreset[] = [
  {
    name: 'Smooth',
    description: 'Smooth and gentle animations',
    config: {
      duration: 800,
      easing: 'gentle',
      stagger: 100,
      showTrails: true,
      highlightIntensity: 0.8
    },
    settings: {
      autoPlay: false,
      loop: false,
      showDetails: true,
      highlightPath: true,
      animationSpeed: 1,
      pauseOnError: true
    }
  },
  {
    name: 'Fast',
    description: 'Quick animations for experienced users',
    config: {
      duration: 300,
      easing: 'wobbly',
      stagger: 50,
      showTrails: false,
      highlightIntensity: 1
    },
    settings: {
      autoPlay: true,
      loop: false,
      showDetails: false,
      highlightPath: true,
      animationSpeed: 2,
      pauseOnError: false
    }
  },
  {
    name: 'Educational',
    description: 'Detailed animations with explanations',
    config: {
      duration: 1200,
      easing: 'gentle',
      stagger: 200,
      showTrails: true,
      highlightIntensity: 1.2
    },
    settings: {
      autoPlay: false,
      loop: false,
      showDetails: true,
      highlightPath: true,
      animationSpeed: 0.7,
      pauseOnError: true
    }
  }
];

export interface AnimationMetrics {
  totalSteps: number;
  currentStep: number;
  completionRate: number;
  averageStepDuration: number;
  totalAnimationTime: number;
  userInteractions: number;
  pauseCount: number;
  skipCount: number;
}