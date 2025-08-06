import { useState, useCallback, useEffect } from 'react';
import { 
  Achievement, 
  UserProgress, 
  TutorialState, 
  OnboardingStep 
} from '../types/animation';
import { AutomataType } from '../types/automata';

export interface UseOnboardingProps {
  automataType: AutomataType;
  onAchievementUnlocked?: (achievement: Achievement) => void;
  onProgressUpdate?: (progress: UserProgress) => void;
  persistProgress?: boolean;
}

export interface UseOnboardingReturn {
  // State
  isOnboardingActive: boolean;
  currentStep: number;
  tutorialState: TutorialState;
  userProgress: UserProgress;
  achievements: Achievement[];
  
  // Controls
  startOnboarding: () => void;
  completeOnboarding: () => void;
  skipOnboarding: () => void;
  nextStep: () => void;
  previousStep: () => void;
  goToStep: (step: number) => void;
  
  // Progress tracking
  markStepComplete: (stepId: string) => void;
  markStepSkipped: (stepId: string) => void;
  updateProgress: (updates: Partial<UserProgress>) => void;
  unlockAchievement: (achievementId: string) => void;
  
  // Utilities
  getCompletionRate: () => number;
  shouldShowOnboarding: () => boolean;
  resetProgress: () => void;
  exportProgress: () => UserProgress;
}

const STORAGE_KEY = 'automata-onboarding-progress';

const DEFAULT_USER_PROGRESS: UserProgress = {
  hasCreatedAutomaton: false,
  hasRunSimulation: false,
  hasUsedAI: false,
  totalAutomataCreated: 0,
  successfulSimulations: 0,
  completedTutorials: [],
  achievementsUnlocked: [],
  totalTimeSpent: 0,
  lastActiveDate: new Date()
};

const DEFAULT_ACHIEVEMENTS: Achievement[] = [
  {
    id: 'first_steps',
    title: 'First Steps',
    description: 'Complete your first tutorial',
    icon: '🎯',
    unlocked: false,
    progress: 0,
    maxProgress: 1,
    category: 'tutorial'
  },
  {
    id: 'automaton_creator',
    title: 'Automaton Creator',
    description: 'Create your first automaton',
    icon: '⚙️',
    unlocked: false,
    progress: 0,
    maxProgress: 1,
    category: 'creation'
  },
  {
    id: 'simulation_novice',
    title: 'Simulation Novice',
    description: 'Run your first simulation',
    icon: '▶️',
    unlocked: false,
    progress: 0,
    maxProgress: 1,
    category: 'simulation'
  },
  {
    id: 'simulation_master',
    title: 'Simulation Master',
    description: 'Run 10 successful simulations',
    icon: '🏆',
    unlocked: false,
    progress: 0,
    maxProgress: 10,
    category: 'simulation'
  },
  {
    id: 'ai_explorer',
    title: 'AI Explorer',
    description: 'Use AI assistance for the first time',
    icon: '🤖',
    unlocked: false,
    progress: 0,
    maxProgress: 1,
    category: 'advanced'
  },
  {
    id: 'multi_type_master',
    title: 'Multi-Type Master',
    description: 'Work with 3 different automata types',
    icon: '🎭',
    unlocked: false,
    progress: 0,
    maxProgress: 3,
    category: 'advanced'
  },
  {
    id: 'perfectionist',
    title: 'Perfectionist',
    description: 'Create 5 automata without errors',
    icon: '✨',
    unlocked: false,
    progress: 0,
    maxProgress: 5,
    category: 'creation'
  },
  {
    id: 'speed_demon',
    title: 'Speed Demon',
    description: 'Complete a tutorial in under 2 minutes',
    icon: '⚡',
    unlocked: false,
    progress: 0,
    maxProgress: 1,
    category: 'tutorial'
  }
];

export const useOnboarding = ({
  automataType,
  onAchievementUnlocked,
  onProgressUpdate,
  persistProgress = true
}: UseOnboardingProps): UseOnboardingReturn => {
  // Load initial state from storage
  const loadProgressFromStorage = useCallback((): UserProgress => {
    if (!persistProgress) return DEFAULT_USER_PROGRESS;
    
    try {
      const stored = localStorage.getItem(`${STORAGE_KEY}-${automataType}`);
      return stored ? { ...DEFAULT_USER_PROGRESS, ...JSON.parse(stored) } : DEFAULT_USER_PROGRESS;
    } catch {
      return DEFAULT_USER_PROGRESS;
    }
  }, [automataType, persistProgress]);

  const loadAchievementsFromStorage = useCallback((): Achievement[] => {
    if (!persistProgress) return DEFAULT_ACHIEVEMENTS;
    
    try {
      const stored = localStorage.getItem(`${STORAGE_KEY}-achievements`);
      const storedAchievements = stored ? JSON.parse(stored) : {};
      
      return DEFAULT_ACHIEVEMENTS.map(achievement => ({
        ...achievement,
        ...(storedAchievements[achievement.id] || {})
      }));
    } catch {
      return DEFAULT_ACHIEVEMENTS;
    }
  }, [persistProgress]);

  // State
  const [isOnboardingActive, setIsOnboardingActive] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [tutorialState, setTutorialState] = useState<TutorialState>({
    isActive: false,
    currentStep: 0,
    completedSteps: new Set(),
    skippedSteps: new Set(),
    startTime: new Date(),
    totalDuration: 0
  });
  const [userProgress, setUserProgress] = useState<UserProgress>(loadProgressFromStorage);
  const [achievements, setAchievements] = useState<Achievement[]>(loadAchievementsFromStorage);

  // Save to storage
  const saveProgressToStorage = useCallback((progress: UserProgress) => {
    if (!persistProgress) return;
    
    try {
      localStorage.setItem(`${STORAGE_KEY}-${automataType}`, JSON.stringify(progress));
    } catch (error) {
      console.warn('Failed to save onboarding progress:', error);
    }
  }, [automataType, persistProgress]);

  const saveAchievementsToStorage = useCallback((achievements: Achievement[]) => {
    if (!persistProgress) return;
    
    try {
      const achievementData = achievements.reduce((acc, achievement) => {
        acc[achievement.id] = {
          unlocked: achievement.unlocked,
          progress: achievement.progress,
          unlockedAt: achievement.unlockedAt
        };
        return acc;
      }, {} as any);
      
      localStorage.setItem(`${STORAGE_KEY}-achievements`, JSON.stringify(achievementData));
    } catch (error) {
      console.warn('Failed to save achievements:', error);
    }
  }, [persistProgress]);

  // Control functions
  const startOnboarding = useCallback(() => {
    setIsOnboardingActive(true);
    setTutorialState(prev => ({
      ...prev,
      isActive: true,
      startTime: new Date()
    }));
  }, []);

  const completeOnboarding = useCallback(() => {
    setIsOnboardingActive(false);
    setTutorialState(prev => {
      const endTime = new Date();
      const duration = endTime.getTime() - prev.startTime.getTime();
      
      return {
        ...prev,
        isActive: false,
        totalDuration: duration
      };
    });

    // Update progress
    const newProgress = {
      ...userProgress,
      completedTutorials: [...userProgress.completedTutorials, automataType],
      totalTimeSpent: userProgress.totalTimeSpent + tutorialState.totalDuration,
      lastActiveDate: new Date()
    };

    setUserProgress(newProgress);
    saveProgressToStorage(newProgress);
    onProgressUpdate?.(newProgress);

    // Unlock achievement
    unlockAchievement('first_steps');
  }, [userProgress, tutorialState.totalDuration, automataType, saveProgressToStorage, onProgressUpdate]);

  const skipOnboarding = useCallback(() => {
    setIsOnboardingActive(false);
    setTutorialState(prev => ({
      ...prev,
      isActive: false,
      skippedSteps: new Set([...prev.skippedSteps, `tutorial-${automataType}`])
    }));
  }, [automataType]);

  const nextStep = useCallback(() => {
    setCurrentStep(prev => prev + 1);
    setTutorialState(prev => ({
      ...prev,
      currentStep: prev.currentStep + 1
    }));
  }, []);

  const previousStep = useCallback(() => {
    setCurrentStep(prev => Math.max(0, prev - 1));
    setTutorialState(prev => ({
      ...prev,
      currentStep: Math.max(0, prev.currentStep - 1)
    }));
  }, []);

  const goToStep = useCallback((step: number) => {
    setCurrentStep(step);
    setTutorialState(prev => ({
      ...prev,
      currentStep: step
    }));
  }, []);

  // Progress tracking
  const markStepComplete = useCallback((stepId: string) => {
    setTutorialState(prev => ({
      ...prev,
      completedSteps: new Set([...prev.completedSteps, stepId])
    }));
  }, []);

  const markStepSkipped = useCallback((stepId: string) => {
    setTutorialState(prev => ({
      ...prev,
      skippedSteps: new Set([...prev.skippedSteps, stepId])
    }));
  }, []);

  const updateProgress = useCallback((updates: Partial<UserProgress>) => {
    const newProgress = { 
      ...userProgress, 
      ...updates, 
      lastActiveDate: new Date() 
    };
    
    setUserProgress(newProgress);
    saveProgressToStorage(newProgress);
    onProgressUpdate?.(newProgress);

    // Check for achievement unlocks based on progress
    if (updates.hasCreatedAutomaton && !userProgress.hasCreatedAutomaton) {
      unlockAchievement('automaton_creator');
    }
    
    if (updates.hasRunSimulation && !userProgress.hasRunSimulation) {
      unlockAchievement('simulation_novice');
    }
    
    if (updates.hasUsedAI && !userProgress.hasUsedAI) {
      unlockAchievement('ai_explorer');
    }
    
    if (updates.successfulSimulations === 10) {
      unlockAchievement('simulation_master');
    }
  }, [userProgress, saveProgressToStorage, onProgressUpdate]);

  const unlockAchievement = useCallback((achievementId: string) => {
    setAchievements(prev => {
      const newAchievements = prev.map(achievement => {
        if (achievement.id === achievementId && !achievement.unlocked) {
          const unlockedAchievement = {
            ...achievement,
            unlocked: true,
            progress: achievement.maxProgress,
            unlockedAt: new Date()
          };
          
          onAchievementUnlocked?.(unlockedAchievement);
          return unlockedAchievement;
        }
        return achievement;
      });
      
      saveAchievementsToStorage(newAchievements);
      return newAchievements;
    });

    // Update user progress
    const newProgress = {
      ...userProgress,
      achievementsUnlocked: [...userProgress.achievementsUnlocked, achievementId]
    };
    
    setUserProgress(newProgress);
    saveProgressToStorage(newProgress);
  }, [userProgress, saveProgressToStorage, saveAchievementsToStorage, onAchievementUnlocked]);

  // Utility functions
  const getCompletionRate = useCallback(() => {
    const totalSteps = tutorialState.completedSteps.size + tutorialState.skippedSteps.size;
    return totalSteps > 0 ? (tutorialState.completedSteps.size / totalSteps) * 100 : 0;
  }, [tutorialState]);

  const shouldShowOnboarding = useCallback(() => {
    return !userProgress.completedTutorials.includes(automataType);
  }, [userProgress.completedTutorials, automataType]);

  const resetProgress = useCallback(() => {
    setUserProgress(DEFAULT_USER_PROGRESS);
    setAchievements(DEFAULT_ACHIEVEMENTS);
    setTutorialState({
      isActive: false,
      currentStep: 0,
      completedSteps: new Set(),
      skippedSteps: new Set(),
      startTime: new Date(),
      totalDuration: 0
    });
    
    if (persistProgress) {
      try {
        localStorage.removeItem(`${STORAGE_KEY}-${automataType}`);
        localStorage.removeItem(`${STORAGE_KEY}-achievements`);
      } catch (error) {
        console.warn('Failed to clear onboarding progress:', error);
      }
    }
  }, [automataType, persistProgress]);

  const exportProgress = useCallback(() => {
    return userProgress;
  }, [userProgress]);

  return {
    // State
    isOnboardingActive,
    currentStep,
    tutorialState,
    userProgress,
    achievements,
    
    // Controls
    startOnboarding,
    completeOnboarding,
    skipOnboarding,
    nextStep,
    previousStep,
    goToStep,
    
    // Progress tracking
    markStepComplete,
    markStepSkipped,
    updateProgress,
    unlockAchievement,
    
    // Utilities
    getCompletionRate,
    shouldShowOnboarding,
    resetProgress,
    exportProgress
  };
};