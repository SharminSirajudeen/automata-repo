import { useState, useCallback, useEffect, useRef } from 'react';
import { useSpring, config } from '@react-spring/web';
import { 
  AnimationConfig, 
  AnimationStep, 
  AnimationSettings, 
  AnimationEvent,
  AnimationMetrics,
  DEFAULT_ANIMATION_PRESETS 
} from '../types/animation';
import { SimulationStep } from '../types/automata';

export interface UseAnimationSystemProps {
  simulationSteps: SimulationStep[];
  onStepChange?: (step: number) => void;
  onAnimationEvent?: (event: AnimationEvent) => void;
  initialConfig?: Partial<AnimationConfig>;
  initialSettings?: Partial<AnimationSettings>;
}

export interface UseAnimationSystemReturn {
  // State
  currentStep: number;
  isPlaying: boolean;
  isPaused: boolean;
  isComplete: boolean;
  animationConfig: AnimationConfig;
  animationSettings: AnimationSettings;
  metrics: AnimationMetrics;
  
  // Controls
  play: () => void;
  pause: () => void;
  stop: () => void;
  reset: () => void;
  stepForward: () => void;
  stepBackward: () => void;
  seekTo: (step: number) => void;
  
  // Configuration
  updateConfig: (config: Partial<AnimationConfig>) => void;
  updateSettings: (settings: Partial<AnimationSettings>) => void;
  applyPreset: (presetName: string) => void;
  
  // Animation springs
  progressSpring: any;
  stepSpring: any;
  
  // Utilities
  getStepProgress: () => number;
  getTimeRemaining: () => number;
  exportMetrics: () => AnimationMetrics;
}

export const useAnimationSystem = ({
  simulationSteps,
  onStepChange,
  onAnimationEvent,
  initialConfig,
  initialSettings
}: UseAnimationSystemProps): UseAnimationSystemReturn => {
  // Default configuration
  const defaultConfig: AnimationConfig = {
    duration: 800,
    easing: 'wobbly',
    stagger: 100,
    showTrails: true,
    highlightIntensity: 1,
    ...initialConfig
  };

  const defaultSettings: AnimationSettings = {
    autoPlay: false,
    loop: false,
    showDetails: true,
    highlightPath: true,
    animationSpeed: 1,
    pauseOnError: true,
    ...initialSettings
  };

  // State
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [animationConfig, setAnimationConfig] = useState<AnimationConfig>(defaultConfig);
  const [animationSettings, setAnimationSettings] = useState<AnimationSettings>(defaultSettings);
  const [startTime, setStartTime] = useState<Date | null>(null);
  const [metrics, setMetrics] = useState<AnimationMetrics>({
    totalSteps: simulationSteps.length,
    currentStep: 0,
    completionRate: 0,
    averageStepDuration: 0,
    totalAnimationTime: 0,
    userInteractions: 0,
    pauseCount: 0,
    skipCount: 0
  });

  // Refs for cleanup
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const stepTimestamps = useRef<number[]>([]);
  const animationFrameRef = useRef<number | null>(null);
  const isUnmountedRef = useRef(false);

  // Animation springs
  const progressSpring = useSpring({
    progress: currentStep / Math.max(1, simulationSteps.length - 1),
    config: config[animationConfig.easing]
  });

  const stepSpring = useSpring({
    step: currentStep,
    config: config[animationConfig.easing]
  });

  // Computed values
  const isComplete = currentStep >= simulationSteps.length - 1;

  // Emit animation events with safety check
  const emitEvent = useCallback((event: AnimationEvent) => {
    if (!isUnmountedRef.current) {
      onAnimationEvent?.(event);
    }
  }, [onAnimationEvent]);

  // Update metrics with safety check
  const updateMetrics = useCallback((updates: Partial<AnimationMetrics>) => {
    if (!isUnmountedRef.current) {
      setMetrics(prev => ({ ...prev, ...updates }));
    }
  }, []);

  // Control functions
  const play = useCallback(() => {
    if (isComplete && !animationSettings.loop) return;
    
    setIsPlaying(true);
    setIsPaused(false);
    
    if (!startTime) {
      setStartTime(new Date());
    }
    
    emitEvent({ type: 'start', timestamp: Date.now(), step: currentStep });
    updateMetrics({ userInteractions: metrics.userInteractions + 1 });
  }, [isComplete, animationSettings.loop, startTime, currentStep, metrics.userInteractions, emitEvent, updateMetrics]);

  const pause = useCallback(() => {
    setIsPlaying(false);
    setIsPaused(true);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    emitEvent({ type: 'pause', timestamp: Date.now(), step: currentStep });
    updateMetrics({ 
      pauseCount: metrics.pauseCount + 1,
      userInteractions: metrics.userInteractions + 1 
    });
  }, [currentStep, metrics.pauseCount, metrics.userInteractions, emitEvent, updateMetrics]);

  const stop = useCallback(() => {
    setIsPlaying(false);
    setIsPaused(false);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    emitEvent({ type: 'stop', timestamp: Date.now(), step: currentStep });
    updateMetrics({ userInteractions: metrics.userInteractions + 1 });
  }, [currentStep, metrics.userInteractions, emitEvent, updateMetrics]);

  const reset = useCallback(() => {
    if (isUnmountedRef.current) return;
    
    // Clear all timers and animations
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    setCurrentStep(0);
    setIsPlaying(false);
    setIsPaused(false);
    setStartTime(null);
    
    // Clear timestamps array instead of reassigning
    stepTimestamps.current.length = 0;
    
    onStepChange?.(0);
    emitEvent({ type: 'reset', timestamp: Date.now() });
    updateMetrics({ 
      currentStep: 0,
      completionRate: 0,
      userInteractions: metrics.userInteractions + 1 
    });
  }, [onStepChange, metrics.userInteractions, emitEvent, updateMetrics]);

  const stepForward = useCallback(() => {
    if (isUnmountedRef.current) return;
    
    if (currentStep < simulationSteps.length - 1) {
      const newStep = currentStep + 1;
      setCurrentStep(newStep);
      
      // Prevent unbounded array growth
      if (stepTimestamps.current.length < 1000) {
        stepTimestamps.current[newStep] = Date.now();
      }
      
      onStepChange?.(newStep);
      emitEvent({ type: 'step', timestamp: Date.now(), step: newStep });
      
      updateMetrics({ 
        currentStep: newStep,
        completionRate: (newStep / (simulationSteps.length - 1)) * 100,
        userInteractions: metrics.userInteractions + 1 
      });
    } else if (animationSettings.loop && !isUnmountedRef.current) {
      reset();
      if (!isUnmountedRef.current) {
        play();
      }
    }
  }, [currentStep, simulationSteps.length, animationSettings.loop, onStepChange, metrics.userInteractions, emitEvent, updateMetrics, reset, play]);

  const stepBackward = useCallback(() => {
    if (currentStep > 0) {
      const newStep = currentStep - 1;
      setCurrentStep(newStep);
      
      onStepChange?.(newStep);
      emitEvent({ type: 'step', timestamp: Date.now(), step: newStep });
      
      updateMetrics({ 
        currentStep: newStep,
        completionRate: (newStep / (simulationSteps.length - 1)) * 100,
        userInteractions: metrics.userInteractions + 1 
      });
    }
  }, [currentStep, onStepChange, metrics.userInteractions, emitEvent, updateMetrics]);

  const seekTo = useCallback((step: number) => {
    if (step >= 0 && step < simulationSteps.length) {
      setCurrentStep(step);
      stepTimestamps.current[step] = Date.now();
      
      onStepChange?.(step);
      emitEvent({ type: 'step', timestamp: Date.now(), step });
      
      updateMetrics({ 
        currentStep: step,
        completionRate: (step / (simulationSteps.length - 1)) * 100,
        userInteractions: metrics.userInteractions + 1 
      });
    }
  }, [simulationSteps.length, onStepChange, metrics.userInteractions, emitEvent, updateMetrics]);

  // Configuration functions
  const updateConfig = useCallback((config: Partial<AnimationConfig>) => {
    setAnimationConfig(prev => ({ ...prev, ...config }));
  }, []);

  const updateSettings = useCallback((settings: Partial<AnimationSettings>) => {
    setAnimationSettings(prev => ({ ...prev, ...settings }));
  }, []);

  const applyPreset = useCallback((presetName: string) => {
    const preset = DEFAULT_ANIMATION_PRESETS.find(p => p.name === presetName);
    if (preset) {
      setAnimationConfig(preset.config);
      setAnimationSettings(preset.settings);
    }
  }, []);

  // Utility functions
  const getStepProgress = useCallback(() => {
    return currentStep / Math.max(1, simulationSteps.length - 1);
  }, [currentStep, simulationSteps.length]);

  const getTimeRemaining = useCallback(() => {
    const remainingSteps = simulationSteps.length - 1 - currentStep;
    const avgDuration = metrics.averageStepDuration || animationConfig.duration;
    return remainingSteps * avgDuration * (1 / animationSettings.animationSpeed);
  }, [simulationSteps.length, currentStep, metrics.averageStepDuration, animationConfig.duration, animationSettings.animationSpeed]);

  const exportMetrics = useCallback(() => {
    if (isUnmountedRef.current) return metrics;
    
    const now = Date.now();
    const totalTime = startTime ? now - startTime.getTime() : 0;
    const validTimestamps = stepTimestamps.current.filter(t => t > 0);
    const averageDuration = validTimestamps.length > 1 
      ? (validTimestamps[validTimestamps.length - 1] - validTimestamps[0]) / (validTimestamps.length - 1)
      : 0;

    return {
      ...metrics,
      totalAnimationTime: totalTime,
      averageStepDuration: Math.max(0, averageDuration)
    };
  }, [metrics, startTime]);

  // Auto-play effect with proper cleanup
  useEffect(() => {
    if (isPlaying && !isComplete && !isUnmountedRef.current) {
      const effectiveDuration = Math.max(100, animationConfig.duration / animationSettings.animationSpeed);
      
      intervalRef.current = setInterval(() => {
        if (!isUnmountedRef.current) {
          stepForward();
        }
      }, effectiveDuration);

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      };
    }
  }, [isPlaying, isComplete, animationConfig.duration, animationSettings.animationSpeed, stepForward]);

  // Update metrics when steps change
  useEffect(() => {
    updateMetrics({ 
      totalSteps: simulationSteps.length,
      completionRate: simulationSteps.length > 0 ? (currentStep / (simulationSteps.length - 1)) * 100 : 0
    });
  }, [simulationSteps.length, currentStep, updateMetrics]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isUnmountedRef.current = true;
      
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      
      // Clear arrays to prevent memory leaks
      stepTimestamps.current.length = 0;
    };
  }, []);

  return {
    // State
    currentStep,
    isPlaying,
    isPaused,
    isComplete,
    animationConfig,
    animationSettings,
    metrics: exportMetrics(),
    
    // Controls
    play,
    pause,
    stop,
    reset,
    stepForward,
    stepBackward,
    seekTo,
    
    // Configuration
    updateConfig,
    updateSettings,
    applyPreset,
    
    // Animation springs
    progressSpring,
    stepSpring,
    
    // Utilities
    getStepProgress,
    getTimeRemaining,
    exportMetrics
  };
};