import { useCallback, useEffect, useRef, useState, useMemo } from 'react';
import { useSpring, config } from '@react-spring/web';

export interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  memoryUsage?: number;
  animationLoad: number;
  droppedFrames: number;
}

export interface PerformanceSettings {
  maxFPS: number;
  adaptiveQuality: boolean;
  reduceMotion: boolean;
  enableGPUAcceleration: boolean;
  memoryThreshold: number;
}

export const useAnimationPerformance = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 60,
    frameTime: 16.67,
    animationLoad: 0,
    droppedFrames: 0
  });
  
  const [settings, setSettings] = useState<PerformanceSettings>({
    maxFPS: 60,
    adaptiveQuality: true,
    reduceMotion: false,
    enableGPUAcceleration: true,
    memoryThreshold: 100 // MB
  });

  const frameCount = useRef(0);
  const lastFrameTime = useRef(performance.now());
  const fpsBuffer = useRef<number[]>([]);
  const animationFrameId = useRef<number>();

  // Detect user preference for reduced motion
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    
    const handleChange = (e: MediaQueryListEvent) => {
      setSettings(prev => ({ ...prev, reduceMotion: e.matches }));
    };

    setSettings(prev => ({ ...prev, reduceMotion: mediaQuery.matches }));
    mediaQuery.addEventListener('change', handleChange);

    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Performance monitoring loop
  useEffect(() => {
    const measurePerformance = () => {
      const now = performance.now();
      const deltaTime = now - lastFrameTime.current;
      
      frameCount.current++;
      fpsBuffer.current.push(1000 / deltaTime);
      
      // Keep only the last 60 frames for averaging
      if (fpsBuffer.current.length > 60) {
        fpsBuffer.current.shift();
      }
      
      // Update metrics every 30 frames
      if (frameCount.current % 30 === 0) {
        const avgFPS = fpsBuffer.current.reduce((sum, fps) => sum + fps, 0) / fpsBuffer.current.length;
        
        setMetrics(prev => ({
          ...prev,
          fps: Math.round(avgFPS),
          frameTime: deltaTime,
          animationLoad: Math.min((60 / avgFPS) * 100, 100), // Percentage of ideal performance
          droppedFrames: prev.droppedFrames + (deltaTime > 16.67 ? 1 : 0)
        }));

        // Adaptive quality adjustment
        if (settings.adaptiveQuality) {
          if (avgFPS < 30 && !settings.reduceMotion) {
            // Performance is poor, reduce quality
            setSettings(prev => ({ ...prev, reduceMotion: true }));
          } else if (avgFPS > 55 && settings.reduceMotion) {
            // Performance is good, restore quality
            setSettings(prev => ({ ...prev, reduceMotion: false }));
          }
        }
      }
      
      lastFrameTime.current = now;
      animationFrameId.current = requestAnimationFrame(measurePerformance);
    };

    animationFrameId.current = requestAnimationFrame(measurePerformance);

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [settings.adaptiveQuality, settings.reduceMotion]);

  // Memory usage monitoring (if supported)
  useEffect(() => {
    const updateMemoryUsage = () => {
      if ('memory' in performance) {
        const memInfo = (performance as any).memory;
        const memoryUsage = Math.round(memInfo.usedJSHeapSize / (1024 * 1024)); // MB
        
        setMetrics(prev => ({
          ...prev,
          memoryUsage
        }));
      }
    };

    const interval = setInterval(updateMemoryUsage, 5000); // Every 5 seconds
    updateMemoryUsage(); // Initial check

    return () => clearInterval(interval);
  }, []);

  // Optimized animation config based on performance
  const optimizedConfig = useMemo(() => {
    if (settings.reduceMotion) {
      return {
        ...config.gentle,
        duration: 100, // Much faster animations
        tension: 300,
        friction: 30
      };
    }

    if (metrics.fps < 30) {
      return {
        ...config.slow,
        duration: Math.max(200, config.slow.duration),
        tension: Math.min(150, config.slow.tension)
      };
    }

    if (metrics.fps < 45) {
      return config.gentle;
    }

    return config.wobbly; // High performance, use smooth animations
  }, [settings.reduceMotion, metrics.fps]);

  const shouldSkipAnimation = useCallback((animationType: 'transition' | 'gesture' | 'decoration') => {
    if (settings.reduceMotion) {
      return animationType === 'decoration'; // Skip decorative animations only
    }

    if (metrics.fps < 20) {
      return true; // Skip all animations if performance is very poor
    }

    if (metrics.fps < 30 && animationType === 'decoration') {
      return true; // Skip decorative animations if performance is poor
    }

    return false;
  }, [settings.reduceMotion, metrics.fps]);

  const getOptimizedStagger = useCallback((itemCount: number, baseStagger: number) => {
    if (settings.reduceMotion) return 0;
    
    if (itemCount > 20 && metrics.fps < 45) {
      return Math.max(10, baseStagger * 0.3); // Reduce stagger for many items
    }

    if (metrics.fps < 30) {
      return Math.max(20, baseStagger * 0.5);
    }

    return baseStagger;
  }, [settings.reduceMotion, metrics.fps]);

  const updateSettings = useCallback((newSettings: Partial<PerformanceSettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  }, []);

  const resetMetrics = useCallback(() => {
    setMetrics({
      fps: 60,
      frameTime: 16.67,
      animationLoad: 0,
      droppedFrames: 0,
      memoryUsage: metrics.memoryUsage
    });
    frameCount.current = 0;
    fpsBuffer.current = [];
  }, [metrics.memoryUsage]);

  // Performance status indicator
  const performanceStatus = useMemo(() => {
    if (metrics.fps >= 50) return 'excellent';
    if (metrics.fps >= 30) return 'good';
    if (metrics.fps >= 20) return 'poor';
    return 'critical';
  }, [metrics.fps]);

  return {
    metrics,
    settings,
    optimizedConfig,
    shouldSkipAnimation,
    getOptimizedStagger,
    updateSettings,
    resetMetrics,
    performanceStatus
  };
};

export default useAnimationPerformance;