import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { useSpring, useSpringValue, animated, useTransition, config } from '@react-spring/web';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { Badge } from './ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Separator } from './ui/separator';
import { Progress } from './ui/progress';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { 
  Play, 
  Pause, 
  Square, 
  SkipForward, 
  SkipBack, 
  RotateCcw,
  Settings,
  Zap,
  Activity,
  TreePine,
  Download,
  Camera,
  Video,
  Maximize2,
  Monitor,
  Smartphone,
  AlertCircle,
  CheckCircle2,
  XCircle
} from 'lucide-react';
import { 
  ExtendedAutomaton, 
  SimulationStep, 
  TMAutomaton, 
  PDAAutomaton, 
  CFGAutomaton,
  State,
  Transition,
  TMTransition,
  PDATransition,
  CFGProduction
} from '../types/automata';
import { cn } from '../lib/utils';

export interface AnimationConfig {
  duration: number;
  easing: keyof typeof config;
  stagger: number;
  showTrails: boolean;
  highlightIntensity: number;
  enableExport?: boolean;
  exportFormat?: 'gif' | 'webm' | 'mp4';
  exportQuality?: 'low' | 'medium' | 'high';
  exportFPS?: number;
}

export interface AnimationStep {
  id: string;
  type: 'state_transition' | 'input_read' | 'stack_operation' | 'tape_operation' | 'production_apply';
  timestamp: number;
  data: any;
  description: string;
}

interface AnimationSystemProps {
  automaton: ExtendedAutomaton;
  simulationSteps: SimulationStep[];
  currentStep: number;
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onStep: (direction: 'forward' | 'backward') => void;
  onSeek: (step: number) => void;
  onReset: () => void;
  className?: string;
  onError?: (error: Error) => void;
  showExportOptions?: boolean;
  compactMode?: boolean;
}

export const AnimationSystem: React.FC<AnimationSystemProps> = ({
  automaton,
  simulationSteps,
  currentStep,
  isPlaying,
  onPlay,
  onPause,
  onStop,
  onStep,
  onSeek,
  onReset,
  className,
  onError,
  showExportOptions = false,
  compactMode = false
}) => {
  const [animationConfig, setAnimationConfig] = useState<AnimationConfig>({
    duration: 800,
    easing: 'wobbly',
    stagger: 100,
    showTrails: true,
    highlightIntensity: 1,
    enableExport: showExportOptions,
    exportFormat: 'gif',
    exportQuality: 'medium',
    exportFPS: 30
  });

  const [showSettings, setShowSettings] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [viewMode, setViewMode] = useState<'desktop' | 'mobile'>('desktop');
  const [animationError, setAnimationError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrames = useRef<ImageData[]>([]);
  const animationProgress = useSpringValue(0);
  const currentStepData = simulationSteps[currentStep];

  // Error handling
  const handleError = useCallback((error: Error, context: string) => {
    const errorMessage = `Animation error in ${context}: ${error.message}`;
    setAnimationError(errorMessage);
    onError?.(error);
    console.error(errorMessage, error);
  }, [onError]);

  // Clear error after timeout
  useEffect(() => {
    if (animationError) {
      const timeout = setTimeout(() => setAnimationError(null), 5000);
      return () => clearTimeout(timeout);
    }
  }, [animationError]);

  // Export functionality
  const startRecording = useCallback(async () => {
    if (!canvasRef.current) return;
    
    try {
      setIsExporting(true);
      setExportProgress(0);
      animationFrames.current = [];
      
      // Reset animation to beginning
      onReset();
      
      // Capture frames for each step
      for (let i = 0; i < simulationSteps.length; i++) {
        await new Promise(resolve => setTimeout(resolve, animationConfig.duration));
        onSeek(i);
        
        // Capture frame
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (ctx) {
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          animationFrames.current.push(imageData);
        }
        
        setExportProgress((i + 1) / simulationSteps.length * 100);
      }
      
      await exportAnimation();
    } catch (error) {
      handleError(error as Error, 'recording');
    } finally {
      setIsExporting(false);
      setExportProgress(0);
    }
  }, [simulationSteps.length, animationConfig.duration, onReset, onSeek, handleError]);

  const exportAnimation = useCallback(async () => {
    try {
      if (animationFrames.current.length === 0) return;
      
      // Create a simple export (this would need a proper implementation with libraries like gif.js)
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      // For now, just export the current frame as PNG
      canvas.width = 800;
      canvas.height = 600;
      
      // Draw the last frame
      const lastFrame = animationFrames.current[animationFrames.current.length - 1];
      ctx.putImageData(lastFrame, 0, 0);
      
      // Download as PNG (simplified version)
      const link = document.createElement('a');
      link.download = `automata-animation-${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
    } catch (error) {
      handleError(error as Error, 'export');
    }
  }, [handleError]);

  // Responsive design helpers
  const isMobile = viewMode === 'mobile';
  const controlSize = isMobile ? 'sm' : 'default';
  const spacing = isMobile ? 'space-x-1' : 'space-x-2';

  // Main animation controller
  const mainSpring = useSpring({
    progress: currentStep / Math.max(1, simulationSteps.length - 1),
    config: config[animationConfig.easing],
    onRest: () => {
      try {
        if (isPlaying && currentStep < simulationSteps.length - 1) {
          setTimeout(() => onStep('forward'), animationConfig.duration);
        }
      } catch (error) {
        handleError(error as Error, 'main spring animation');
      }
    }
  });

  // State transition animations
  const stateTransitions = useTransition(
    currentStepData?.current_state ? [currentStepData.current_state] : [],
    {
      from: { opacity: 0, scale: 0.8, transform: 'translateY(-20px)' },
      enter: { opacity: 1, scale: 1.2, transform: 'translateY(0px)' },
      leave: { opacity: 0.6, scale: 1, transform: 'translateY(0px)' },
      config: config[animationConfig.easing],
      trail: animationConfig.stagger
    }
  );

  // Input tape animation for Turing Machine
  const TapeAnimation: React.FC<{ step: SimulationStep }> = ({ step }) => {
    const tapeSpring = useSpring({
      headPosition: step.head_position || 0,
      config: config.gentle
    });

    const tapeContents = step.tape_contents || [];
    const inputPosition = step.input_position || 0;

    const cellTransitions = useTransition(
      tapeContents.map((symbol, index) => ({ symbol, index, key: index })),
      {
        from: { opacity: 0, transform: 'scale(0.5)' },
        enter: { opacity: 1, transform: 'scale(1)' },
        update: { opacity: 1, transform: 'scale(1)' },
        leave: { opacity: 0, transform: 'scale(0.5)' },
        keys: (item) => item.key,
        config: config.wobbly
      }
    );

    return (
      <div className="flex flex-col space-y-4">
        <div className="flex items-center space-x-2">
          <Activity className="h-4 w-4" />
          <span className="text-sm font-medium">Tape Contents</span>
        </div>
        <div className="relative">
          <div className="flex space-x-1 overflow-x-auto pb-2">
            {cellTransitions((style, item) => (
              <animated.div
                style={style}
                className={cn(
                  "flex-shrink-0 w-10 h-10 border-2 rounded flex items-center justify-center text-sm font-mono",
                  item.index === (step.head_position || 0)
                    ? "border-blue-500 bg-blue-100 dark:bg-blue-900"
                    : "border-gray-300 dark:border-gray-600"
                )}
              >
                {item.symbol || '□'}
              </animated.div>
            ))}
          </div>
          <animated.div
            style={{
              transform: tapeSpring.headPosition.to(pos => `translateX(${pos * 44}px)`)
            }}
            className="absolute -top-6 w-0 h-0 border-l-4 border-r-4 border-b-4 border-transparent border-b-blue-500"
          />
        </div>
      </div>
    );
  };

  // Stack visualization for PDA
  const StackAnimation: React.FC<{ step: SimulationStep }> = ({ step }) => {
    const stackContents = step.stack_contents || [];
    
    const stackTransitions = useTransition(
      stackContents.map((symbol, index) => ({ symbol, index, key: `${symbol}-${index}` })),
      {
        from: { opacity: 0, transform: 'translateY(20px) scale(0.8)' },
        enter: { opacity: 1, transform: 'translateY(0px) scale(1)' },
        leave: { opacity: 0, transform: 'translateY(-20px) scale(0.8)' },
        keys: (item) => item.key,
        config: config.wobbly,
        trail: 50
      }
    );

    return (
      <div className="flex flex-col space-y-4">
        <div className="flex items-center space-x-2">
          <Zap className="h-4 w-4" />
          <span className="text-sm font-medium">Stack Contents</span>
        </div>
        <div className="flex flex-col-reverse space-y-reverse space-y-1 max-h-60 overflow-y-auto">
          {stackTransitions((style, item) => (
            <animated.div
              style={style}
              className="px-3 py-2 bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900 dark:to-pink-900 rounded border border-purple-300 dark:border-purple-600 text-center font-mono"
            >
              {item.symbol}
            </animated.div>
          ))}
          {stackContents.length === 0 && (
            <div className="px-3 py-2 bg-gray-100 dark:bg-gray-800 rounded border border-gray-300 dark:border-gray-600 text-center text-gray-500">
              Stack Empty
            </div>
          )}
        </div>
      </div>
    );
  };

  // Parse tree animation for CFG
  const ParseTreeAnimation: React.FC<{ productions: CFGProduction[], currentProduction?: string }> = ({ 
    productions, 
    currentProduction 
  }) => {
    const nodeTransitions = useTransition(
      productions.map((prod, index) => ({ ...prod, index })),
      {
        from: { opacity: 0, scale: 0.5, transform: 'translateY(-10px)' },
        enter: { opacity: 1, scale: 1, transform: 'translateY(0px)' },
        update: (item) => ({
          opacity: item.id === currentProduction ? 1 : 0.6,
          scale: item.id === currentProduction ? 1.1 : 1,
          transform: 'translateY(0px)'
        }),
        leave: { opacity: 0, scale: 0.5, transform: 'translateY(10px)' },
        keys: (item) => item.id,
        config: config.gentle,
        trail: animationConfig.stagger
      }
    );

    return (
      <div className="flex flex-col space-y-4">
        <div className="flex items-center space-x-2">
          <TreePine className="h-4 w-4" />
          <span className="text-sm font-medium">Parse Tree</span>
        </div>
        <div className="space-y-2 max-h-60 overflow-y-auto">
          {nodeTransitions((style, item) => (
            <animated.div
              style={style}
              className={cn(
                "px-3 py-2 rounded border text-sm font-mono",
                item.id === currentProduction
                  ? "bg-green-100 border-green-500 dark:bg-green-900 dark:border-green-400"
                  : "bg-gray-100 border-gray-300 dark:bg-gray-800 dark:border-gray-600"
              )}
            >
              {item.left_side} → {item.right_side}
            </animated.div>
          ))}
        </div>
      </div>
    );
  };

  // Input string animation
  const InputAnimation: React.FC<{ step: SimulationStep }> = ({ step }) => {
    const inputPosition = step.input_position || 0;
    const remainingInput = step.remaining_input || '';
    const processedInput = currentStepData?.remaining_input ? 
      (simulationSteps[0]?.remaining_input || '').slice(0, inputPosition) : '';

    const charTransitions = useTransition(
      [...processedInput, ...remainingInput].map((char, index) => ({
        char,
        index,
        processed: index < inputPosition,
        current: index === inputPosition,
        key: `${char}-${index}`
      })),
      {
        from: { opacity: 0.5, scale: 0.8 },
        enter: { opacity: 1, scale: 1 },
        update: (item) => ({
          opacity: item.processed ? 0.5 : item.current ? 1 : 0.8,
          scale: item.current ? 1.2 : 1,
        }),
        leave: { opacity: 0, scale: 0.5 },
        keys: (item) => item.key,
        config: config.gentle
      }
    );

    return (
      <div className="flex flex-col space-y-2">
        <div className="text-sm font-medium">Input String</div>
        <div className="flex space-x-1">
          {charTransitions((style, item) => (
            <animated.div
              style={style}
              className={cn(
                "w-8 h-8 border rounded flex items-center justify-center text-sm font-mono",
                item.processed ? "bg-green-100 border-green-500 dark:bg-green-900" :
                item.current ? "bg-blue-100 border-blue-500 dark:bg-blue-900 animate-pulse" :
                "bg-gray-100 border-gray-300 dark:bg-gray-800 dark:border-gray-600"
              )}
            >
              {item.char}
            </animated.div>
          ))}
        </div>
      </div>
    );
  };

  const renderAutomatonSpecificAnimation = () => {
    if (!currentStepData) return null;

    switch (automaton.type) {
      case 'tm':
        return <TapeAnimation step={currentStepData} />;
      case 'pda':
        return <StackAnimation step={currentStepData} />;
      case 'cfg':
        const cfgAutomaton = automaton as CFGAutomaton;
        return <ParseTreeAnimation 
          productions={cfgAutomaton.productions} 
          currentProduction={currentStepData.action_description} 
        />;
      default:
        return <InputAnimation step={currentStepData} />;
    }
  };

  return (
    <TooltipProvider>
      <Card className={cn("w-full", className)}>
        <CardHeader className={cn("pb-3", compactMode && "pb-2 px-3")}>
          <div className="flex items-center justify-between">
            <CardTitle className={cn("flex items-center space-x-2", compactMode && "text-base")}>
              <Activity className={cn("h-5 w-5", compactMode && "h-4 w-4")} />
              <span>Animation System</span>
            </CardTitle>
            <div className="flex items-center space-x-1">
              {/* View Mode Toggle */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setViewMode(prev => prev === 'desktop' ? 'mobile' : 'desktop')}
                  >
                    {viewMode === 'desktop' ? <Monitor className="h-4 w-4" /> : <Smartphone className="h-4 w-4" />}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  Switch to {viewMode === 'desktop' ? 'mobile' : 'desktop'} view
                </TooltipContent>
              </Tooltip>
              
              {/* Export Options */}
              {showExportOptions && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={startRecording}
                      disabled={isExporting || simulationSteps.length === 0}
                    >
                      {isExporting ? <Camera className="h-4 w-4 animate-pulse" /> : <Download className="h-4 w-4" />}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    Export animation
                  </TooltipContent>
                </Tooltip>
              )}
              
              {/* Settings */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowSettings(!showSettings)}
                  >
                    <Settings className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  Animation settings
                </TooltipContent>
              </Tooltip>
            </div>
          </div>
          
          {/* Error Display */}
          <AnimatePresence>
            {animationError && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex items-center space-x-2 p-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md"
              >
                <AlertCircle className="h-4 w-4 text-red-500" />
                <span className="text-sm text-red-700 dark:text-red-300">{animationError}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setAnimationError(null)}
                  className="ml-auto h-6 w-6 p-0"
                >
                  <XCircle className="h-3 w-3" />
                </Button>
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Export Progress */}
          <AnimatePresence>
            {isExporting && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="space-y-2"
              >
                <div className="flex items-center space-x-2">
                  <Video className="h-4 w-4 animate-pulse" />
                  <span className="text-sm">Exporting animation...</span>
                </div>
                <Progress value={exportProgress} className="w-full" />
              </motion.div>
            )}
          </AnimatePresence>
        
        {showSettings && (
          <div className="space-y-4 pt-4 border-t">
            <div className="grid grid-cols-1 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Animation Speed</label>
                <Slider
                  value={[animationConfig.duration]}
                  onValueChange={([value]) => 
                    setAnimationConfig(prev => ({ ...prev, duration: value }))
                  }
                  max={2000}
                  min={100}
                  step={50}
                  className="w-full"
                />
                <div className="text-xs text-gray-500">{animationConfig.duration}ms</div>
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium">Stagger Delay</label>
                <Slider
                  value={[animationConfig.stagger]}
                  onValueChange={([value]) => 
                    setAnimationConfig(prev => ({ ...prev, stagger: value }))
                  }
                  max={300}
                  min={0}
                  step={10}
                  className="w-full"
                />
                <div className="text-xs text-gray-500">{animationConfig.stagger}ms</div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={animationConfig.showTrails}
                    onChange={(e) => 
                      setAnimationConfig(prev => ({ ...prev, showTrails: e.target.checked }))
                    }
                    className="rounded"
                  />
                  <label className="text-sm">Show animation trails</label>
                </div>
                
                {showExportOptions && (
                  <div className="space-y-2 p-3 bg-gray-50 dark:bg-gray-800/50 rounded-md">
                    <h4 className="text-sm font-medium">Export Options</h4>
                    <div className="space-y-2">
                      <div>
                        <label className="text-xs text-gray-600 dark:text-gray-400">Format</label>
                        <select 
                          className="w-full mt-1 text-sm border rounded px-2 py-1"
                          value={animationConfig.exportFormat}
                          onChange={(e) => setAnimationConfig(prev => ({ 
                            ...prev, 
                            exportFormat: e.target.value as 'gif' | 'webm' | 'mp4'
                          }))}
                        >
                          <option value="gif">GIF</option>
                          <option value="webm">WebM</option>
                          <option value="mp4">MP4</option>
                        </select>
                      </div>
                      <div>
                        <label className="text-xs text-gray-600 dark:text-gray-400">Quality</label>
                        <select 
                          className="w-full mt-1 text-sm border rounded px-2 py-1"
                          value={animationConfig.exportQuality}
                          onChange={(e) => setAnimationConfig(prev => ({ 
                            ...prev, 
                            exportQuality: e.target.value as 'low' | 'medium' | 'high'
                          }))}
                        >
                          <option value="low">Low</option>
                          <option value="medium">Medium</option>
                          <option value="high">High</option>
                        </select>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Progress</span>
            <span>{currentStep + 1} / {simulationSteps.length}</span>
          </div>
          <Progress value={(currentStep / Math.max(1, simulationSteps.length - 1)) * 100} />
        </div>

        {/* Control Panel */}
        <div className={cn("flex items-center justify-center", spacing)}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size={controlSize}
                onClick={onReset}
                disabled={currentStep === 0}
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Reset to beginning</TooltipContent>
          </Tooltip>
          
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size={controlSize}
                onClick={() => onStep('backward')}
                disabled={currentStep === 0}
              >
                <SkipBack className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Previous step</TooltipContent>
          </Tooltip>
          
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onClick={isPlaying ? onPause : onPlay}
                disabled={currentStep >= simulationSteps.length - 1}
                className={cn("px-6", compactMode && "px-4")}
                size={controlSize}
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{isPlaying ? 'Pause' : 'Play'} animation</TooltipContent>
          </Tooltip>
          
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size={controlSize}
                onClick={() => onStep('forward')}
                disabled={currentStep >= simulationSteps.length - 1}
              >
                <SkipForward className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Next step</TooltipContent>
          </Tooltip>
          
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size={controlSize}
                onClick={onStop}
                disabled={!isPlaying}
              >
                <Square className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Stop animation</TooltipContent>
          </Tooltip>
        </div>

        {/* Current Step Information */}
        {currentStepData && (
          <div className="space-y-4">
            <Separator />
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Badge variant="secondary">
                  Step {currentStepData.step_number}
                </Badge>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {currentStepData.action_description}
                </p>
              </div>
              
              <div className="space-y-2">
                <div className="text-sm">
                  <span className="font-medium">Current State: </span>
                  <Badge variant="outline">{currentStepData.current_state}</Badge>
                </div>
                {currentStepData.remaining_input && (
                  <div className="text-sm">
                    <span className="font-medium">Remaining Input: </span>
                    <code className="px-1 py-0.5 bg-gray-100 dark:bg-gray-800 rounded">
                      {currentStepData.remaining_input}
                    </code>
                  </div>
                )}
              </div>
            </div>

            <Separator />

            {/* Automaton-specific animations */}
            <animated.div style={mainSpring}>
              {renderAutomatonSpecificAnimation()}
            </animated.div>
          </div>
        )}

        {simulationSteps.length === 0 && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-8 text-gray-500"
          >
            <Activity className="h-12 w-12 mx-auto mb-4 opacity-30" />
            <p className="text-lg font-medium">No simulation data available</p>
            <p className="text-sm mt-2">Start a simulation to see animations</p>
          </motion.div>
        )}
        
        {/* Hidden canvas for export functionality */}
        <canvas 
          ref={canvasRef} 
          className="hidden" 
          width={800} 
          height={600}
        />
      </CardContent>
    </Card>
    </TooltipProvider>
  );
};

export default AnimationSystem;