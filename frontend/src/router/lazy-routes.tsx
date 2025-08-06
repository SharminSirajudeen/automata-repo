/**
 * Route-based code splitting configuration
 * Implements lazy loading for all major application routes
 */
import React from 'react';
import { lazyWithRetry, LoadingFallbacks, preloadComponent } from '@/utils/lazy-loader';

// Main application pages - split by route
export const LazyRoutes = {
  // Core application pages
  ProblemView: lazyWithRetry(
    () => import('@/components/ProblemView'),
    'ProblemView',
    { fallback: () => LoadingFallbacks.page }
  ),

  ComprehensiveProblemView: lazyWithRetry(
    () => import('@/components/ComprehensiveProblemView'),
    'ComprehensiveProblemView',
    { fallback: () => LoadingFallbacks.page }
  ),

  // Automata-related components
  AutomataCanvas: lazyWithRetry(
    () => import('@/components/AutomataCanvas'),
    'AutomataCanvas',
    { fallback: () => LoadingFallbacks.component }
  ),

  EnhancedAutomataCanvas: lazyWithRetry(
    () => import('@/components/EnhancedAutomataCanvas'),
    'EnhancedAutomataCanvas',
    { fallback: () => LoadingFallbacks.component }
  ),

  CollaborativeCanvas: lazyWithRetry(
    () => import('@/components/CollaborativeCanvas'),
    'CollaborativeCanvas',
    { fallback: () => LoadingFallbacks.component }
  ),

  // Advanced features
  AdvancedJFLAPInterface: lazyWithRetry(
    () => import('@/components/AdvancedJFLAPInterface'),
    'AdvancedJFLAPInterface',
    { fallback: () => LoadingFallbacks.page }
  ),

  AdvancedGrammarEditor: lazyWithRetry(
    () => import('@/components/AdvancedGrammarEditor'),
    'AdvancedGrammarEditor',
    { fallback: () => LoadingFallbacks.component }
  ),

  // Theory and visualization components
  ComplexityTheory: lazyWithRetry(
    () => import('@/components/ComplexityTheory'),
    'ComplexityTheory',
    { fallback: () => LoadingFallbacks.page }
  ),

  PumpingLemmaVisualizer: lazyWithRetry(
    () => import('@/components/PumpingLemmaVisualizer'),
    'PumpingLemmaVisualizer',
    { fallback: () => LoadingFallbacks.chart }
  ),

  SLRParserVisualization: lazyWithRetry(
    () => import('@/components/SLRParserVisualization'),
    'SLRParserVisualization',
    { fallback: () => LoadingFallbacks.chart }
  ),

  // AI and tutoring components
  AITutor: lazyWithRetry(
    () => import('@/components/AITutor'),
    'AITutor',
    { fallback: () => LoadingFallbacks.component }
  ),

  AIAssistantPanel: lazyWithRetry(
    () => import('@/components/AIAssistantPanel'),
    'AIAssistantPanel',
    { fallback: () => LoadingFallbacks.component }
  ),

  ProofAssistant: lazyWithRetry(
    () => import('@/components/ProofAssistant'),
    'ProofAssistant',
    { fallback: () => LoadingFallbacks.component }
  ),

  // Learning and progress components
  AdaptiveLearning: lazyWithRetry(
    () => import('@/components/AdaptiveLearning'),
    'AdaptiveLearning',
    { fallback: () => LoadingFallbacks.page }
  ),

  LearningMode: lazyWithRetry(
    () => import('@/components/LearningMode'),
    'LearningMode',
    { fallback: () => LoadingFallbacks.page }
  ),

  CourseStructure: lazyWithRetry(
    () => import('@/components/CourseStructure'),
    'CourseStructure',
    { fallback: () => LoadingFallbacks.page }
  ),

  ProgressVisualization: lazyWithRetry(
    () => import('@/components/ProgressVisualization'),
    'ProgressVisualization',
    { fallback: () => LoadingFallbacks.chart }
  ),

  // Research and papers
  ResearchPapers: lazyWithRetry(
    () => import('@/components/ResearchPapers'),
    'ResearchPapers',
    { fallback: () => LoadingFallbacks.page }
  ),

  // Turing machine components
  UniversalTuringMachine: lazyWithRetry(
    () => import('@/components/UniversalTuringMachine'),
    'UniversalTuringMachine',
    { fallback: () => LoadingFallbacks.component }
  ),

  MultiTapeTuringMachine: lazyWithRetry(
    () => import('@/components/MultiTapeTuringMachine'),
    'MultiTapeTuringMachine',
    { fallback: () => LoadingFallbacks.component }
  ),

  // Simulation and engines
  SimulationEngine: lazyWithRetry(
    () => import('@/components/SimulationEngine'),
    'SimulationEngine',
    { fallback: () => LoadingFallbacks.component }
  ),

  // Project and collaboration management
  ProjectManager: lazyWithRetry(
    () => import('@/components/ProjectManager'),
    'ProjectManager',
    { fallback: () => LoadingFallbacks.page }
  ),

  CollaborativeWorkspace: lazyWithRetry(
    () => import('@/components/CollaborativeWorkspace'),
    'CollaborativeWorkspace',
    { fallback: () => LoadingFallbacks.page }
  ),

  RoomManager: lazyWithRetry(
    () => import('@/components/RoomManager'),
    'RoomManager',
    { fallback: () => LoadingFallbacks.component }
  ),

  // Utility and settings components
  FormalVerification: lazyWithRetry(
    () => import('@/components/FormalVerification'),
    'FormalVerification',
    { fallback: () => LoadingFallbacks.component }
  ),

  CodeExporter: lazyWithRetry(
    () => import('@/components/CodeExporter'),
    'CodeExporter',
    { fallback: () => LoadingFallbacks.dialog }
  ),

  AccessibilitySettings: lazyWithRetry(
    () => import('@/components/AccessibilitySettings'),
    'AccessibilitySettings',
    { fallback: () => LoadingFallbacks.component }
  ),

  // L-System and specialized renderers
  LSystemRenderer: lazyWithRetry(
    () => import('@/components/LSystemRenderer'),
    'LSystemRenderer',
    { fallback: () => LoadingFallbacks.component }
  ),

  // Onboarding and tutorials
  GoogleDriveOnboarding: lazyWithRetry(
    () => import('@/components/GoogleDriveOnboarding'),
    'GoogleDriveOnboarding',
    { fallback: () => LoadingFallbacks.dialog }
  ),

  ExampleGallery: lazyWithRetry(
    () => import('@/components/ExampleGallery'),
    'ExampleGallery',
    { fallback: () => LoadingFallbacks.page }
  ),
} as const;

/**
 * Route configuration with preloading strategy
 */
export const RoutePreloadStrategy = {
  // Critical path - preload immediately
  critical: [
    'ProblemView',
    'AutomataCanvas',
    'AITutor',
  ],

  // High priority - preload on idle
  high: [
    'ComprehensiveProblemView',
    'EnhancedAutomataCanvas',
    'AIAssistantPanel',
    'AdaptiveLearning',
  ],

  // Medium priority - preload on user interaction
  medium: [
    'ComplexityTheory',
    'ProofAssistant',
    'AdvancedJFLAPInterface',
    'CourseStructure',
  ],

  // Low priority - lazy load only when needed
  low: [
    'ResearchPapers',
    'GoogleDriveOnboarding',
    'AccessibilitySettings',
  ],
};

/**
 * Preload components based on strategy
 */
export function initializePreloading() {
  // Preload critical components immediately
  RoutePreloadStrategy.critical.forEach(componentName => {
    const componentKey = componentName as keyof typeof LazyRoutes;
    if (LazyRoutes[componentKey]) {
      // Trigger preload by accessing the component
      LazyRoutes[componentKey];
    }
  });

  // Preload high priority components on idle
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      RoutePreloadStrategy.high.forEach(componentName => {
        const componentKey = componentName as keyof typeof LazyRoutes;
        if (LazyRoutes[componentKey]) {
          LazyRoutes[componentKey];
        }
      });
    });
  }

  // Preload medium priority components on user interaction
  const preloadMediumPriority = () => {
    RoutePreloadStrategy.medium.forEach(componentName => {
      const componentKey = componentName as keyof typeof LazyRoutes;
      if (LazyRoutes[componentKey]) {
        LazyRoutes[componentKey];
      }
    });
  };

  // Listen for first user interaction
  const events = ['mousedown', 'keydown', 'touchstart'];
  const handleFirstInteraction = () => {
    preloadMediumPriority();
    events.forEach(event => {
      document.removeEventListener(event, handleFirstInteraction);
    });
  };

  events.forEach(event => {
    document.addEventListener(event, handleFirstInteraction, { once: true });
  });
}

/**
 * Dynamic import helper for additional modules
 */
export const DynamicImports = {
  // Chart libraries
  recharts: () => import('recharts'),
  
  // Animation libraries  
  framerMotion: () => import('framer-motion'),
  
  // Collaboration tools
  yjs: () => import('yjs'),
  yWebsocket: () => import('y-websocket'),
  
  // Socket.io
  socketIO: () => import('socket.io-client'),
  
  // Date utilities
  dateFns: () => import('date-fns'),
  
  // Form validation
  zod: () => import('zod'),
  
  // UUID generation
  uuid: () => import('uuid'),
};

/**
 * Route-specific component bundles
 */
export const RouteBundle = {
  automata: [
    LazyRoutes.AutomataCanvas,
    LazyRoutes.EnhancedAutomataCanvas,
    LazyRoutes.SimulationEngine,
  ],
  
  collaboration: [
    LazyRoutes.CollaborativeCanvas,
    LazyRoutes.CollaborativeWorkspace,
    LazyRoutes.RoomManager,
  ],
  
  theory: [
    LazyRoutes.ComplexityTheory,
    LazyRoutes.PumpingLemmaVisualizer,
    LazyRoutes.SLRParserVisualization,
  ],
  
  ai: [
    LazyRoutes.AITutor,
    LazyRoutes.AIAssistantPanel,
    LazyRoutes.ProofAssistant,
  ],
  
  learning: [
    LazyRoutes.AdaptiveLearning,
    LazyRoutes.LearningMode,
    LazyRoutes.CourseStructure,
    LazyRoutes.ProgressVisualization,
  ],
  
  advanced: [
    LazyRoutes.AdvancedJFLAPInterface,
    LazyRoutes.AdvancedGrammarEditor,
    LazyRoutes.UniversalTuringMachine,
    LazyRoutes.MultiTapeTuringMachine,
  ],
};

export default LazyRoutes;