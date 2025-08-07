/**
 * Enhanced Router with lazy loading, prefetching, and performance optimization
 * Implements intelligent route splitting and loading strategies
 */
import React, { Suspense, useEffect, useState, useCallback } from 'react';
import { createBrowserRouter, RouterProvider, Outlet, useNavigate, useLocation, Navigate } from 'react-router-dom';
import { LazyWrapper, LoadingFallbacks, SmartPrefetcher, preloadComponents } from '@/utils/lazy-loading';
import { LazyRoutes, RoutePreloadStrategy, initializePreloading } from './lazy-routes';
import { ChunkErrorBoundary } from '@/components/ChunkErrorBoundary';

// Route transition animations and performance monitoring
interface RouteTransition {
  isLoading: boolean;
  progress: number;
  route: string;
}

// Performance metrics tracking
interface RouteMetrics {
  loadTime: number;
  chunkSize?: number;
  cacheHit: boolean;
}

const routeMetrics = new Map<string, RouteMetrics>();

/**
 * Root layout component with error boundary and loading management
 */
function RootLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const [transition, setTransition] = useState<RouteTransition>({
    isLoading: false,
    progress: 0,
    route: location.pathname
  });

  // Track navigation for smart prefetching
  useEffect(() => {
    SmartPrefetcher.trackNavigation(location.pathname);
  }, [location.pathname]);

  // Prefetch predicted routes
  useEffect(() => {
    const predictedRoutes = SmartPrefetcher.predictNextRoutes(location.pathname);
    console.log('Predicted next routes:', predictedRoutes);
    // Here you would trigger prefetching for predicted routes
  }, [location.pathname]);

  // Loading transition handler
  const handleRouteChange = useCallback((newRoute: string) => {
    setTransition({
      isLoading: true,
      progress: 0,
      route: newRoute
    });

    // Simulate loading progress (in real app, this would be based on actual loading events)
    const progressInterval = setInterval(() => {
      setTransition(prev => ({
        ...prev,
        progress: Math.min(prev.progress + Math.random() * 30, 90)
      }));
    }, 100);

    // Complete loading after component loads
    setTimeout(() => {
      clearInterval(progressInterval);
      setTransition({
        isLoading: false,
        progress: 100,
        route: newRoute
      });
    }, 300);
  }, []);

  return (
    <ChunkErrorBoundary>
      <div className="min-h-screen bg-background text-foreground">
        {/* Loading progress bar */}
        {transition.isLoading && (
          <div className="fixed top-0 left-0 right-0 z-50">
            <div 
              className="h-1 bg-primary transition-all duration-300 ease-out"
              style={{ width: `${transition.progress}%` }}
            />
          </div>
        )}

        <LazyWrapper
          fallback={<LoadingFallbacks.page />}
          onError={(error, errorInfo) => {
            console.error('Route loading error:', error, errorInfo);
            // Track route loading errors
            routeMetrics.set(location.pathname, {
              loadTime: performance.now(),
              cacheHit: false
            });
          }}
        >
          <Outlet />
        </LazyWrapper>
      </div>
    </ChunkErrorBoundary>
  );
}

/**
 * Enhanced route protection component
 */
interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredPermissions?: string[];
  fallback?: React.ReactNode;
}

function ProtectedRoute({ children, requiredPermissions = [], fallback }: ProtectedRouteProps) {
  // Simulate authentication check (replace with real auth logic)
  const [isAuthenticated, setIsAuthenticated] = useState(true);
  const [hasPermissions, setHasPermissions] = useState(true);

  useEffect(() => {
    // Simulate async auth check
    const checkAuth = async () => {
      try {
        // Replace with real auth check
        await new Promise(resolve => setTimeout(resolve, 100));
        setIsAuthenticated(true);
        setHasPermissions(true);
      } catch {
        setIsAuthenticated(false);
      }
    };

    checkAuth();
  }, [requiredPermissions]);

  if (!isAuthenticated) {
    return fallback || <Navigate to="/login" replace />;
  }

  if (!hasPermissions) {
    return fallback || <Navigate to="/unauthorized" replace />;
  }

  return <>{children}</>;
}

/**
 * Main App Router component with lazy loading
 */
function MainApp() {
  return (
    <Suspense fallback={LoadingFallbacks.page}>
      <LazyRoutes.ProblemSelector />
    </Suspense>
  );
}

/**
 * Problem View Route
 */
function ProblemViewRoute() {
  return (
    <Suspense fallback={LoadingFallbacks.page}>
      <LazyRoutes.ComprehensiveProblemView />
    </Suspense>
  );
}

/**
 * AI Features Route Bundle
 */
function AIFeaturesRoute() {
  return (
    <div className="space-y-6">
      <Suspense fallback={LoadingFallbacks.component}>
        <LazyRoutes.AITutor />
      </Suspense>
      <Suspense fallback={LoadingFallbacks.component}>
        <LazyRoutes.AIAssistantPanel />
      </Suspense>
    </div>
  );
}

/**
 * Automata Workspace Route
 */
function AutomataWorkspaceRoute() {
  return (
    <div className="h-screen flex flex-col">
      <Suspense fallback={LoadingFallbacks.canvas}>
        <LazyRoutes.EnhancedAutomataCanvas />
      </Suspense>
      <div className="flex-1">
        <Suspense fallback={LoadingFallbacks.component}>
          <LazyRoutes.SimulationEngine />
        </Suspense>
      </div>
    </div>
  );
}

/**
 * Collaboration Route
 */
function CollaborationRoute() {
  return (
    <ProtectedRoute requiredPermissions={['collaborate']}>
      <Suspense fallback={LoadingFallbacks.page}>
        <LazyRoutes.CollaborativeWorkspace />
      </Suspense>
    </ProtectedRoute>
  );
}

/**
 * Theory and Visualization Route
 */
function TheoryRoute() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Suspense fallback={LoadingFallbacks.component}>
        <LazyRoutes.ComplexityTheory />
      </Suspense>
      <Suspense fallback={LoadingFallbacks.chart}>
        <LazyRoutes.PumpingLemmaVisualizer />
      </Suspense>
    </div>
  );
}

/**
 * Advanced Features Route
 */
function AdvancedFeaturesRoute() {
  return (
    <ProtectedRoute requiredPermissions={['advanced']}>
      <div className="space-y-6">
        <Suspense fallback={LoadingFallbacks.component}>
          <LazyRoutes.AdvancedJFLAPInterface />
        </Suspense>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Suspense fallback={LoadingFallbacks.component}>
            <LazyRoutes.UniversalTuringMachine />
          </Suspense>
          <Suspense fallback={LoadingFallbacks.component}>
            <LazyRoutes.MultiTapeTuringMachine />
          </Suspense>
        </div>
      </div>
    </ProtectedRoute>
  );
}

/**
 * Learning Dashboard Route
 */
function LearningDashboardRoute() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Suspense fallback={LoadingFallbacks.component}>
            <LazyRoutes.AdaptiveLearning />
          </Suspense>
        </div>
        <div>
          <Suspense fallback={LoadingFallbacks.chart}>
            <LazyRoutes.ProgressVisualization />
          </Suspense>
        </div>
      </div>
      <Suspense fallback={LoadingFallbacks.page}>
        <LazyRoutes.CourseStructure />
      </Suspense>
    </div>
  );
}

/**
 * Settings and Utilities Route
 */
function SettingsRoute() {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <Suspense fallback={LoadingFallbacks.component}>
        <LazyRoutes.AccessibilitySettings />
      </Suspense>
      <Suspense fallback={LoadingFallbacks.dialog}>
        <LazyRoutes.CodeExporter />
      </Suspense>
    </div>
  );
}

/**
 * Research and Papers Route
 */
function ResearchRoute() {
  return (
    <Suspense fallback={LoadingFallbacks.page}>
      <LazyRoutes.ResearchPapers />
    </Suspense>
  );
}

/**
 * Error Page Components
 */
function NotFoundPage() {
  const navigate = useNavigate();
  
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-muted-foreground">404</h1>
        <p className="text-xl text-muted-foreground mt-4">Page not found</p>
        <button
          onClick={() => navigate('/')}
          className="mt-6 px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
        >
          Go Home
        </button>
      </div>
    </div>
  );
}

function ErrorPage() {
  const navigate = useNavigate();
  
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-destructive">Error</h1>
        <p className="text-lg text-muted-foreground mt-4">Something went wrong</p>
        <div className="space-x-4 mt-6">
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            Retry
          </button>
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/90 transition-colors"
          >
            Go Home
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Router configuration with optimized lazy loading
 */
export const router = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,
    errorElement: <ErrorPage />,
    children: [
      {
        index: true,
        element: <MainApp />
      },
      {
        path: 'problems',
        children: [
          {
            index: true,
            element: <MainApp />
          },
          {
            path: ':id',
            element: <ProblemViewRoute />
          }
        ]
      },
      {
        path: 'automata',
        element: <AutomataWorkspaceRoute />
      },
      {
        path: 'ai',
        element: <AIFeaturesRoute />
      },
      {
        path: 'theory',
        element: <TheoryRoute />
      },
      {
        path: 'collaborate',
        element: <CollaborationRoute />
      },
      {
        path: 'advanced',
        element: <AdvancedFeaturesRoute />
      },
      {
        path: 'learn',
        element: <LearningDashboardRoute />
      },
      {
        path: 'research',
        element: <ResearchRoute />
      },
      {
        path: 'settings',
        element: <SettingsRoute />
      },
      {
        path: '*',
        element: <NotFoundPage />
      }
    ]
  }
]);

/**
 * Router Provider with Performance Monitoring
 */
interface AppRouterProps {
  onRouteChange?: (route: string, metrics: RouteMetrics) => void;
}

export function AppRouter({ onRouteChange }: AppRouterProps) {
  useEffect(() => {
    // Initialize preloading strategies
    initializePreloading();
    
    // Preload high-priority components
    preloadComponents([
      { importFn: () => import('@/components/ProblemSelector'), name: 'ProblemSelector', priority: 10 },
      { importFn: () => import('@/components/AutomataCanvas'), name: 'AutomataCanvas', priority: 9 },
      { importFn: () => import('@/components/AITutor'), name: 'AITutor', priority: 8 },
    ], 'idle');

    // Performance monitoring
    const observer = new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => {
        if (entry.entryType === 'navigation') {
          const navEntry = entry as PerformanceNavigationTiming;
          const metrics: RouteMetrics = {
            loadTime: navEntry.loadEventEnd - navEntry.navigationStart,
            cacheHit: navEntry.transferSize === 0
          };
          
          routeMetrics.set(window.location.pathname, metrics);
          onRouteChange?.(window.location.pathname, metrics);
        }
      });
    });

    observer.observe({ entryTypes: ['navigation', 'resource'] });

    return () => observer.disconnect();
  }, [onRouteChange]);

  return <RouterProvider router={router} />;
}

/**
 * Export utilities for external use
 */
export { routeMetrics, SmartPrefetcher };
export type { RouteMetrics, RouteTransition };

export default AppRouter;