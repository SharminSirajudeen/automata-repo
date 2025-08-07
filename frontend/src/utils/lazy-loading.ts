/**
 * Advanced lazy loading utilities for comprehensive code splitting
 * Provides preloading, error boundaries, retry logic, and performance monitoring
 */
import { 
  lazy, 
  ComponentType, 
  LazyExoticComponent, 
  Suspense, 
  ReactNode, 
  useState, 
  useEffect, 
  useCallback,
  useMemo,
  Component,
  ErrorInfo,
  ReactElement
} from 'react';

// Performance monitoring
interface LoadingMetrics {
  startTime: number;
  endTime?: number;
  duration?: number;
  retries: number;
  success: boolean;
  error?: string;
}

interface LazyLoadOptions {
  preload?: boolean;
  retries?: number;
  delay?: number;
  timeout?: number;
  fallback?: ComponentType;
  onError?: (error: Error, componentName: string) => void;
  onLoad?: (metrics: LoadingMetrics, componentName: string) => void;
  priority?: 'high' | 'medium' | 'low';
  prefetch?: boolean;
}

interface ImportFunction<T = any> {
  (): Promise<{ default: ComponentType<T> }>;
}

// Global caches and registries
const preloadCache = new Map<string, Promise<any>>();
const loadingMetrics = new Map<string, LoadingMetrics>();
const retryAttempts = new Map<string, number>();
const failedComponents = new Set<string>();

// Network-aware loading strategy
const getNetworkStatus = (): 'fast' | 'slow' | 'offline' => {
  if (!navigator.onLine) return 'offline';
  
  const connection = (navigator as any).connection;
  if (!connection) return 'fast';
  
  const effectiveType = connection.effectiveType;
  if (effectiveType === '4g') return 'fast';
  if (effectiveType === '3g') return 'slow';
  return 'slow';
};

/**
 * Enhanced lazy loading with comprehensive error handling and retry logic
 */
export function lazyWithRetry<T = any>(
  importFunction: ImportFunction<T>,
  componentName: string,
  options: LazyLoadOptions = {}
): LazyExoticComponent<ComponentType<T>> {
  const { 
    retries = 3, 
    delay = 1000, 
    timeout = 30000,
    preload = false,
    onError,
    onLoad,
    priority = 'medium',
    prefetch = false
  } = options;

  // Initialize metrics
  if (!loadingMetrics.has(componentName)) {
    loadingMetrics.set(componentName, {
      startTime: 0,
      retries: 0,
      success: false
    });
  }

  const loadComponent = async (attempt = 0): Promise<{ default: ComponentType<T> }> => {
    const metrics = loadingMetrics.get(componentName)!;
    
    // Start timing if first attempt
    if (attempt === 0) {
      metrics.startTime = performance.now();
    }

    try {
      // Create timeout promise
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error(`Component ${componentName} loading timeout`)), timeout);
      });

      // Load component with timeout
      const componentModule = await Promise.race([
        importFunction(),
        timeoutPromise
      ]);

      // Success - update metrics
      metrics.endTime = performance.now();
      metrics.duration = metrics.endTime - metrics.startTime;
      metrics.success = true;
      metrics.retries = attempt;

      onLoad?.(metrics, componentName);
      
      // Remove from failed components if previously failed
      failedComponents.delete(componentName);
      
      return componentModule;
    } catch (error) {
      metrics.retries = attempt + 1;
      
      if (attempt < retries) {
        // Calculate adaptive delay based on network and attempt
        const networkStatus = getNetworkStatus();
        const baseDelay = networkStatus === 'slow' ? delay * 2 : delay;
        const backoffDelay = baseDelay * Math.pow(2, attempt);
        const jitterDelay = backoffDelay + (Math.random() * 1000);
        
        console.warn(
          `Attempt ${attempt + 1}/${retries + 1} failed for ${componentName}. Retrying in ${jitterDelay}ms...`,
          error
        );
        
        await new Promise(resolve => setTimeout(resolve, jitterDelay));
        return loadComponent(attempt + 1);
      }
      
      // Final failure - update metrics and caches
      metrics.success = false;
      metrics.error = error instanceof Error ? error.message : String(error);
      failedComponents.add(componentName);
      
      console.error(`Failed to load component ${componentName} after ${retries + 1} attempts:`, error);
      onError?.(error instanceof Error ? error : new Error(String(error)), componentName);
      
      // Return enhanced fallback component
      return {
        default: options.fallback || createEnhancedFallback(componentName, error, () => {
          // Retry mechanism in fallback
          preloadCache.delete(componentName);
          failedComponents.delete(componentName);
          retryAttempts.delete(componentName);
          window.location.reload();
        })
      };
    }
  };

  const componentPromise = loadComponent();
  
  // Cache management
  if (preload || prefetch) {
    preloadCache.set(componentName, componentPromise);
  }

  // Priority-based preloading
  if (priority === 'high') {
    // Preload immediately
    componentPromise.catch(() => {}); // Prevent unhandled rejection
  } else if (priority === 'medium' && 'requestIdleCallback' in window) {
    // Preload on idle
    requestIdleCallback(() => {
      componentPromise.catch(() => {});
    });
  }

  return lazy(() => componentPromise);
}

/**
 * Create enhanced fallback component with retry functionality
 */
function createEnhancedFallback(
  componentName: string, 
  error: unknown,
  onRetry: () => void
): ComponentType {
  return () => {
    const [retrying, setRetrying] = useState(false);
    
    const handleRetry = async () => {
      setRetrying(true);
      try {
        await new Promise(resolve => setTimeout(resolve, 1000));
        onRetry();
      } catch (e) {
        console.error('Retry failed:', e);
      } finally {
        setRetrying(false);
      }
    };

    return (
      <div className="flex items-center justify-center p-8 min-h-[200px] bg-red-50 dark:bg-red-950 rounded-lg border border-red-200 dark:border-red-800">
        <div className="text-center max-w-md">
          <div className="mb-4">
            <svg className="w-12 h-12 mx-auto text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          
          <h3 className="text-lg font-semibold text-red-900 dark:text-red-100 mb-2">
            Component Failed to Load
          </h3>
          
          <p className="text-sm text-red-700 dark:text-red-300 mb-4">
            <strong>{componentName}</strong> could not be loaded due to a network error or timeout.
          </p>
          
          {error instanceof Error && (
            <details className="text-xs text-red-600 dark:text-red-400 mb-4 text-left">
              <summary className="cursor-pointer hover:text-red-800 dark:hover:text-red-200">
                Technical Details
              </summary>
              <pre className="mt-2 p-2 bg-red-100 dark:bg-red-900 rounded text-xs overflow-auto">
                {error.message}
              </pre>
            </details>
          )}
          
          <div className="space-y-2">
            <button
              onClick={handleRetry}
              disabled={retrying}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {retrying ? 'Retrying...' : 'Retry Loading'}
            </button>
            
            <button
              onClick={() => window.location.reload()}
              className="block w-full px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
            >
              Refresh Page
            </button>
          </div>
          
          <div className="mt-4 text-xs text-gray-500 dark:text-gray-400">
            Try checking your internet connection or contact support if the problem persists.
          </div>
        </div>
      </div>
    );
  };
}

/**
 * Bulk preload components based on usage patterns
 */
export function preloadComponents(
  components: Array<{ importFn: ImportFunction; name: string; priority?: number }>,
  strategy: 'immediate' | 'idle' | 'interaction' = 'idle'
): void {
  const sortedComponents = components.sort((a, b) => (b.priority || 0) - (a.priority || 0));
  
  const preloadBatch = (batch: typeof components) => {
    batch.forEach(({ importFn, name }) => {
      if (!preloadCache.has(name) && !failedComponents.has(name)) {
        const promise = importFn().catch(error => {
          console.warn(`Preload failed for ${name}:`, error);
          failedComponents.add(name);
        });
        preloadCache.set(name, promise);
      }
    });
  };

  switch (strategy) {
    case 'immediate':
      preloadBatch(sortedComponents);
      break;
      
    case 'idle':
      if ('requestIdleCallback' in window) {
        requestIdleCallback(() => preloadBatch(sortedComponents), { timeout: 5000 });
      } else {
        setTimeout(() => preloadBatch(sortedComponents), 100);
      }
      break;
      
    case 'interaction':
      const events = ['mousedown', 'keydown', 'touchstart', 'scroll'];
      const handleFirstInteraction = () => {
        preloadBatch(sortedComponents);
        events.forEach(event => {
          document.removeEventListener(event, handleFirstInteraction);
        });
      };
      
      events.forEach(event => {
        document.addEventListener(event, handleFirstInteraction, { once: true, passive: true });
      });
      break;
  }
}

/**
 * Advanced loading states with skeleton screens
 */
export const LoadingFallbacks = {
  page: (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center space-y-4">
        <div className="relative">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent mx-auto"></div>
          <div className="absolute inset-0 rounded-full h-12 w-12 border-4 border-primary/20 mx-auto"></div>
        </div>
        <div className="space-y-2">
          <div className="h-4 bg-muted animate-pulse rounded w-32 mx-auto"></div>
          <div className="h-3 bg-muted animate-pulse rounded w-24 mx-auto"></div>
        </div>
      </div>
    </div>
  ),

  component: (
    <div className="animate-pulse space-y-4 p-4">
      <div className="flex items-center space-x-4">
        <div className="rounded-full bg-muted h-10 w-10"></div>
        <div className="flex-1 space-y-2">
          <div className="h-4 bg-muted rounded w-3/4"></div>
          <div className="h-3 bg-muted rounded w-1/2"></div>
        </div>
      </div>
      <div className="space-y-2">
        <div className="h-4 bg-muted rounded"></div>
        <div className="h-4 bg-muted rounded w-5/6"></div>
        <div className="h-4 bg-muted rounded w-4/6"></div>
      </div>
    </div>
  ),

  dialog: (
    <div className="flex items-center justify-center p-8">
      <div className="relative">
        <div className="animate-spin rounded-full h-8 w-8 border-3 border-primary border-t-transparent"></div>
        <div className="absolute inset-0 rounded-full h-8 w-8 border-3 border-primary/20"></div>
      </div>
    </div>
  ),

  chart: (
    <div className="w-full h-64 bg-muted/30 rounded-lg p-6">
      <div className="animate-pulse h-full flex flex-col">
        <div className="flex-1 flex items-end space-x-2 mb-4">
          {Array.from({ length: 8 }, (_, i) => (
            <div 
              key={i}
              className="bg-muted flex-1 rounded-t"
              style={{ height: `${Math.random() * 80 + 20}%` }}
            />
          ))}
        </div>
        <div className="flex justify-between text-xs">
          {Array.from({ length: 4 }, (_, i) => (
            <div key={i} className="h-3 bg-muted rounded w-12" />
          ))}
        </div>
      </div>
    </div>
  ),

  canvas: (
    <div className="w-full h-96 bg-muted/20 rounded-lg flex items-center justify-center">
      <div className="text-center space-y-4">
        <div className="relative">
          <div className="animate-spin rounded-full h-10 w-10 border-3 border-primary border-t-transparent mx-auto"></div>
        </div>
        <div className="text-sm text-muted-foreground">Initializing canvas...</div>
      </div>
    </div>
  ),

  minimal: (
    <div className="flex items-center justify-center p-2">
      <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary border-t-transparent"></div>
    </div>
  )
};

/**
 * Optimized intersection observer for viewport-based loading
 */
export function useIntersectionLazyLoad(
  threshold = 0.1,
  rootMargin = '50px',
  triggerOnce = true
) {
  const [isVisible, setIsVisible] = useState(false);
  const [element, setElement] = useState<HTMLElement | null>(null);

  const observer = useMemo(
    () => new IntersectionObserver(
      ([entry]) => {
        const isIntersecting = entry.isIntersecting;
        setIsVisible(isIntersecting);
        
        if (isIntersecting && triggerOnce && element) {
          observer?.disconnect();
        }
      },
      { threshold, rootMargin }
    ),
    [threshold, rootMargin, triggerOnce, element]
  );

  useEffect(() => {
    if (!element) return;

    observer.observe(element);
    return () => observer.disconnect();
  }, [element, observer]);

  return [setElement, isVisible] as const;
}

/**
 * Performance monitoring and analytics
 */
export class LazyLoadAnalytics {
  static getMetrics(): Map<string, LoadingMetrics> {
    return new Map(loadingMetrics);
  }

  static getFailedComponents(): Set<string> {
    return new Set(failedComponents);
  }

  static getLoadingStats() {
    const metrics = Array.from(loadingMetrics.values());
    const successful = metrics.filter(m => m.success);
    const failed = metrics.filter(m => !m.success);
    
    return {
      total: metrics.length,
      successful: successful.length,
      failed: failed.length,
      averageLoadTime: successful.reduce((acc, m) => acc + (m.duration || 0), 0) / successful.length,
      successRate: successful.length / metrics.length,
      totalRetries: metrics.reduce((acc, m) => acc + m.retries, 0)
    };
  }

  static exportReport(): string {
    const stats = this.getLoadingStats();
    const detailedMetrics = this.getMetrics();
    
    return JSON.stringify({
      timestamp: new Date().toISOString(),
      summary: stats,
      detailed: Array.from(detailedMetrics.entries()).map(([name, metrics]) => ({
        component: name,
        ...metrics
      })),
      failedComponents: Array.from(this.getFailedComponents())
    }, null, 2);
  }
}

/**
 * Smart prefetching based on user behavior
 */
export class SmartPrefetcher {
  private static routes = new Map<string, number>(); // route -> access count
  private static navigationHistory: string[] = [];
  
  static trackNavigation(route: string) {
    this.navigationHistory.push(route);
    this.routes.set(route, (this.routes.get(route) || 0) + 1);
    
    // Keep only last 50 navigation entries
    if (this.navigationHistory.length > 50) {
      this.navigationHistory = this.navigationHistory.slice(-50);
    }
  }

  static predictNextRoutes(currentRoute: string, limit = 3): string[] {
    const predictions = new Map<string, number>();
    
    // Find patterns in navigation history
    for (let i = 0; i < this.navigationHistory.length - 1; i++) {
      if (this.navigationHistory[i] === currentRoute) {
        const nextRoute = this.navigationHistory[i + 1];
        predictions.set(nextRoute, (predictions.get(nextRoute) || 0) + 1);
      }
    }
    
    // Sort by frequency and return top predictions
    return Array.from(predictions.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([route]) => route);
  }
  
  static getMostAccessedRoutes(limit = 5): string[] {
    return Array.from(this.routes.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([route]) => route);
  }
}

/**
 * Higher-order component for lazy loading with enhanced error boundaries
 */
export interface LazyWrapperProps {
  fallback?: ReactNode;
  children: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

export function LazyWrapper({ fallback, children, onError }: LazyWrapperProps) {
  const defaultFallback = (
    <div className="flex items-center justify-center p-8">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      <span className="ml-3 text-muted-foreground">Loading...</span>
    </div>
  );

  return (
    <LazyErrorBoundary onError={onError}>
      <Suspense fallback={fallback || defaultFallback}>
        {children}
      </Suspense>
    </LazyErrorBoundary>
  );
}

/**
 * Enhanced error boundary for lazy-loaded components
 */
interface LazyErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

interface LazyErrorBoundaryProps {
  children: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  fallback?: ComponentType<{ error: Error; retry: () => void }>;
}

class LazyErrorBoundary extends Component<LazyErrorBoundaryProps, LazyErrorBoundaryState> {
  constructor(props: LazyErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): LazyErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('LazyErrorBoundary caught an error:', error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  retry = () => {
    this.setState({ hasError: false, error: undefined });
  };

  render() {
    if (this.state.hasError && this.state.error) {
      const FallbackComponent = this.props.fallback;
      
      if (FallbackComponent) {
        return <FallbackComponent error={this.state.error} retry={this.retry} />;
      }

      return (
        <div className="flex items-center justify-center p-8 min-h-[200px] bg-destructive/10 rounded-lg border border-destructive/20">
          <div className="text-center max-w-md">
            <h3 className="text-lg font-semibold text-destructive mb-2">
              Something went wrong
            </h3>
            <p className="text-sm text-muted-foreground mb-4">
              An error occurred while loading this component.
            </p>
            <button
              onClick={this.retry}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
            >
              Try again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Preload a single component with caching
 */
export function preloadComponent(
  importFunction: ImportFunction,
  componentName: string
): void {
  if (!preloadCache.has(componentName)) {
    preloadCache.set(componentName, importFunction());
  }
}