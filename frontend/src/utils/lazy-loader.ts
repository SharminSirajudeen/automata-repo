/**
 * Advanced lazy loading utilities for code splitting
 * Provides preloading, error handling, and loading states
 */
import { lazy, ComponentType, LazyExoticComponent, Suspense, ReactNode, useState, useEffect } from 'react';

interface LazyLoadOptions {
  preload?: boolean;
  retries?: number;
  delay?: number;
  fallback?: ComponentType;
}

interface ImportFunction<T = any> {
  (): Promise<{ default: ComponentType<T> }>;
}

// Preload cache to store component promises
const preloadCache = new Map<string, Promise<any>>();

/**
 * Enhanced lazy loading with retry logic and preloading
 */
export function lazyWithRetry<T = any>(
  importFunction: ImportFunction<T>,
  componentName: string,
  options: LazyLoadOptions = {}
): LazyExoticComponent<ComponentType<T>> {
  const { retries = 3, delay = 1000, preload = false } = options;

  const loadComponent = async (attempt = 0): Promise<{ default: ComponentType<T> }> => {
    try {
      return await importFunction();
    } catch (error) {
      if (attempt < retries) {
        // Add exponential backoff
        const backoffDelay = delay * Math.pow(2, attempt);
        await new Promise(resolve => setTimeout(resolve, backoffDelay));
        return loadComponent(attempt + 1);
      }
      
      console.error(`Failed to load component ${componentName} after ${retries} attempts:`, error);
      
      // Return fallback component on final failure
      return {
        default: options.fallback || (() => (
          <div className="flex items-center justify-center p-8">
            <div className="text-center">
              <h3 className="text-lg font-medium text-gray-900">Failed to load component</h3>
              <p className="mt-2 text-sm text-gray-600">
                {componentName} could not be loaded. Please refresh the page.
              </p>
              <button
                onClick={() => window.location.reload()}
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Refresh Page
              </button>
            </div>
          </div>
        ))
      };
    }
  };

  const componentPromise = loadComponent();
  
  if (preload) {
    preloadCache.set(componentName, componentPromise);
  }

  return lazy(() => componentPromise);
}

/**
 * Preload a component
 */
export function preloadComponent(
  importFunction: ImportFunction,
  componentName: string
): void {
  if (!preloadCache.has(componentName)) {
    preloadCache.set(componentName, importFunction());
  }
}

/**
 * Higher-order component for lazy loading with loading states
 */
export interface LazyWrapperProps {
  fallback?: ReactNode;
  children: ReactNode;
}

export function LazyWrapper({ fallback, children }: LazyWrapperProps) {
  const defaultFallback = (
    <div className="flex items-center justify-center p-8">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      <span className="ml-3 text-gray-600">Loading...</span>
    </div>
  );

  return (
    <Suspense fallback={fallback || defaultFallback}>
      {children}
    </Suspense>
  );
}

/**
 * Advanced loading states for different component types
 */
export const LoadingFallbacks = {
  page: (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
        <p className="mt-4 text-gray-600">Loading page...</p>
      </div>
    </div>
  ),

  component: (
    <div className="flex items-center justify-center p-4">
      <div className="animate-pulse flex space-x-4">
        <div className="rounded-full bg-gray-300 h-6 w-6"></div>
        <div className="flex-1 space-y-2 py-1">
          <div className="h-4 bg-gray-300 rounded w-3/4"></div>
          <div className="h-4 bg-gray-300 rounded w-1/2"></div>
        </div>
      </div>
    </div>
  ),

  dialog: (
    <div className="flex items-center justify-center p-6">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
    </div>
  ),

  chart: (
    <div className="w-full h-64 flex items-center justify-center bg-gray-50 rounded-lg">
      <div className="text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
        <p className="mt-2 text-sm text-gray-600">Loading chart...</p>
      </div>
    </div>
  )
};

/**
 * Intersection Observer based lazy loading for components
 */
export function useIntersectionLazyLoad(threshold = 0.1) {
  const [isVisible, setIsVisible] = useState(false);
  const [element, setElement] = useState<HTMLElement | null>(null);

  useEffect(() => {
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold }
    );

    observer.observe(element);

    return () => observer.disconnect();
  }, [element, threshold]);

  return [setElement, isVisible] as const;
}