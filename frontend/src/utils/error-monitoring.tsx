import * as Sentry from '@sentry/react';
import { BrowserTracing } from '@sentry/react';
import React, { useEffect } from 'react';
import {
  useLocation,
  useNavigationType,
  createRoutesFromChildren,
  matchRoutes
} from 'react-router-dom';

interface ErrorMonitoringConfig {
  dsn?: string;
  environment: string;
  debug?: boolean;
  tracesSampleRate?: number;
  replaysSessionSampleRate?: number;
  replaysOnErrorSampleRate?: number;
}

class ErrorMonitoring {
  private initialized = false;
  private config: ErrorMonitoringConfig;

  constructor(config: ErrorMonitoringConfig) {
    this.config = config;
  }

  /**
   * Initialize Sentry error monitoring
   */
  initialize(): void {
    if (this.initialized || !this.config.dsn) {
      return;
    }

    try {
      Sentry.init({
        dsn: this.config.dsn,
        environment: this.config.environment,
        debug: this.config.debug || false,
        integrations: [
          new BrowserTracing({
            // Set up automatic route change tracking for React Router
            routingInstrumentation: Sentry.reactRouterV6Instrumentation(
              useEffect,
              useLocation,
              useNavigationType,
              createRoutesFromChildren,
              matchRoutes
            ),
          }),
          new Sentry.Replay({
            // Capture 10% of all sessions,
            // plus 100% of sessions with an error
            maskAllText: false,
            blockAllMedia: false,
          }),
        ],
        tracesSampleRate: this.config.tracesSampleRate || 0.1,
        replaysSessionSampleRate: this.config.replaysSessionSampleRate || 0.1,
        replaysOnErrorSampleRate: this.config.replaysOnErrorSampleRate || 1.0,
        
        // Performance monitoring
        beforeSend: (event, hint) => {
          // Filter out development errors
          if (this.config.environment === 'development') {
            console.warn('Sentry event (dev mode):', event);
            return null;
          }

          // Filter out specific errors we don't want to track
          if (event.exception) {
            const error = hint.originalException;
            if (this.shouldIgnoreError(error)) {
              return null;
            }
          }

          return event;
        },

        // Set user context
        initialScope: {
          tags: {
            component: 'frontend',
            version: import.meta.env.VITE_APP_VERSION || '1.0.0',
          },
        },
      });

      this.initialized = true;
      console.log('Error monitoring initialized successfully');
    } catch (error) {
      console.error('Failed to initialize error monitoring:', error);
    }
  }

  /**
   * Determine if an error should be ignored
   */
  private shouldIgnoreError(error: any): boolean {
    if (!error) return true;

    const ignoredPatterns = [
      // Network errors that are often not actionable
      'Network Error',
      'NetworkError',
      'Failed to fetch',
      'Load failed',
      
      // Browser extension errors
      'extension',
      'chrome-extension',
      'moz-extension',
      
      // AdBlocker related
      'BlockedByClient',
      'adblock',
      
      // Common false positives
      'ResizeObserver loop limit exceeded',
      'Non-Error promise rejection captured',
      'Script error',
      
      // Development-specific errors
      'HMR',
      'Hot reload',
    ];

    const errorMessage = error.message || error.toString();
    return ignoredPatterns.some(pattern => 
      errorMessage.toLowerCase().includes(pattern.toLowerCase())
    );
  }

  /**
   * Set user context for error tracking
   */
  setUser(user: {
    id?: string;
    email?: string;
    username?: string;
    [key: string]: any;
  }): void {
    if (!this.initialized) return;

    Sentry.setUser({
      id: user.id,
      email: user.email,
      username: user.username,
      ...user,
    });
  }

  /**
   * Set additional context for error tracking
   */
  setContext(key: string, context: Record<string, any>): void {
    if (!this.initialized) return;

    Sentry.setContext(key, context);
  }

  /**
   * Set tags for error categorization
   */
  setTags(tags: Record<string, string>): void {
    if (!this.initialized) return;

    Sentry.setTags(tags);
  }

  /**
   * Add breadcrumb for debugging context
   */
  addBreadcrumb(message: string, category: string = 'custom', level: Sentry.SeverityLevel = 'info', data?: any): void {
    if (!this.initialized) return;

    Sentry.addBreadcrumb({
      message,
      category,
      level,
      data,
      timestamp: Date.now() / 1000,
    });
  }

  /**
   * Capture an exception manually
   */
  captureException(error: Error, context?: Record<string, any>): string | undefined {
    if (!this.initialized) {
      console.error('Error monitoring not initialized:', error);
      return;
    }

    return Sentry.captureException(error, {
      contexts: context ? { additional: context } : undefined,
    });
  }

  /**
   * Capture a message manually
   */
  captureMessage(message: string, level: Sentry.SeverityLevel = 'info'): string | undefined {
    if (!this.initialized) {
      console.log(`Message (${level}):`, message);
      return;
    }

    return Sentry.captureMessage(message, level);
  }

  /**
   * Start a performance transaction
   */
  startTransaction(name: string, operation: string = 'navigation'): Sentry.Transaction | undefined {
    if (!this.initialized) return;

    return Sentry.startTransaction({
      name,
      op: operation,
    });
  }

  /**
   * Profile a function for performance monitoring
   */
  withProfiler<T>(name: string, fn: () => T): T {
    if (!this.initialized) {
      return fn();
    }

    const transaction = this.startTransaction(name, 'function');
    try {
      const result = fn();
      transaction?.setStatus('ok');
      return result;
    } catch (error) {
      transaction?.setStatus('internal_error');
      throw error;
    } finally {
      transaction?.finish();
    }
  }

  /**
   * Profile an async function for performance monitoring
   */
  async withProfilerAsync<T>(name: string, fn: () => Promise<T>): Promise<T> {
    if (!this.initialized) {
      return await fn();
    }

    const transaction = this.startTransaction(name, 'async_function');
    try {
      const result = await fn();
      transaction?.setStatus('ok');
      return result;
    } catch (error) {
      transaction?.setStatus('internal_error');
      throw error;
    } finally {
      transaction?.finish();
    }
  }

  /**
   * Flush pending events (useful before page unload)
   */
  async flush(timeout: number = 2000): Promise<boolean> {
    if (!this.initialized) return true;

    return await Sentry.flush(timeout);
  }

  /**
   * Close the Sentry client
   */
  async close(timeout: number = 2000): Promise<boolean> {
    if (!this.initialized) return true;

    const result = await Sentry.close(timeout);
    this.initialized = false;
    return result;
  }
}

// Create singleton instance
const errorMonitoring = new ErrorMonitoring({
  dsn: import.meta.env.VITE_SENTRY_DSN,
  environment: import.meta.env.VITE_APP_ENV || 'development',
  debug: import.meta.env.DEV,
  tracesSampleRate: import.meta.env.PROD ? 0.1 : 1.0,
  replaysSessionSampleRate: import.meta.env.PROD ? 0.1 : 1.0,
  replaysOnErrorSampleRate: 1.0,
});

// Auto-initialize on module load
errorMonitoring.initialize();

// React Error Boundary component with Sentry integration
export const ErrorBoundary = Sentry.withErrorBoundary(
  ({ children }: { children: React.ReactNode }) => children,
  {
    fallback: ({ error, resetError }) => (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="max-w-md w-full bg-white dark:bg-gray-800 shadow-lg rounded-lg p-6">
          <div className="flex items-center justify-center w-12 h-12 mx-auto bg-red-100 rounded-full mb-4">
            <svg
              className="w-6 h-6 text-red-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
              />
            </svg>
          </div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white text-center mb-2">
            Something went wrong
          </h2>
          <p className="text-gray-600 dark:text-gray-300 text-center mb-6">
            We've been notified about this error and will fix it soon.
          </p>
          <div className="flex space-x-3">
            <button
              onClick={resetError}
              className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors"
            >
              Try Again
            </button>
            <button
              onClick={() => window.location.reload()}
              className="flex-1 bg-gray-600 text-white py-2 px-4 rounded-md hover:bg-gray-700 transition-colors"
            >
              Reload Page
            </button>
          </div>
        </div>
      </div>
    ),
    showDialog: false,
  }
);

// Performance monitoring hooks
export const usePerformanceMonitoring = () => {
  return {
    startTransaction: errorMonitoring.startTransaction.bind(errorMonitoring),
    withProfiler: errorMonitoring.withProfiler.bind(errorMonitoring),
    withProfilerAsync: errorMonitoring.withProfilerAsync.bind(errorMonitoring),
    addBreadcrumb: errorMonitoring.addBreadcrumb.bind(errorMonitoring),
    setContext: errorMonitoring.setContext.bind(errorMonitoring),
  };
};

// React hook for error boundary
export const useErrorHandler = () => {
  return {
    captureException: errorMonitoring.captureException.bind(errorMonitoring),
    captureMessage: errorMonitoring.captureMessage.bind(errorMonitoring),
    setUser: errorMonitoring.setUser.bind(errorMonitoring),
    setContext: errorMonitoring.setContext.bind(errorMonitoring),
    addBreadcrumb: errorMonitoring.addBreadcrumb.bind(errorMonitoring),
  };
};

// Higher-order component for profiling
export function withPerformanceMonitoring<P extends object>(
  Component: React.ComponentType<P>,
  componentName: string
) {
  return React.memo((props: P) => {
    return errorMonitoring.withProfiler(`render_${componentName}`, () => (
      <Component {...props} />
    ));
  });
}

export default errorMonitoring;