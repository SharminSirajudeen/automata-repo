/**
 * Advanced Chunk Error Boundary Component
 * Handles chunk loading failures with intelligent retry mechanisms and user feedback
 */
import React, { Component, ErrorInfo, ReactNode, useState, useEffect } from 'react';
import { AlertCircle, RefreshCw, Home, Bug, Wifi, WifiOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';

// Error categorization
enum ErrorType {
  CHUNK_LOAD_ERROR = 'ChunkLoadError',
  NETWORK_ERROR = 'NetworkError',
  TIMEOUT_ERROR = 'TimeoutError',
  SYNTAX_ERROR = 'SyntaxError',
  MODULE_NOT_FOUND = 'ModuleNotFoundError',
  UNKNOWN_ERROR = 'UnknownError'
}

interface ChunkError {
  type: ErrorType;
  message: string;
  stack?: string;
  chunkName?: string;
  timestamp: number;
  retryCount: number;
  userAgent: string;
  url: string;
  networkStatus: 'online' | 'offline';
}

interface ChunkErrorBoundaryState {
  hasError: boolean;
  error?: ChunkError;
  retryAttempts: number;
  isRetrying: boolean;
}

interface ChunkErrorBoundaryProps {
  children: ReactNode;
  fallback?: React.ComponentType<ChunkErrorFallbackProps>;
  onError?: (error: ChunkError, errorInfo: ErrorInfo) => void;
  maxRetries?: number;
  retryDelay?: number;
  showTechnicalDetails?: boolean;
  enableAutoRetry?: boolean;
}

interface ChunkErrorFallbackProps {
  error: ChunkError;
  retry: () => void;
  goHome: () => void;
  isRetrying: boolean;
  canRetry: boolean;
  showTechnicalDetails?: boolean;
}

// Utility functions
const categorizeError = (error: Error): ErrorType => {
  const message = error.message.toLowerCase();
  const stack = error.stack?.toLowerCase() || '';

  if (message.includes('loading chunk') || message.includes('loading css chunk')) {
    return ErrorType.CHUNK_LOAD_ERROR;
  }
  if (message.includes('network') || message.includes('fetch')) {
    return ErrorType.NETWORK_ERROR;
  }
  if (message.includes('timeout')) {
    return ErrorType.TIMEOUT_ERROR;
  }
  if (message.includes('syntax') || stack.includes('syntaxerror')) {
    return ErrorType.SYNTAX_ERROR;
  }
  if (message.includes('module') && message.includes('not found')) {
    return ErrorType.MODULE_NOT_FOUND;
  }
  return ErrorType.UNKNOWN_ERROR;
};

const getNetworkStatus = (): 'online' | 'offline' => {
  return navigator.onLine ? 'online' : 'offline';
};

const extractChunkName = (error: Error): string | undefined => {
  const match = error.message.match(/loading chunk (\w+)/i);
  return match ? match[1] : undefined;
};

const getErrorSeverity = (errorType: ErrorType): 'low' | 'medium' | 'high' => {
  switch (errorType) {
    case ErrorType.CHUNK_LOAD_ERROR:
    case ErrorType.NETWORK_ERROR:
      return 'medium';
    case ErrorType.TIMEOUT_ERROR:
      return 'medium';
    case ErrorType.SYNTAX_ERROR:
    case ErrorType.MODULE_NOT_FOUND:
      return 'high';
    default:
      return 'low';
  }
};

const getErrorIcon = (errorType: ErrorType) => {
  switch (errorType) {
    case ErrorType.NETWORK_ERROR:
      return <WifiOff className="h-6 w-6" />;
    case ErrorType.CHUNK_LOAD_ERROR:
      return <RefreshCw className="h-6 w-6" />;
    default:
      return <AlertCircle className="h-6 w-6" />;
  }
};

const getErrorColor = (errorType: ErrorType): string => {
  const severity = getErrorSeverity(errorType);
  switch (severity) {
    case 'high': return 'destructive';
    case 'medium': return 'destructive';
    case 'low': return 'secondary';
    default: return 'destructive';
  }
};

const canRetryError = (errorType: ErrorType): boolean => {
  return [
    ErrorType.CHUNK_LOAD_ERROR,
    ErrorType.NETWORK_ERROR,
    ErrorType.TIMEOUT_ERROR
  ].includes(errorType);
};

// Network status hook
const useNetworkStatus = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [connectionType, setConnectionType] = useState<string>('unknown');

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Get connection info if available
    const connection = (navigator as any).connection;
    if (connection) {
      setConnectionType(connection.effectiveType || 'unknown');
      
      const handleConnectionChange = () => {
        setConnectionType(connection.effectiveType || 'unknown');
      };
      
      connection.addEventListener('change', handleConnectionChange);
      
      return () => {
        window.removeEventListener('online', handleOnline);
        window.removeEventListener('offline', handleOffline);
        connection.removeEventListener('change', handleConnectionChange);
      };
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return { isOnline, connectionType };
};

// Advanced Error Fallback Component
function AdvancedChunkErrorFallback({
  error,
  retry,
  goHome,
  isRetrying,
  canRetry,
  showTechnicalDetails = false
}: ChunkErrorFallbackProps) {
  const [expanded, setExpanded] = useState(false);
  const { isOnline, connectionType } = useNetworkStatus();
  
  const severity = getErrorSeverity(error.type);
  const errorColor = getErrorColor(error.type);
  const errorIcon = getErrorIcon(error.type);

  const getErrorTitle = () => {
    switch (error.type) {
      case ErrorType.CHUNK_LOAD_ERROR:
        return 'Component Loading Failed';
      case ErrorType.NETWORK_ERROR:
        return 'Network Connection Error';
      case ErrorType.TIMEOUT_ERROR:
        return 'Request Timeout';
      case ErrorType.SYNTAX_ERROR:
        return 'Code Syntax Error';
      case ErrorType.MODULE_NOT_FOUND:
        return 'Module Not Found';
      default:
        return 'Unexpected Error';
    }
  };

  const getErrorDescription = () => {
    switch (error.type) {
      case ErrorType.CHUNK_LOAD_ERROR:
        return 'A component failed to load. This might be due to a network issue or the component being temporarily unavailable.';
      case ErrorType.NETWORK_ERROR:
        return 'Unable to establish a network connection. Please check your internet connection and try again.';
      case ErrorType.TIMEOUT_ERROR:
        return 'The request took too long to complete. This might be due to a slow network connection.';
      case ErrorType.SYNTAX_ERROR:
        return 'There\'s a code error that prevents the component from loading properly. This requires developer attention.';
      case ErrorType.MODULE_NOT_FOUND:
        return 'A required component or module could not be found. This might be a configuration issue.';
      default:
        return 'An unexpected error occurred while loading the component.';
    }
  };

  const getSuggestions = (): string[] => {
    switch (error.type) {
      case ErrorType.CHUNK_LOAD_ERROR:
        return [
          'Check your internet connection',
          'Try refreshing the page',
          'Clear your browser cache',
          'Try again in a few minutes'
        ];
      case ErrorType.NETWORK_ERROR:
        return [
          'Verify your internet connection',
          'Check if you\'re behind a firewall',
          'Try using a different network',
          'Contact your network administrator'
        ];
      case ErrorType.TIMEOUT_ERROR:
        return [
          'Check your connection speed',
          'Try again with a better connection',
          'Close other bandwidth-heavy applications'
        ];
      default:
        return [
          'Refresh the page',
          'Try again later',
          'Contact support if the issue persists'
        ];
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-background">
      <Card className="w-full max-w-2xl">
        <CardHeader className="text-center">
          <div className={`inline-flex items-center justify-center w-16 h-16 mx-auto mb-4 rounded-full bg-${errorColor}/10`}>
            <div className={`text-${errorColor}`}>
              {errorIcon}
            </div>
          </div>
          
          <CardTitle className="text-2xl font-bold">
            {getErrorTitle()}
          </CardTitle>
          
          <CardDescription className="text-base mt-2">
            {getErrorDescription()}
          </CardDescription>
          
          <div className="flex items-center justify-center gap-2 mt-4">
            <Badge variant="outline">
              {error.type.replace(/([A-Z])/g, ' $1').trim()}
            </Badge>
            
            <Badge variant={isOnline ? 'default' : 'destructive'}>
              {isOnline ? <Wifi className="w-3 h-3 mr-1" /> : <WifiOff className="w-3 h-3 mr-1" />}
              {isOnline ? 'Online' : 'Offline'}
            </Badge>
            
            {connectionType !== 'unknown' && (
              <Badge variant="secondary">
                {connectionType.toUpperCase()}
              </Badge>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            {canRetry && (
              <Button
                onClick={retry}
                disabled={isRetrying || !isOnline}
                className="flex items-center gap-2"
                size="lg"
              >
                <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
                {isRetrying ? 'Retrying...' : 'Try Again'}
              </Button>
            )}
            
            <Button
              onClick={goHome}
              variant="outline"
              size="lg"
              className="flex items-center gap-2"
            >
              <Home className="w-4 h-4" />
              Go Home
            </Button>
          </div>

          {/* Suggestions */}
          <Alert>
            <Bug className="h-4 w-4" />
            <AlertTitle>Troubleshooting Steps</AlertTitle>
            <AlertDescription>
              <ul className="list-disc list-inside mt-2 space-y-1">
                {getSuggestions().map((suggestion, index) => (
                  <li key={index} className="text-sm">{suggestion}</li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>

          {/* Retry Information */}
          {error.retryCount > 0 && (
            <div className="text-center text-sm text-muted-foreground">
              Attempted {error.retryCount} time{error.retryCount !== 1 ? 's' : ''}
            </div>
          )}

          {/* Technical Details */}
          {showTechnicalDetails && (
            <Collapsible open={expanded} onOpenChange={setExpanded}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full justify-between">
                  Technical Details
                  <span className={`transition-transform ${expanded ? 'rotate-180' : ''}`}>
                    ▼
                  </span>
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-3">
                <div className="p-4 bg-muted rounded-lg space-y-2 text-sm font-mono">
                  <div><strong>Error:</strong> {error.message}</div>
                  <div><strong>Type:</strong> {error.type}</div>
                  {error.chunkName && (
                    <div><strong>Chunk:</strong> {error.chunkName}</div>
                  )}
                  <div><strong>Time:</strong> {new Date(error.timestamp).toLocaleString()}</div>
                  <div><strong>URL:</strong> {error.url}</div>
                  <div><strong>User Agent:</strong> {error.userAgent}</div>
                  
                  {error.stack && (
                    <details className="mt-2">
                      <summary className="cursor-pointer hover:text-primary">
                        Stack Trace
                      </summary>
                      <pre className="mt-2 p-2 bg-background rounded text-xs overflow-auto">
                        {error.stack}
                      </pre>
                    </details>
                  )}
                </div>
              </CollapsibleContent>
            </Collapsible>
          )}

          {/* Contact Information */}
          <div className="text-center text-sm text-muted-foreground">
            If this problem persists, please contact support with the error details above.
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Main Error Boundary Class
export class ChunkErrorBoundary extends Component<ChunkErrorBoundaryProps, ChunkErrorBoundaryState> {
  private retryTimeoutId?: NodeJS.Timeout;
  
  constructor(props: ChunkErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      retryAttempts: 0,
      isRetrying: false
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ChunkErrorBoundaryState> {
    const chunkError: ChunkError = {
      type: categorizeError(error),
      message: error.message,
      stack: error.stack,
      chunkName: extractChunkName(error),
      timestamp: Date.now(),
      retryCount: 0,
      userAgent: navigator.userAgent,
      url: window.location.href,
      networkStatus: getNetworkStatus()
    };

    return {
      hasError: true,
      error: chunkError
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ChunkErrorBoundary caught an error:', error, errorInfo);
    
    // Report error to monitoring service
    this.reportError(error, errorInfo);
    
    // Call onError callback
    if (this.props.onError && this.state.error) {
      this.props.onError(this.state.error, errorInfo);
    }

    // Auto-retry for certain error types
    if (this.props.enableAutoRetry && this.state.error && canRetryError(this.state.error.type)) {
      this.scheduleRetry();
    }
  }

  componentWillUnmount() {
    if (this.retryTimeoutId) {
      clearTimeout(this.retryTimeoutId);
    }
  }

  private reportError = (error: Error, errorInfo: ErrorInfo) => {
    // Send error report to monitoring service (e.g., Sentry, LogRocket, etc.)
    const errorReport = {
      error: {
        message: error.message,
        stack: error.stack,
        name: error.name
      },
      errorInfo: {
        componentStack: errorInfo.componentStack
      },
      context: {
        url: window.location.href,
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString(),
        networkStatus: navigator.onLine ? 'online' : 'offline'
      }
    };

    // Example: Send to your error tracking service
    console.warn('Error Report:', errorReport);
    
    // In production, you would send this to your error tracking service:
    // errorTrackingService.captureException(error, { extra: errorReport });
  };

  private scheduleRetry = () => {
    const { retryDelay = 3000, maxRetries = 3 } = this.props;
    
    if (this.state.retryAttempts < maxRetries) {
      this.retryTimeoutId = setTimeout(() => {
        this.handleRetry();
      }, retryDelay * Math.pow(2, this.state.retryAttempts)); // Exponential backoff
    }
  };

  private handleRetry = () => {
    const { maxRetries = 3 } = this.props;
    
    if (this.state.retryAttempts >= maxRetries) {
      console.warn('Maximum retry attempts reached');
      return;
    }

    this.setState(prevState => ({
      isRetrying: true,
      retryAttempts: prevState.retryAttempts + 1,
      error: prevState.error ? {
        ...prevState.error,
        retryCount: prevState.retryAttempts + 1
      } : undefined
    }));

    // Clear the error state after a brief delay to trigger re-render
    setTimeout(() => {
      this.setState({
        hasError: false,
        error: undefined,
        isRetrying: false
      });
    }, 1000);
  };

  private handleGoHome = () => {
    // Navigate to home page
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError && this.state.error) {
      const FallbackComponent = this.props.fallback || AdvancedChunkErrorFallback;
      const { maxRetries = 3 } = this.props;
      
      return (
        <FallbackComponent
          error={this.state.error}
          retry={this.handleRetry}
          goHome={this.handleGoHome}
          isRetrying={this.state.isRetrying}
          canRetry={canRetryError(this.state.error.type) && this.state.retryAttempts < maxRetries}
          showTechnicalDetails={this.props.showTechnicalDetails}
        />
      );
    }

    return this.props.children;
  }
}

// Export types and utilities
export type { ChunkError, ChunkErrorBoundaryProps, ChunkErrorFallbackProps };
export { ErrorType, categorizeError, canRetryError, getErrorSeverity };

// Default export
export default ChunkErrorBoundary;