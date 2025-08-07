/**
 * Performance Monitoring and Bundle Optimization Analytics
 * Tracks loading metrics, chunk performance, and user experience
 */

interface PerformanceMetrics {
  // Bundle metrics
  bundleSize: number;
  gzippedSize: number;
  chunkCount: number;
  
  // Loading metrics
  initialLoadTime: number;
  timeToInteractive: number;
  firstContentfulPaint: number;
  largestContentfulPaint: number;
  
  // Chunk loading metrics
  chunkLoadTimes: Map<string, number>;
  chunkErrors: Map<string, number>;
  
  // User experience metrics
  cumulativeLayoutShift: number;
  firstInputDelay: number;
  
  // Network metrics
  connectionType: string;
  effectiveType: string;
  downlink: number;
  rtt: number;
}

class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: Partial<PerformanceMetrics> = {};
  private observers: PerformanceObserver[] = [];
  private startTime: number = performance.now();
  
  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }
  
  constructor() {
    this.initializeObservers();
    this.trackNetworkInformation();
    this.trackResourceLoadTimes();
  }
  
  private initializeObservers() {
    // Web Vitals Observer
    if ('PerformanceObserver' in window) {
      // Largest Contentful Paint
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1] as PerformanceEntry & { startTime: number };
        this.metrics.largestContentfulPaint = lastEntry.startTime;
      });
      
      try {
        lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
        this.observers.push(lcpObserver);
      } catch (e) {
        console.warn('LCP observer not supported:', e);
      }
      
      // First Input Delay
      const fidObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          this.metrics.firstInputDelay = entry.processingStart - entry.startTime;
        });
      });
      
      try {
        fidObserver.observe({ entryTypes: ['first-input'] });
        this.observers.push(fidObserver);
      } catch (e) {
        console.warn('FID observer not supported:', e);
      }
      
      // Cumulative Layout Shift
      let clsValue = 0;
      const clsObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries() as any[]) {
          if (!entry.hadRecentInput) {
            clsValue += entry.value;
            this.metrics.cumulativeLayoutShift = clsValue;
          }
        }
      });
      
      try {
        clsObserver.observe({ entryTypes: ['layout-shift'] });
        this.observers.push(clsObserver);
      } catch (e) {
        console.warn('CLS observer not supported:', e);
      }
      
      // Resource timing for chunks
      const resourceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry: PerformanceEntry) => {
          if (entry.name.includes('chunk') || entry.name.includes('.js')) {
            const loadTime = entry.duration || 0;
            const chunkName = this.extractChunkName(entry.name);
            
            if (!this.metrics.chunkLoadTimes) {
              this.metrics.chunkLoadTimes = new Map();
            }
            
            this.metrics.chunkLoadTimes.set(chunkName, loadTime);
          }
        });
      });
      
      try {
        resourceObserver.observe({ entryTypes: ['resource'] });
        this.observers.push(resourceObserver);
      } catch (e) {
        console.warn('Resource observer not supported:', e);
      }
    }
  }
  
  private trackNetworkInformation() {
    const connection = (navigator as any).connection;
    if (connection) {
      this.metrics.connectionType = connection.type || 'unknown';
      this.metrics.effectiveType = connection.effectiveType || 'unknown';
      this.metrics.downlink = connection.downlink || 0;
      this.metrics.rtt = connection.rtt || 0;
      
      connection.addEventListener('change', () => {
        this.metrics.connectionType = connection.type || 'unknown';
        this.metrics.effectiveType = connection.effectiveType || 'unknown';
        this.metrics.downlink = connection.downlink || 0;
        this.metrics.rtt = connection.rtt || 0;
      });
    }
  }
  
  private trackResourceLoadTimes() {
    // Track critical resource load times
    window.addEventListener('load', () => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      
      this.metrics.initialLoadTime = navigation.loadEventEnd - navigation.navigationStart;
      this.metrics.timeToInteractive = navigation.domInteractive - navigation.navigationStart;
      this.metrics.firstContentfulPaint = this.getFirstContentfulPaint();
      
      this.logPerformanceMetrics();
    });
  }
  
  private getFirstContentfulPaint(): number {
    const paintEntries = performance.getEntriesByType('paint');
    const fcpEntry = paintEntries.find(entry => entry.name === 'first-contentful-paint');
    return fcpEntry ? fcpEntry.startTime : 0;
  }
  
  private extractChunkName(url: string): string {
    const match = url.match(/([^/]+)\.chunk\.js$/);
    return match ? match[1] : url.split('/').pop() || 'unknown';
  }
  
  // Public methods
  recordChunkError(chunkName: string) {
    if (!this.metrics.chunkErrors) {
      this.metrics.chunkErrors = new Map();
    }
    
    const currentCount = this.metrics.chunkErrors.get(chunkName) || 0;
    this.metrics.chunkErrors.set(chunkName, currentCount + 1);
  }
  
  recordCustomMetric(name: string, value: number) {
    performance.mark(`custom-${name}`, { detail: { value } });
  }
  
  measureCodeSplitEffectiveness(): {
    totalChunks: number;
    averageChunkSize: number;
    largestChunk: { name: string; size: number } | null;
    cachingEffectiveness: number;
  } {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    const jsResources = resources.filter(r => r.name.endsWith('.js'));
    
    let totalSize = 0;
    let largestChunk: { name: string; size: number } | null = null;
    let cachedResources = 0;
    
    jsResources.forEach(resource => {
      const size = resource.transferSize || resource.encodedBodySize || 0;
      totalSize += size;
      
      if (!largestChunk || size > largestChunk.size) {
        largestChunk = {
          name: this.extractChunkName(resource.name),
          size
        };
      }
      
      // Check if resource was served from cache
      if (resource.transferSize === 0 && resource.decodedBodySize > 0) {
        cachedResources++;
      }
    });
    
    return {
      totalChunks: jsResources.length,
      averageChunkSize: jsResources.length ? totalSize / jsResources.length : 0,
      largestChunk,
      cachingEffectiveness: jsResources.length ? cachedResources / jsResources.length : 0
    };
  }
  
  getBundleOptimizationReport(): {
    metrics: Partial<PerformanceMetrics>;
    recommendations: string[];
    score: number;
  } {
    const recommendations: string[] = [];
    let score = 100;
    
    // Analyze LCP
    if (this.metrics.largestContentfulPaint && this.metrics.largestContentfulPaint > 2500) {
      recommendations.push('Improve Largest Contentful Paint: Consider optimizing images and critical resource loading');
      score -= 15;
    }
    
    // Analyze FID
    if (this.metrics.firstInputDelay && this.metrics.firstInputDelay > 100) {
      recommendations.push('Reduce First Input Delay: Minimize JavaScript execution time');
      score -= 10;
    }
    
    // Analyze CLS
    if (this.metrics.cumulativeLayoutShift && this.metrics.cumulativeLayoutShift > 0.1) {
      recommendations.push('Reduce Cumulative Layout Shift: Ensure proper sizing for dynamic content');
      score -= 10;
    }
    
    // Analyze bundle size
    const splitEffectiveness = this.measureCodeSplitEffectiveness();
    if (splitEffectiveness.averageChunkSize > 200000) { // 200KB
      recommendations.push('Reduce chunk size: Consider further code splitting for large chunks');
      score -= 15;
    }
    
    // Analyze caching
    if (splitEffectiveness.cachingEffectiveness < 0.5) {
      recommendations.push('Improve caching: Optimize cache headers and chunk naming strategy');
      score -= 10;
    }
    
    // Check chunk errors
    if (this.metrics.chunkErrors && this.metrics.chunkErrors.size > 0) {
      recommendations.push('Fix chunk loading errors: Implement better error handling and retry logic');
      score -= 20;
    }
    
    return {
      metrics: this.metrics,
      recommendations,
      score: Math.max(0, score)
    };
  }
  
  exportPerformanceData(): string {
    const report = this.getBundleOptimizationReport();
    const splitEffectiveness = this.measureCodeSplitEffectiveness();
    
    return JSON.stringify({
      timestamp: new Date().toISOString(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      metrics: report.metrics,
      splitEffectiveness,
      recommendations: report.recommendations,
      score: report.score,
      vitals: {
        lcp: this.metrics.largestContentfulPaint,
        fid: this.metrics.firstInputDelay,
        cls: this.metrics.cumulativeLayoutShift,
        fcp: this.metrics.firstContentfulPaint,
        tti: this.metrics.timeToInteractive
      }
    }, null, 2);
  }
  
  private logPerformanceMetrics() {
    console.group('🚀 Performance Metrics');
    console.log('📊 Bundle Analysis:', this.measureCodeSplitEffectiveness());
    console.log('⚡ Web Vitals:', {
      LCP: this.metrics.largestContentfulPaint,
      FID: this.metrics.firstInputDelay,
      CLS: this.metrics.cumulativeLayoutShift,
      FCP: this.metrics.firstContentfulPaint,
      TTI: this.metrics.timeToInteractive
    });
    console.log('🌐 Network:', {
      connection: this.metrics.connectionType,
      effective: this.metrics.effectiveType,
      downlink: this.metrics.downlink,
      rtt: this.metrics.rtt
    });
    
    const report = this.getBundleOptimizationReport();
    console.log(`📈 Optimization Score: ${report.score}/100`);
    
    if (report.recommendations.length > 0) {
      console.group('💡 Recommendations');
      report.recommendations.forEach(rec => console.log(`• ${rec}`));
      console.groupEnd();
    }
    
    console.groupEnd();
  }
  
  destroy() {
    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];
  }
}

// React hook for performance monitoring
export function usePerformanceMonitor() {
  const monitor = PerformanceMonitor.getInstance();
  
  return {
    recordChunkError: monitor.recordChunkError.bind(monitor),
    recordCustomMetric: monitor.recordCustomMetric.bind(monitor),
    getReport: monitor.getBundleOptimizationReport.bind(monitor),
    exportData: monitor.exportPerformanceData.bind(monitor)
  };
}

// Global performance monitor instance
export const performanceMonitor = PerformanceMonitor.getInstance();

// Cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    performanceMonitor.destroy();
  });
}

export default PerformanceMonitor;