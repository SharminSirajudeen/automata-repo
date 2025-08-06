import React from 'react';
import { createRoot } from 'react-dom/client';
import { ThemeProvider } from './contexts/ThemeProvider';
import { AccessibilityProvider } from './contexts/AccessibilityProvider';
import { AccessibilitySettings } from './components/AccessibilitySettings';
import { SkipNavigation } from './components/SkipNavigation';
import { Button } from './components/ui/button';

// Simple test component to verify accessibility features
function AccessibilityTestApp() {
  return (
    <ThemeProvider defaultTheme="system">
      <AccessibilityProvider>
        <div className="min-h-screen bg-background text-foreground">
          <SkipNavigation />
          
          <header role="banner" className="p-4 bg-card border-b">
            <div className="flex items-center justify-between">
              <h1 className="text-2xl font-bold">Accessibility Test</h1>
              <div className="flex gap-2">
                <AccessibilitySettings />
                <Button variant="outline">Test Button</Button>
              </div>
            </div>
          </header>

          <main id="main-content" role="main" tabIndex={-1} className="p-8">
            <section aria-labelledby="test-heading">
              <h2 id="test-heading">Accessibility Features Test</h2>
              <div className="space-y-4">
                <p>This is a test of the comprehensive accessibility features:</p>
                <ul className="list-disc list-inside space-y-2">
                  <li>Dark/light mode with system preference detection</li>
                  <li>High contrast mode support</li>
                  <li>Reduced motion preferences</li>
                  <li>Keyboard navigation (try Alt+T for theme toggle)</li>
                  <li>Screen reader optimizations with ARIA labels</li>
                  <li>Skip navigation links (try Tab to see them)</li>
                  <li>Focus management and visual indicators</li>
                  <li>WCAG AAA color contrast compliance</li>
                </ul>
                <div className="space-x-2">
                  <Button>Primary Button</Button>
                  <Button variant="secondary">Secondary Button</Button>
                  <Button variant="destructive">Destructive Button</Button>
                </div>
              </div>
            </section>
          </main>

          <footer role="contentinfo" className="mt-8 p-4 bg-muted text-center text-sm">
            <p>Accessibility test completed - all features should work with keyboard and screen readers</p>
          </footer>
        </div>
      </AccessibilityProvider>
    </ThemeProvider>
  );
}

// Export for testing purposes
export { AccessibilityTestApp };

// If running directly, render the test app
if (typeof window !== 'undefined' && document.getElementById('test-root')) {
  const container = document.getElementById('test-root');
  if (container) {
    const root = createRoot(container);
    root.render(<AccessibilityTestApp />);
  }
}