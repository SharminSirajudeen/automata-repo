import React, { createContext, useContext, useEffect, useState, useRef, useCallback } from 'react';

interface KeyboardShortcut {
  key: string;
  ctrlKey?: boolean;
  altKey?: boolean;
  shiftKey?: boolean;
  metaKey?: boolean;
  action: string;
  description: string;
  handler: () => void;
}

interface AccessibilityContextType {
  // Screen reader
  announceMessage: (message: string, priority?: 'polite' | 'assertive') => void;
  
  // Focus management
  focusElement: (selector: string) => void;
  focusTrap: (element: HTMLElement | null) => void;
  releaseFocusTrap: () => void;
  skipToContent: () => void;
  
  // Keyboard shortcuts
  shortcuts: KeyboardShortcut[];
  addShortcut: (shortcut: KeyboardShortcut) => void;
  removeShortcut: (action: string) => void;
  
  // Settings
  keyboardNavigation: boolean;
  setKeyboardNavigation: (enabled: boolean) => void;
  screenReaderOptimizations: boolean;
  setScreenReaderOptimizations: (enabled: boolean) => void;
  announcements: boolean;
  setAnnouncements: (enabled: boolean) => void;
  
  // Font size
  fontSize: 'small' | 'medium' | 'large' | 'extra-large';
  setFontSize: (size: 'small' | 'medium' | 'large' | 'extra-large') => void;
  
  // Current focus
  currentFocusIndex: number;
  focusableElements: HTMLElement[];
  updateFocusableElements: () => void;
}

const AccessibilityContext = createContext<AccessibilityContextType | undefined>(undefined);

interface AccessibilityProviderProps {
  children: React.ReactNode;
}

export function AccessibilityProvider({ children }: AccessibilityProviderProps) {
  const [shortcuts, setShortcuts] = useState<KeyboardShortcut[]>([]);
  const [keyboardNavigation, setKeyboardNavigationState] = useState(() => {
    return localStorage.getItem('keyboard-navigation') !== 'false';
  });
  const [screenReaderOptimizations, setScreenReaderOptimizationsState] = useState(() => {
    return localStorage.getItem('screen-reader-optimizations') === 'true';
  });
  const [announcements, setAnnouncementsState] = useState(() => {
    return localStorage.getItem('announcements') !== 'false';
  });
  const [fontSize, setFontSizeState] = useState<'small' | 'medium' | 'large' | 'extra-large'>(() => {
    return (localStorage.getItem('font-size') as any) || 'medium';
  });

  const [currentFocusIndex, setCurrentFocusIndex] = useState(-1);
  const [focusableElements, setFocusableElements] = useState<HTMLElement[]>([]);

  const announcementRef = useRef<HTMLDivElement>(null);
  const focusTrapRef = useRef<HTMLElement | null>(null);
  const focusHistoryRef = useRef<HTMLElement[]>([]);

  // Update focusable elements
  const updateFocusableElements = useCallback(() => {
    const focusableSelectors = [
      'a[href]',
      'button:not([disabled])',
      'textarea:not([disabled])',
      'input:not([disabled]):not([type="hidden"])',
      'select:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
      '[contenteditable="true"]'
    ].join(', ');

    const elements = Array.from(document.querySelectorAll(focusableSelectors)) as HTMLElement[];
    setFocusableElements(elements.filter(el => {
      const style = getComputedStyle(el);
      return style.display !== 'none' && style.visibility !== 'hidden';
    }));
  }, []);

  // Screen reader announcements
  const announceMessage = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    if (!announcements || !announcementRef.current) return;
    
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', priority);
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    
    announcementRef.current.appendChild(announcement);
    
    // Clean up after announcement
    setTimeout(() => {
      if (announcementRef.current && announcement.parentNode) {
        announcementRef.current.removeChild(announcement);
      }
    }, 1000);
  }, [announcements]);

  // Focus management
  const focusElement = useCallback((selector: string) => {
    const element = document.querySelector(selector) as HTMLElement;
    if (element) {
      element.focus();
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, []);

  const skipToContent = useCallback(() => {
    const mainContent = document.querySelector('main') || document.querySelector('#main-content') || document.querySelector('[role="main"]');
    if (mainContent) {
      (mainContent as HTMLElement).focus();
      (mainContent as HTMLElement).scrollIntoView({ behavior: 'smooth' });
      announceMessage('Skipped to main content');
    }
  }, [announceMessage]);

  const focusTrap = useCallback((element: HTMLElement | null) => {
    focusTrapRef.current = element;
    if (element) {
      const focusableElements = element.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      ) as NodeListOf<HTMLElement>;
      
      if (focusableElements.length > 0) {
        focusableElements[0].focus();
      }
    }
  }, []);

  const releaseFocusTrap = useCallback(() => {
    focusTrapRef.current = null;
  }, []);

  // Keyboard shortcuts
  const addShortcut = useCallback((shortcut: KeyboardShortcut) => {
    setShortcuts(prev => [...prev.filter(s => s.action !== shortcut.action), shortcut]);
  }, []);

  const removeShortcut = useCallback((action: string) => {
    setShortcuts(prev => prev.filter(s => s.action !== action));
  }, []);

  // Settings handlers
  const setKeyboardNavigation = useCallback((enabled: boolean) => {
    localStorage.setItem('keyboard-navigation', enabled.toString());
    setKeyboardNavigationState(enabled);
    announceMessage(`Keyboard navigation ${enabled ? 'enabled' : 'disabled'}`);
  }, [announceMessage]);

  const setScreenReaderOptimizations = useCallback((enabled: boolean) => {
    localStorage.setItem('screen-reader-optimizations', enabled.toString());
    setScreenReaderOptimizationsState(enabled);
    announceMessage(`Screen reader optimizations ${enabled ? 'enabled' : 'disabled'}`);
  }, [announceMessage]);

  const setAnnouncements = useCallback((enabled: boolean) => {
    localStorage.setItem('announcements', enabled.toString());
    setAnnouncementsState(enabled);
    if (enabled) {
      announceMessage('Announcements enabled');
    }
  }, [announceMessage]);

  const setFontSize = useCallback((size: 'small' | 'medium' | 'large' | 'extra-large') => {
    localStorage.setItem('font-size', size);
    setFontSizeState(size);
    announceMessage(`Font size changed to ${size}`);
    
    // Apply font size to document
    const root = document.documentElement;
    root.classList.remove('text-sm', 'text-base', 'text-lg', 'text-xl');
    
    switch (size) {
      case 'small':
        root.classList.add('text-sm');
        break;
      case 'medium':
        root.classList.add('text-base');
        break;
      case 'large':
        root.classList.add('text-lg');
        break;
      case 'extra-large':
        root.classList.add('text-xl');
        break;
    }
  }, [announceMessage]);

  // Default shortcuts
  useEffect(() => {
    const defaultShortcuts: KeyboardShortcut[] = [
      {
        key: '/',
        action: 'search',
        description: 'Focus search input',
        handler: () => focusElement('input[type="search"], input[placeholder*="search" i]')
      },
      {
        key: 'h',
        action: 'home',
        description: 'Go to home',
        handler: () => window.location.href = '/'
      },
      {
        key: '?',
        shiftKey: true,
        action: 'help',
        description: 'Show keyboard shortcuts',
        handler: () => announceMessage('Press Alt+K for keyboard shortcuts help')
      },
      {
        key: 's',
        action: 'skip-content',
        description: 'Skip to main content',
        handler: skipToContent
      },
      {
        key: 't',
        altKey: true,
        action: 'toggle-theme',
        description: 'Toggle theme',
        handler: () => {
          const event = new CustomEvent('toggle-theme');
          window.dispatchEvent(event);
        }
      }
    ];

    defaultShortcuts.forEach(addShortcut);
  }, [addShortcut, focusElement, skipToContent, announceMessage]);

  // Keyboard event handler
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!keyboardNavigation) return;

      const matchingShortcut = shortcuts.find(shortcut => {
        return shortcut.key.toLowerCase() === event.key.toLowerCase() &&
               !!shortcut.ctrlKey === event.ctrlKey &&
               !!shortcut.altKey === event.altKey &&
               !!shortcut.shiftKey === event.shiftKey &&
               !!shortcut.metaKey === event.metaKey;
      });

      if (matchingShortcut) {
        event.preventDefault();
        matchingShortcut.handler();
        announceMessage(`Executed: ${matchingShortcut.description}`);
        return;
      }

      // Handle focus trap
      if (focusTrapRef.current && event.key === 'Tab') {
        const focusableElements = focusTrapRef.current.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        ) as NodeListOf<HTMLElement>;
        
        if (focusableElements.length > 0) {
          const first = focusableElements[0];
          const last = focusableElements[focusableElements.length - 1];
          
          if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
          } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
          }
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [keyboardNavigation, shortcuts, announceMessage]);

  // Update focusable elements on mount and DOM changes
  useEffect(() => {
    updateFocusableElements();
    
    const observer = new MutationObserver(updateFocusableElements);
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['disabled', 'tabindex', 'aria-hidden']
    });

    return () => observer.disconnect();
  }, [updateFocusableElements]);

  // Apply font size on mount
  useEffect(() => {
    setFontSize(fontSize);
  }, []);

  const value: AccessibilityContextType = {
    announceMessage,
    focusElement,
    focusTrap,
    releaseFocusTrap,
    skipToContent,
    shortcuts,
    addShortcut,
    removeShortcut,
    keyboardNavigation,
    setKeyboardNavigation,
    screenReaderOptimizations,
    setScreenReaderOptimizations,
    announcements,
    setAnnouncements,
    fontSize,
    setFontSize,
    currentFocusIndex,
    focusableElements,
    updateFocusableElements
  };

  return (
    <AccessibilityContext.Provider value={value}>
      {children}
      {/* Announcement region for screen readers */}
      <div 
        ref={announcementRef}
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      />
    </AccessibilityContext.Provider>
  );
}

export const useAccessibility = () => {
  const context = useContext(AccessibilityContext);
  if (context === undefined) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider');
  }
  return context;
};