import { useCallback, useEffect, useState, useRef } from 'react';

export interface AccessibilitySettings {
  highContrast: boolean;
  reducedMotion: boolean;
  screenReaderOptimized: boolean;
  keyboardNavigation: boolean;
  focusVisible: boolean;
  largeText: boolean;
  audioDescriptions: boolean;
}

export interface KeyboardShortcut {
  key: string;
  ctrlKey?: boolean;
  altKey?: boolean;
  shiftKey?: boolean;
  description: string;
  action: () => void;
}

export const useAccessibility = () => {
  const [settings, setSettings] = useState<AccessibilitySettings>({
    highContrast: false,
    reducedMotion: false,
    screenReaderOptimized: false,
    keyboardNavigation: true,
    focusVisible: true,
    largeText: false,
    audioDescriptions: false
  });

  const [announcements, setAnnouncements] = useState<string[]>([]);
  const [keyboardShortcuts, setKeyboardShortcuts] = useState<KeyboardShortcut[]>([]);
  const announcementTimeoutRef = useRef<NodeJS.Timeout>();

  // Detect user preferences from system
  useEffect(() => {
    const detectPreferences = () => {
      const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      const highContrast = window.matchMedia('(prefers-contrast: high)').matches;
      const largeFonts = window.matchMedia('(prefers-font-size: large)').matches;

      setSettings(prev => ({
        ...prev,
        reducedMotion,
        highContrast,
        largeText: largeFonts
      }));
    };

    detectPreferences();

    // Listen for changes
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const contrastQuery = window.matchMedia('(prefers-contrast: high)');
    
    const handleMotionChange = (e: MediaQueryListEvent) => {
      setSettings(prev => ({ ...prev, reducedMotion: e.matches }));
    };
    
    const handleContrastChange = (e: MediaQueryListEvent) => {
      setSettings(prev => ({ ...prev, highContrast: e.matches }));
    };

    motionQuery.addEventListener('change', handleMotionChange);
    contrastQuery.addEventListener('change', handleContrastChange);

    return () => {
      motionQuery.removeEventListener('change', handleMotionChange);
      contrastQuery.removeEventListener('change', handleContrastChange);
    };
  }, []);

  // Screen reader announcements
  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    setAnnouncements(prev => [...prev, message]);
    
    // Create ARIA live region for screen readers
    const liveRegion = document.createElement('div');
    liveRegion.setAttribute('aria-live', priority);
    liveRegion.setAttribute('aria-atomic', 'true');
    liveRegion.className = 'sr-only';
    liveRegion.textContent = message;
    
    document.body.appendChild(liveRegion);
    
    // Clean up after announcement
    if (announcementTimeoutRef.current) {
      clearTimeout(announcementTimeoutRef.current);
    }
    
    announcementTimeoutRef.current = setTimeout(() => {
      document.body.removeChild(liveRegion);
      setAnnouncements(prev => prev.slice(1));
    }, 3000);
  }, []);

  // Keyboard navigation support
  const registerShortcut = useCallback((shortcut: KeyboardShortcut) => {
    setKeyboardShortcuts(prev => [...prev, shortcut]);
  }, []);

  const unregisterShortcut = useCallback((key: string) => {
    setKeyboardShortcuts(prev => prev.filter(s => s.key !== key));
  }, []);

  // Handle keyboard events
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!settings.keyboardNavigation) return;

      const matchingShortcut = keyboardShortcuts.find(shortcut => {
        const keyMatch = shortcut.key.toLowerCase() === event.key.toLowerCase();
        const ctrlMatch = (shortcut.ctrlKey ?? false) === event.ctrlKey;
        const altMatch = (shortcut.altKey ?? false) === event.altKey;
        const shiftMatch = (shortcut.shiftKey ?? false) === event.shiftKey;
        
        return keyMatch && ctrlMatch && altMatch && shiftMatch;
      });

      if (matchingShortcut) {
        event.preventDefault();
        matchingShortcut.action();
        announce(`Keyboard shortcut activated: ${matchingShortcut.description}`);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [keyboardShortcuts, settings.keyboardNavigation, announce]);

  // Focus management
  const manageFocus = useCallback((element: HTMLElement | null, announceText?: string) => {
    if (!element || !settings.keyboardNavigation) return;

    // Ensure element is focusable
    if (!element.hasAttribute('tabindex') && !['button', 'input', 'select', 'textarea', 'a'].includes(element.tagName.toLowerCase())) {
      element.setAttribute('tabindex', '-1');
    }

    element.focus();

    if (announceText) {
      announce(announceText);
    }
  }, [settings.keyboardNavigation, announce]);

  // ARIA label helpers
  const getAriaLabel = useCallback((baseLabel: string, context?: string) => {
    if (!context) return baseLabel;
    return `${baseLabel}. ${context}`;
  }, []);

  const getAriaDescribedBy = useCallback((elementId: string, descriptions: string[]) => {
    if (descriptions.length === 0) return undefined;
    
    // Create description elements
    descriptions.forEach((description, index) => {
      const descId = `${elementId}-desc-${index}`;
      const existing = document.getElementById(descId);
      
      if (!existing) {
        const descElement = document.createElement('div');
        descElement.id = descId;
        descElement.className = 'sr-only';
        descElement.textContent = description;
        document.body.appendChild(descElement);
      }
    });

    return descriptions.map((_, index) => `${elementId}-desc-${index}`).join(' ');
  }, []);

  // Color contrast helpers
  const getContrastClass = useCallback((level: 'normal' | 'high' = 'normal') => {
    if (!settings.highContrast && level === 'normal') return '';
    
    return settings.highContrast ? 'high-contrast' : '';
  }, [settings.highContrast]);

  // Animation helpers
  const shouldReduceMotion = useCallback((animationType: 'essential' | 'decorative' = 'decorative') => {
    if (!settings.reducedMotion) return false;
    
    // Always allow essential animations (like focus indicators)
    return animationType === 'decorative';
  }, [settings.reducedMotion]);

  // Text size helpers
  const getTextSizeClass = useCallback(() => {
    return settings.largeText ? 'text-lg' : '';
  }, [settings.largeText]);

  // Update settings
  const updateSettings = useCallback((newSettings: Partial<AccessibilitySettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
    
    // Announce setting changes
    Object.entries(newSettings).forEach(([key, value]) => {
      const settingName = key.replace(/([A-Z])/g, ' $1').toLowerCase();
      announce(`${settingName} ${value ? 'enabled' : 'disabled'}`);
    });
  }, [announce]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (announcementTimeoutRef.current) {
        clearTimeout(announcementTimeoutRef.current);
      }
    };
  }, []);

  // Skip link functionality
  const createSkipLink = useCallback((targetId: string, label: string) => {
    const skipLink = document.createElement('a');
    skipLink.href = `#${targetId}`;
    skipLink.className = 'skip-link sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-blue-600 focus:text-white focus:rounded';
    skipLink.textContent = label;
    
    return skipLink;
  }, []);

  return {
    settings,
    announce,
    registerShortcut,
    unregisterShortcut,
    manageFocus,
    getAriaLabel,
    getAriaDescribedBy,
    getContrastClass,
    shouldReduceMotion,
    getTextSizeClass,
    updateSettings,
    createSkipLink,
    keyboardShortcuts,
    announcements
  };
};

export default useAccessibility;