import React, { createContext, useContext, useEffect, useState } from 'react';

type Theme = 'dark' | 'light' | 'system';
type ResolvedTheme = 'dark' | 'light';

interface ThemeProviderContextType {
  theme: Theme;
  resolvedTheme: ResolvedTheme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  systemPreference: ResolvedTheme;
  highContrast: boolean;
  setHighContrast: (enabled: boolean) => void;
  reducedMotion: boolean;
  setReducedMotion: (enabled: boolean) => void;
}

const ThemeProviderContext = createContext<ThemeProviderContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: React.ReactNode;
  defaultTheme?: Theme;
  storageKey?: string;
}

export function ThemeProvider({ 
  children, 
  defaultTheme = 'system', 
  storageKey = 'vite-ui-theme' 
}: ThemeProviderProps) {
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem(storageKey) as Theme) || defaultTheme;
    }
    return defaultTheme;
  });

  const [systemPreference, setSystemPreference] = useState<ResolvedTheme>(() => {
    if (typeof window !== 'undefined') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return 'light';
  });

  const [highContrast, setHighContrastState] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('high-contrast') === 'true';
    }
    return false;
  });

  const [reducedMotion, setReducedMotionState] = useState(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('reduced-motion');
      if (stored !== null) {
        return stored === 'true';
      }
      return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }
    return false;
  });

  const resolvedTheme: ResolvedTheme = theme === 'system' ? systemPreference : theme;

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e: MediaQueryListEvent) => {
      setSystemPreference(e.matches ? 'dark' : 'light');
    };
    
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Listen for system reduced motion changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const handleChange = (e: MediaQueryListEvent) => {
      // Only update if user hasn't explicitly set a preference
      if (localStorage.getItem('reduced-motion') === null) {
        setReducedMotionState(e.matches);
      }
    };
    
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Apply theme to document
  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(resolvedTheme);

    // Apply high contrast
    if (highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }

    // Apply reduced motion
    if (reducedMotion) {
      root.classList.add('reduced-motion');
    } else {
      root.classList.remove('reduced-motion');
    }

    // Set CSS custom properties for better integration
    root.style.setProperty('--theme', resolvedTheme);
    root.style.setProperty('--high-contrast', highContrast ? '1' : '0');
    root.style.setProperty('--reduced-motion', reducedMotion ? '1' : '0');
  }, [resolvedTheme, highContrast, reducedMotion]);

  const setTheme = (newTheme: Theme) => {
    localStorage.setItem(storageKey, newTheme);
    setThemeState(newTheme);
  };

  const toggleTheme = () => {
    const newTheme = resolvedTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
  };

  const setHighContrast = (enabled: boolean) => {
    localStorage.setItem('high-contrast', enabled.toString());
    setHighContrastState(enabled);
  };

  const setReducedMotion = (enabled: boolean) => {
    localStorage.setItem('reduced-motion', enabled.toString());
    setReducedMotionState(enabled);
  };

  const value = {
    theme,
    resolvedTheme,
    setTheme,
    toggleTheme,
    systemPreference,
    highContrast,
    setHighContrast,
    reducedMotion,
    setReducedMotion,
  };

  return (
    <ThemeProviderContext.Provider value={value}>
      {children}
    </ThemeProviderContext.Provider>
  );
}

export const useTheme = () => {
  const context = useContext(ThemeProviderContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};