import React, { useState } from 'react';
import { useTheme } from '../contexts/ThemeProvider';
import { useAccessibility } from '../contexts/AccessibilityProvider';
import { Button } from './ui/button';
import { Switch } from './ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Separator } from './ui/separator';
import { Badge } from './ui/badge';
import { 
  Settings, 
  Moon, 
  Sun, 
  Monitor, 
  Eye, 
  Keyboard, 
  Volume2, 
  Type,
  Contrast,
  Zap,
  Info,
  X
} from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';

interface AccessibilitySettingsProps {
  trigger?: React.ReactNode;
}

export function AccessibilitySettings({ trigger }: AccessibilitySettingsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  
  const { 
    theme, 
    setTheme, 
    resolvedTheme, 
    highContrast, 
    setHighContrast, 
    reducedMotion, 
    setReducedMotion 
  } = useTheme();

  const {
    keyboardNavigation,
    setKeyboardNavigation,
    screenReaderOptimizations,
    setScreenReaderOptimizations,
    announcements,
    setAnnouncements,
    fontSize,
    setFontSize,
    shortcuts,
    announceMessage
  } = useAccessibility();

  const handleClose = () => {
    setIsOpen(false);
    announceMessage('Accessibility settings closed');
  };

  const themeOptions = [
    { value: 'light', label: 'Light', icon: Sun },
    { value: 'dark', label: 'Dark', icon: Moon },
    { value: 'system', label: 'System', icon: Monitor }
  ];

  const fontSizeOptions = [
    { value: 'small', label: 'Small (14px)', description: 'Compact text for more content' },
    { value: 'medium', label: 'Medium (16px)', description: 'Default comfortable reading size' },
    { value: 'large', label: 'Large (18px)', description: 'Larger text for better readability' },
    { value: 'extra-large', label: 'Extra Large (20px)', description: 'Maximum text size for accessibility' }
  ];

  const defaultTrigger = (
    <Button variant="outline" size="sm" className="gap-2">
      <Settings className="h-4 w-4" />
      Accessibility
    </Button>
  );

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        {trigger || defaultTrigger}
      </DialogTrigger>
      <DialogContent 
        className="max-w-2xl max-h-[80vh] overflow-y-auto"
        onCloseAutoFocus={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Accessibility Settings
          </DialogTitle>
          <DialogDescription>
            Customize the interface to meet your accessibility needs. All settings are saved automatically.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Theme Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Eye className="h-4 w-4" />
                Visual Preferences
              </CardTitle>
              <CardDescription>
                Adjust theme, colors, and visual appearance
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Theme Selection */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Color Theme</label>
                <div className="grid grid-cols-3 gap-2">
                  {themeOptions.map(({ value, label, icon: Icon }) => (
                    <Button
                      key={value}
                      variant={theme === value ? "default" : "outline"}
                      size="sm"
                      onClick={() => {
                        setTheme(value as any);
                        announceMessage(`Theme changed to ${label}`);
                      }}
                      className="flex items-center gap-2 justify-start"
                      aria-pressed={theme === value}
                    >
                      <Icon className="h-4 w-4" />
                      {label}
                    </Button>
                  ))}
                </div>
                <div className="text-xs text-muted-foreground">
                  Currently using: <Badge variant="secondary">{resolvedTheme}</Badge>
                </div>
              </div>

              <Separator />

              {/* High Contrast */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-sm font-medium flex items-center gap-2">
                    <Contrast className="h-4 w-4" />
                    High Contrast Mode
                  </label>
                  <p className="text-xs text-muted-foreground">
                    Increases color contrast for better visibility
                  </p>
                </div>
                <Switch
                  checked={highContrast}
                  onCheckedChange={setHighContrast}
                  aria-label="Toggle high contrast mode"
                />
              </div>

              {/* Reduced Motion */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-sm font-medium flex items-center gap-2">
                    <Zap className="h-4 w-4" />
                    Reduce Motion
                  </label>
                  <p className="text-xs text-muted-foreground">
                    Minimizes animations and transitions
                  </p>
                </div>
                <Switch
                  checked={reducedMotion}
                  onCheckedChange={setReducedMotion}
                  aria-label="Toggle reduced motion"
                />
              </div>

              {/* Font Size */}
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Type className="h-4 w-4" />
                  Font Size
                </label>
                <Select value={fontSize} onValueChange={setFontSize}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {fontSizeOptions.map(({ value, label, description }) => (
                      <SelectItem key={value} value={value}>
                        <div className="space-y-1">
                          <div className="font-medium">{label}</div>
                          <div className="text-xs text-muted-foreground">{description}</div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Interaction Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Keyboard className="h-4 w-4" />
                Interaction & Navigation
              </CardTitle>
              <CardDescription>
                Configure keyboard navigation and input methods
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Keyboard Navigation */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-sm font-medium">Keyboard Navigation</label>
                  <p className="text-xs text-muted-foreground">
                    Enable keyboard shortcuts and navigation
                  </p>
                </div>
                <Switch
                  checked={keyboardNavigation}
                  onCheckedChange={setKeyboardNavigation}
                  aria-label="Toggle keyboard navigation"
                />
              </div>

              {/* Keyboard Shortcuts Help */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-sm font-medium">Keyboard Shortcuts</label>
                  <p className="text-xs text-muted-foreground">
                    View available keyboard shortcuts
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowShortcuts(!showShortcuts)}
                  aria-label="Toggle keyboard shortcuts help"
                >
                  {showShortcuts ? 'Hide' : 'Show'} Shortcuts
                </Button>
              </div>

              {showShortcuts && (
                <Card className="bg-muted/50">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Available Keyboard Shortcuts</CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="space-y-2">
                      {shortcuts.map(({ key, ctrlKey, altKey, shiftKey, metaKey, description }) => {
                        const modifiers = [];
                        if (ctrlKey) modifiers.push('Ctrl');
                        if (altKey) modifiers.push('Alt');
                        if (shiftKey) modifiers.push('Shift');
                        if (metaKey) modifiers.push('Cmd');
                        
                        const shortcutDisplay = [...modifiers, key.toUpperCase()].join(' + ');
                        
                        return (
                          <div key={description} className="flex justify-between text-xs">
                            <span>{description}</span>
                            <Badge variant="outline" className="font-mono">
                              {shortcutDisplay}
                            </Badge>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>

          {/* Screen Reader Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Volume2 className="h-4 w-4" />
                Screen Reader Support
              </CardTitle>
              <CardDescription>
                Configure screen reader optimizations and announcements
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Screen Reader Optimizations */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-sm font-medium">Screen Reader Optimizations</label>
                  <p className="text-xs text-muted-foreground">
                    Enhanced markup and navigation for screen readers
                  </p>
                </div>
                <Switch
                  checked={screenReaderOptimizations}
                  onCheckedChange={setScreenReaderOptimizations}
                  aria-label="Toggle screen reader optimizations"
                />
              </div>

              {/* Announcements */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-sm font-medium">Voice Announcements</label>
                  <p className="text-xs text-muted-foreground">
                    Announce actions and status changes
                  </p>
                </div>
                <Switch
                  checked={announcements}
                  onCheckedChange={setAnnouncements}
                  aria-label="Toggle voice announcements"
                />
              </div>
            </CardContent>
          </Card>

          {/* Information */}
          <Card className="bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2 text-blue-900 dark:text-blue-100">
                <Info className="h-4 w-4" />
                Accessibility Information
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0 text-xs text-blue-800 dark:text-blue-200 space-y-2">
              <p>
                This application follows WCAG AAA accessibility guidelines and supports:
              </p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>Screen reader compatibility (NVDA, JAWS, VoiceOver)</li>
                <li>Full keyboard navigation support</li>
                <li>High contrast mode for visual impairments</li>
                <li>Customizable font sizes and reduced motion</li>
                <li>Focus management and skip navigation</li>
              </ul>
            </CardContent>
          </Card>
        </div>

        <div className="flex justify-end pt-4">
          <Button onClick={handleClose} className="gap-2">
            <X className="h-4 w-4" />
            Close Settings
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}