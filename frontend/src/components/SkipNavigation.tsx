import React from 'react';
import { useAccessibility } from '../contexts/AccessibilityProvider';

interface SkipNavigationProps {
  links?: Array<{
    href: string;
    label: string;
    target?: string;
  }>;
}

export function SkipNavigation({ links }: SkipNavigationProps) {
  const { announceMessage } = useAccessibility();

  const defaultLinks = [
    { href: '#main-content', label: 'Skip to main content', target: 'main content' },
    { href: '#navigation', label: 'Skip to navigation', target: 'navigation' },
    { href: '#footer', label: 'Skip to footer', target: 'footer' }
  ];

  const skipLinks = links || defaultLinks;

  const handleSkipLink = (event: React.MouseEvent, target?: string) => {
    if (target) {
      announceMessage(`Navigated to ${target}`);
    }
  };

  return (
    <nav
      className="skip-nav-container"
      aria-label="Skip navigation links"
    >
      {skipLinks.map(({ href, label, target }) => (
        <a
          key={href}
          href={href}
          className="skip-nav"
          onClick={(e) => handleSkipLink(e, target)}
          onFocus={() => announceMessage('Skip navigation links available')}
        >
          {label}
        </a>
      ))}
    </nav>
  );
}