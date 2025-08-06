# Animation System and Interactive Onboarding

This directory contains a comprehensive animation system and interactive onboarding experience for automata theory learning. The system provides smooth state transitions, visual feedback, and guided tutorials to enhance the learning experience.

## Components Overview

### 1. AnimationSystem.tsx
A complete animation system for visualizing automaton execution with React Spring animations.

**Features:**
- Smooth state transition animations
- Input tape animation for Turing Machines
- Stack visualization for Pushdown Automata
- Parse tree animation for Context-Free Grammars
- Configurable animation speed and easing
- Play/pause/step controls
- Real-time metrics tracking

**Props:**
```typescript
interface AnimationSystemProps {
  automaton: ExtendedAutomaton;
  simulationSteps: SimulationStep[];
  currentStep: number;
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onStep: (direction: 'forward' | 'backward') => void;
  onSeek: (step: number) => void;
  onReset: () => void;
  className?: string;
}
```

### 2. InteractiveOnboarding.tsx
Step-by-step tutorial system with achievement tracking and progress persistence.

**Features:**
- Automata-specific tutorial content
- Interactive tooltips and guidance
- Achievement system with progress tracking
- Example automata for practice
- Keyboard navigation support
- Persistent progress storage

**Props:**
```typescript
interface InteractiveOnboardingProps {
  automataType: AutomataType;
  isVisible: boolean;
  onClose: () => void;
  onAutomatonCreate?: (automaton: ExtendedAutomaton) => void;
  onExampleLoad?: (example: ExtendedAutomaton) => void;
  currentAutomaton?: ExtendedAutomaton;
  className?: string;
}
```

### 3. AnimationDemo.tsx
Demonstration component showing both systems in action.

### 4. EnhancedAutomataInterface.tsx
Complete integration example combining canvas, animation, and onboarding.

## Hooks

### useAnimationSystem
Custom hook for managing animation state and controls.

```typescript
const {
  currentStep,
  isPlaying,
  play,
  pause,
  stop,
  reset,
  stepForward,
  stepBackward,
  seekTo,
  updateConfig,
  metrics
} = useAnimationSystem({
  simulationSteps,
  onStepChange,
  onAnimationEvent
});
```

### useOnboarding
Custom hook for tutorial progress and achievement management.

```typescript
const {
  isOnboardingActive,
  shouldShowOnboarding,
  startOnboarding,
  completeOnboarding,
  achievements,
  userProgress,
  unlockAchievement
} = useOnboarding({
  automataType,
  onAchievementUnlocked,
  onProgressUpdate
});
```

## Types

### animation.ts
Comprehensive TypeScript definitions for:
- Animation configurations and settings
- Achievement and progress tracking
- Tutorial steps and states
- Animation presets and metrics

## Usage Examples

### Basic Animation System
```tsx
import AnimationSystem from './components/AnimationSystem';
import { useAnimationSystem } from './hooks/useAnimationSystem';

function MyComponent() {
  const {
    currentStep,
    isPlaying,
    play,
    pause,
    // ... other controls
  } = useAnimationSystem({
    simulationSteps: mySimulationSteps
  });

  return (
    <AnimationSystem
      automaton={myAutomaton}
      simulationSteps={mySimulationSteps}
      currentStep={currentStep}
      isPlaying={isPlaying}
      onPlay={play}
      onPause={pause}
      // ... other props
    />
  );
}
```

### Interactive Onboarding
```tsx
import InteractiveOnboarding from './components/InteractiveOnboarding';
import { useOnboarding } from './hooks/useOnboarding';

function MyApp() {
  const {
    isOnboardingActive,
    shouldShowOnboarding,
    startOnboarding,
    completeOnboarding
  } = useOnboarding({
    automataType: 'dfa'
  });

  return (
    <>
      {shouldShowOnboarding() && (
        <button onClick={startOnboarding}>
          Start Tutorial
        </button>
      )}
      
      <InteractiveOnboarding
        automataType="dfa"
        isVisible={isOnboardingActive}
        onClose={completeOnboarding}
        onExampleLoad={handleExampleLoad}
      />
    </>
  );
}
```

### Complete Integration
```tsx
import EnhancedAutomataInterface from './components/EnhancedAutomataInterface';

function AutomataApp() {
  return (
    <EnhancedAutomataInterface
      automataType="dfa"
      showOnboarding={true}
      onAutomatonChange={handleAutomatonChange}
      onSimulationResult={handleSimulationResult}
    />
  );
}
```

## Animation Presets

Three built-in animation presets are available:

1. **Smooth**: Gentle animations for beginners (800ms duration)
2. **Fast**: Quick animations for experienced users (300ms duration)  
3. **Educational**: Detailed animations with explanations (1200ms duration)

## Achievement System

Built-in achievements include:
- **First Steps**: Complete tutorial introduction
- **Automaton Creator**: Create first automaton
- **Simulation Master**: Run multiple successful simulations
- **AI Explorer**: Use AI assistance
- **Multi-Type Master**: Work with different automata types

## Storage and Persistence

User progress and achievements are automatically saved to localStorage:
- Tutorial completion status
- Achievement unlocks and progress
- User statistics and metrics
- Last active date tracking

## Keyboard Shortcuts

- **Arrow Keys**: Navigate tutorial steps
- **Escape**: Close onboarding dialog
- **Space**: Play/pause animations (when focused)

## Customization

### Animation Configuration
```typescript
updateConfig({
  duration: 1000,
  easing: 'gentle',
  stagger: 150,
  showTrails: true,
  highlightIntensity: 1.2
});
```

### Tutorial Steps
Tutorial content is automatically generated based on automata type, but can be customized by modifying the `getTutorialSteps()` function in `InteractiveOnboarding.tsx`.

### Achievements
Add custom achievements by extending the `DEFAULT_ACHIEVEMENTS` array in `useOnboarding.ts`.

## Dependencies

- **@react-spring/web**: Animation library
- **React**: Core framework
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling
- **Radix UI**: UI components

## Browser Support

- Modern browsers with ES2020 support
- localStorage for persistence
- CSS Grid and Flexbox for layouts

## Performance Considerations

- Animations use hardware acceleration when available
- Step-by-step rendering prevents UI blocking
- Configurable animation complexity
- Lazy loading of tutorial content
- Efficient state management with minimal re-renders

## Accessibility

- Keyboard navigation support
- ARIA labels and roles
- High contrast mode compatibility
- Screen reader friendly
- Configurable animation intensity
- Skip options for users with vestibular disorders