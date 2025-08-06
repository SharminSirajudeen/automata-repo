import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { State, Transition, Automaton } from '../types/automata';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Slider } from './ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { 
  Trash2, Play, Square, RotateCcw, Move, MousePointer, Copy, 
  Paste, Undo, Redo, ZoomIn, ZoomOut, Grid, Group, Ungroup,
  Navigation, Layers, Download, Upload, Settings
} from 'lucide-react';

// Command types for undo/redo system
interface Command {
  type: 'add_state' | 'delete_state' | 'move_state' | 'add_transition' | 'delete_transition' | 'update_state' | 'update_transition' | 'group_states' | 'ungroup_states';
  data: any;
  timestamp: number;
}

interface AutomataGroup {
  id: string;
  stateIds: string[];
  label: string;
  color: string;
}

interface ViewportTransform {
  x: number;
  y: number;
  scale: number;
}

interface EnhancedAutomataCanvasProps {
  automaton: Automaton;
  onAutomatonChange: (automaton: Automaton) => void;
  onRequestAIGuidance?: () => void;
  isSimulating?: boolean;
  simulationPath?: string[];
  currentSimulationStep?: number;
  showInteractiveOverlay?: boolean;
  stepExplanations?: { [key: string]: string };
  onStateHover?: (stateId: string) => void;
  onTransitionHover?: (transitionIndex: number) => void;
  alphabet?: string[];
  readOnly?: boolean;
  automatonType?: string;
  enableAdvancedFeatures?: boolean;
}

interface Point {
  x: number;
  y: number;
}

const GRID_SIZE = 20;
const STATE_RADIUS = 30;
const CANVAS_WIDTH = 1200;
const CANVAS_HEIGHT = 800;
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 3.0;

export const EnhancedAutomataCanvas: React.FC<EnhancedAutomataCanvasProps> = ({
  automaton,
  onAutomatonChange,
  onRequestAIGuidance,
  isSimulating = false,
  simulationPath = [],
  currentSimulationStep = 0,
  showInteractiveOverlay = false,
  stepExplanations = {},
  onStateHover,
  onTransitionHover,
  enableAdvancedFeatures = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  
  // Enhanced state management
  const [selectedStates, setSelectedStates] = useState<Set<string>>(new Set());
  const [selectedTransitions, setSelectedTransitions] = useState<Set<number>>(new Set());
  const [copiedStates, setCopiedStates] = useState<State[]>([]);
  const [copiedTransitions, setCopiedTransitions] = useState<Transition[]>([]);
  
  // Command history for undo/redo
  const [commandHistory, setCommandHistory] = useState<Command[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  
  // Tool modes
  const [currentTool, setCurrentTool] = useState<'select' | 'move' | 'lasso' | 'transition'>('select');
  const [isCreatingTransition, setIsCreatingTransition] = useState(false);
  const [transitionStart, setTransitionStart] = useState<string | null>(null);
  
  // Viewport and zoom
  const [viewport, setViewport] = useState<ViewportTransform>({ x: 0, y: 0, scale: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState<Point>({ x: 0, y: 0 });
  
  // Grid and snapping
  const [showGrid, setShowGrid] = useState(true);
  const [snapToGrid, setSnapToGrid] = useState(true);
  
  // Multi-select and lasso
  const [isLassoActive, setIsLassoActive] = useState(false);
  const [lassoPath, setLassoPath] = useState<Point[]>([]);
  
  // Drag and drop
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<Point>({ x: 0, y: 0 });
  const [dragOffset, setDragOffset] = useState<{ [stateId: string]: Point }>({});
  
  // State groups
  const [stateGroups, setStateGroups] = useState<AutomataGroup[]>([]);
  
  // Input dialogs
  const [showSymbolInput, setShowSymbolInput] = useState(false);
  const [newSymbol, setNewSymbol] = useState('');
  const [pendingTransition, setPendingTransition] = useState<{ from: string; to: string } | null>(null);
  
  // Layout algorithms
  const [isApplyingLayout, setIsApplyingLayout] = useState(false);

  // Touch gesture handling for mobile
  const [touchState, setTouchState] = useState<{
    touches: Touch[];
    lastDistance: number;
    lastCenter: Point;
  }>({ touches: [], lastDistance: 0, lastCenter: { x: 0, y: 0 } });

  // Utility functions
  const snapToGridPoint = useCallback((point: Point): Point => {
    if (!snapToGrid) return point;
    return {
      x: Math.round(point.x / GRID_SIZE) * GRID_SIZE,
      y: Math.round(point.y / GRID_SIZE) * GRID_SIZE
    };
  }, [snapToGrid]);

  const screenToCanvas = useCallback((screenPoint: Point): Point => {
    return {
      x: (screenPoint.x - viewport.x) / viewport.scale,
      y: (screenPoint.y - viewport.y) / viewport.scale
    };
  }, [viewport]);

  const canvasToScreen = useCallback((canvasPoint: Point): Point => {
    return {
      x: canvasPoint.x * viewport.scale + viewport.x,
      y: canvasPoint.y * viewport.scale + viewport.y
    };
  }, [viewport]);

  // Command system for undo/redo
  const executeCommand = useCallback((command: Command) => {
    const newHistory = commandHistory.slice(0, historyIndex + 1);
    newHistory.push(command);
    setCommandHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  }, [commandHistory, historyIndex]);

  const undo = useCallback(() => {
    if (historyIndex >= 0) {
      const command = commandHistory[historyIndex];
      // Implement undo logic based on command type
      switch (command.type) {
        case 'add_state':
          onAutomatonChange({
            ...automaton,
            states: automaton.states.filter(s => s.id !== command.data.stateId)
          });
          break;
        case 'delete_state':
          onAutomatonChange({
            ...automaton,
            states: [...automaton.states, command.data.state],
            transitions: [...automaton.transitions, ...command.data.transitions]
          });
          break;
        // Add more undo cases as needed
      }
      setHistoryIndex(historyIndex - 1);
    }
  }, [historyIndex, commandHistory, automaton, onAutomatonChange]);

  const redo = useCallback(() => {
    if (historyIndex < commandHistory.length - 1) {
      setHistoryIndex(historyIndex + 1);
      const command = commandHistory[historyIndex + 1];
      // Implement redo logic
      // Similar to execute command logic
    }
  }, [historyIndex, commandHistory]);

  // Layout algorithms
  const applyForceDirectedLayout = useCallback(() => {
    if (isApplyingLayout) return;
    setIsApplyingLayout(true);

    const states = [...automaton.states];
    const iterations = 100;
    const repulsionStrength = 5000;
    const attractionStrength = 0.1;
    const damping = 0.9;

    for (let iter = 0; iter < iterations; iter++) {
      const forces: { [stateId: string]: Point } = {};
      
      // Initialize forces
      states.forEach(state => {
        forces[state.id] = { x: 0, y: 0 };
      });

      // Repulsion forces between all states
      for (let i = 0; i < states.length; i++) {
        for (let j = i + 1; j < states.length; j++) {
          const state1 = states[i];
          const state2 = states[j];
          const dx = state2.x - state1.x;
          const dy = state2.y - state1.y;
          const distance = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = repulsionStrength / (distance * distance);
          
          forces[state1.id].x -= (dx / distance) * force;
          forces[state1.id].y -= (dy / distance) * force;
          forces[state2.id].x += (dx / distance) * force;
          forces[state2.id].y += (dy / distance) * force;
        }
      }

      // Attraction forces along transitions
      automaton.transitions.forEach(transition => {
        const fromState = states.find(s => s.id === transition.from_state);
        const toState = states.find(s => s.id === transition.to_state);
        if (fromState && toState && fromState.id !== toState.id) {
          const dx = toState.x - fromState.x;
          const dy = toState.y - fromState.y;
          const distance = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = attractionStrength * distance;
          
          forces[fromState.id].x += (dx / distance) * force;
          forces[fromState.id].y += (dy / distance) * force;
          forces[toState.id].x -= (dx / distance) * force;
          forces[toState.id].y -= (dy / distance) * force;
        }
      });

      // Apply forces with damping
      states.forEach(state => {
        state.x += forces[state.id].x * damping;
        state.y += forces[state.id].y * damping;
        
        // Keep states within canvas bounds
        state.x = Math.max(STATE_RADIUS, Math.min(CANVAS_WIDTH - STATE_RADIUS, state.x));
        state.y = Math.max(STATE_RADIUS, Math.min(CANVAS_HEIGHT - STATE_RADIUS, state.y));
      });
    }

    onAutomatonChange({ ...automaton, states });
    setIsApplyingLayout(false);
  }, [automaton, onAutomatonChange, isApplyingLayout]);

  const applyHierarchicalLayout = useCallback(() => {
    if (isApplyingLayout) return;
    setIsApplyingLayout(true);

    const states = [...automaton.states];
    const startState = states.find(s => s.is_start);
    if (!startState) {
      setIsApplyingLayout(false);
      return;
    }

    // Build adjacency list
    const adjacency: { [stateId: string]: string[] } = {};
    states.forEach(state => {
      adjacency[state.id] = [];
    });
    
    automaton.transitions.forEach(transition => {
      if (!adjacency[transition.from_state].includes(transition.to_state)) {
        adjacency[transition.from_state].push(transition.to_state);
      }
    });

    // BFS to assign levels
    const levels: { [stateId: string]: number } = {};
    const queue = [startState.id];
    levels[startState.id] = 0;
    let maxLevel = 0;

    while (queue.length > 0) {
      const currentStateId = queue.shift()!;
      const currentLevel = levels[currentStateId];
      
      adjacency[currentStateId].forEach(nextStateId => {
        if (levels[nextStateId] === undefined) {
          levels[nextStateId] = currentLevel + 1;
          maxLevel = Math.max(maxLevel, currentLevel + 1);
          queue.push(nextStateId);
        }
      });
    }

    // Assign unvisited states to the last level
    states.forEach(state => {
      if (levels[state.id] === undefined) {
        levels[state.id] = maxLevel + 1;
        maxLevel = maxLevel + 1;
      }
    });

    // Group states by level
    const levelGroups: { [level: number]: string[] } = {};
    Object.entries(levels).forEach(([stateId, level]) => {
      if (!levelGroups[level]) levelGroups[level] = [];
      levelGroups[level].push(stateId);
    });

    // Position states
    const levelHeight = CANVAS_HEIGHT / (maxLevel + 2);
    Object.entries(levelGroups).forEach(([level, stateIds]) => {
      const levelNum = parseInt(level);
      const y = levelHeight * (levelNum + 1);
      const stateWidth = CANVAS_WIDTH / (stateIds.length + 1);
      
      stateIds.forEach((stateId, index) => {
        const state = states.find(s => s.id === stateId);
        if (state) {
          state.x = stateWidth * (index + 1);
          state.y = y;
        }
      });
    });

    onAutomatonChange({ ...automaton, states });
    setIsApplyingLayout(false);
  }, [automaton, onAutomatonChange, isApplyingLayout]);

  // Drawing functions
  const drawGrid = useCallback((ctx: CanvasRenderingContext2D) => {
    if (!showGrid) return;
    
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.setLineDash([]);

    const startX = Math.floor(-viewport.x / viewport.scale / GRID_SIZE) * GRID_SIZE;
    const startY = Math.floor(-viewport.y / viewport.scale / GRID_SIZE) * GRID_SIZE;
    const endX = startX + CANVAS_WIDTH / viewport.scale + GRID_SIZE;
    const endY = startY + CANVAS_HEIGHT / viewport.scale + GRID_SIZE;

    for (let x = startX; x <= endX; x += GRID_SIZE) {
      ctx.beginPath();
      ctx.moveTo(x, startY);
      ctx.lineTo(x, endY);
      ctx.stroke();
    }

    for (let y = startY; y <= endY; y += GRID_SIZE) {
      ctx.beginPath();
      ctx.moveTo(startX, y);
      ctx.lineTo(endX, y);
      ctx.stroke();
    }
  }, [showGrid, viewport]);

  const drawState = useCallback((ctx: CanvasRenderingContext2D, state: State) => {
    const isSelected = selectedStates.has(state.id);
    const isCurrentInSimulation = isSimulating && simulationPath[currentSimulationStep] === state.id;
    const wasVisitedInSimulation = isSimulating && simulationPath.slice(0, currentSimulationStep + 1).includes(state.id);
    
    // State fill
    ctx.fillStyle = isCurrentInSimulation ? '#ef4444' : 
                   wasVisitedInSimulation ? '#f59e0b' :
                   isSelected ? '#3b82f6' : '#ffffff';
    
    // State border
    ctx.strokeStyle = state.is_accept ? '#10b981' : '#64748b';
    ctx.lineWidth = isSelected ? 4 : state.is_accept ? 3 : 2;
    
    ctx.beginPath();
    ctx.arc(state.x, state.y, STATE_RADIUS, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Accept state double circle
    if (state.is_accept) {
      ctx.beginPath();
      ctx.arc(state.x, state.y, STATE_RADIUS - 6, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // Start state arrow
    if (state.is_start) {
      ctx.strokeStyle = '#1f2937';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(state.x - STATE_RADIUS - 25, state.y);
      ctx.lineTo(state.x - STATE_RADIUS, state.y);
      ctx.stroke();
      
      // Arrow head
      ctx.beginPath();
      ctx.moveTo(state.x - STATE_RADIUS, state.y);
      ctx.lineTo(state.x - STATE_RADIUS - 8, state.y - 4);
      ctx.moveTo(state.x - STATE_RADIUS, state.y);
      ctx.lineTo(state.x - STATE_RADIUS - 8, state.y + 4);
      ctx.stroke();
    }

    // State label
    ctx.fillStyle = '#1f2937';
    ctx.font = `${14 * viewport.scale}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(state.label || state.id, state.x, state.y);
  }, [selectedStates, isSimulating, simulationPath, currentSimulationStep, viewport.scale]);

  const drawTransition = useCallback((ctx: CanvasRenderingContext2D, transition: Transition, index: number) => {
    const fromState = automaton.states.find(s => s.id === transition.from_state);
    const toState = automaton.states.find(s => s.id === transition.to_state);
    
    if (!fromState || !toState) return;

    const isSelected = selectedTransitions.has(index);
    ctx.strokeStyle = isSelected ? '#3b82f6' : '#64748b';
    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.setLineDash([]);

    if (fromState.id === toState.id) {
      // Self-loop
      const loopRadius = 30;
      ctx.beginPath();
      ctx.arc(fromState.x, fromState.y - STATE_RADIUS - loopRadius, loopRadius, 0, 2 * Math.PI);
      ctx.stroke();
      
      // Label
      ctx.fillStyle = '#1f2937';
      ctx.font = `${12 * viewport.scale}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(transition.symbol, fromState.x, fromState.y - STATE_RADIUS - loopRadius * 2 - 10);
    } else {
      // Regular transition
      const angle = Math.atan2(toState.y - fromState.y, toState.x - fromState.x);
      const startX = fromState.x + Math.cos(angle) * STATE_RADIUS;
      const startY = fromState.y + Math.sin(angle) * STATE_RADIUS;
      const endX = toState.x - Math.cos(angle) * STATE_RADIUS;
      const endY = toState.y - Math.sin(angle) * STATE_RADIUS;

      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      // Arrow head
      const arrowLength = 12;
      const arrowAngle = Math.PI / 6;
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - arrowLength * Math.cos(angle - arrowAngle),
        endY - arrowLength * Math.sin(angle - arrowAngle)
      );
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - arrowLength * Math.cos(angle + arrowAngle),
        endY - arrowLength * Math.sin(angle + arrowAngle)
      );
      ctx.stroke();

      // Label
      const midX = (startX + endX) / 2;
      const midY = (startY + endY) / 2;
      ctx.fillStyle = '#1f2937';
      ctx.font = `${12 * viewport.scale}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(transition.symbol, midX, midY - 8);
    }
  }, [automaton.states, selectedTransitions, viewport.scale]);

  const drawLasso = useCallback((ctx: CanvasRenderingContext2D) => {
    if (!isLassoActive || lassoPath.length < 2) return;
    
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    
    ctx.beginPath();
    ctx.moveTo(lassoPath[0].x, lassoPath[0].y);
    lassoPath.slice(1).forEach(point => {
      ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }, [isLassoActive, lassoPath]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // Apply viewport transformation
    ctx.save();
    ctx.translate(viewport.x, viewport.y);
    ctx.scale(viewport.scale, viewport.scale);

    // Background
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(-viewport.x / viewport.scale, -viewport.y / viewport.scale, 
                 CANVAS_WIDTH / viewport.scale, CANVAS_HEIGHT / viewport.scale);

    // Draw grid
    drawGrid(ctx);

    // Draw state groups
    stateGroups.forEach(group => {
      if (group.stateIds.length > 1) {
        const groupStates = automaton.states.filter(s => group.stateIds.includes(s.id));
        if (groupStates.length > 1) {
          const minX = Math.min(...groupStates.map(s => s.x)) - 50;
          const maxX = Math.max(...groupStates.map(s => s.x)) + 50;
          const minY = Math.min(...groupStates.map(s => s.y)) - 50;
          const maxY = Math.max(...groupStates.map(s => s.y)) + 50;
          
          ctx.strokeStyle = group.color;
          ctx.fillStyle = group.color + '20';
          ctx.lineWidth = 2;
          ctx.setLineDash([10, 5]);
          
          ctx.fillRect(minX, minY, maxX - minX, maxY - minY);
          ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
          ctx.setLineDash([]);
          
          // Group label
          ctx.fillStyle = group.color;
          ctx.font = '14px sans-serif';
          ctx.textAlign = 'left';
          ctx.fillText(group.label, minX + 5, minY - 5);
        }
      }
    });

    // Draw transitions first (so they appear behind states)
    automaton.transitions.forEach((transition, index) => {
      drawTransition(ctx, transition, index);
    });

    // Draw states
    automaton.states.forEach(state => {
      drawState(ctx, state);
    });

    // Draw lasso selection
    drawLasso(ctx);

    ctx.restore();
  }, [viewport, automaton, drawGrid, drawState, drawTransition, drawLasso, stateGroups]);

  // Event handlers
  const handleCanvasMouseDown = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const screenPoint = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const canvasPoint = screenToCanvas(screenPoint);

    if (event.button === 1 || (event.button === 0 && event.altKey)) {
      // Middle mouse or Alt+click for panning
      setIsPanning(true);
      setPanStart(screenPoint);
      return;
    }

    const clickedState = automaton.states.find(state => {
      const distance = Math.sqrt((canvasPoint.x - state.x) ** 2 + (canvasPoint.y - state.y) ** 2);
      return distance <= STATE_RADIUS;
    });

    if (currentTool === 'lasso') {
      setIsLassoActive(true);
      setLassoPath([canvasPoint]);
      return;
    }

    if (currentTool === 'transition' && clickedState) {
      if (!transitionStart) {
        setTransitionStart(clickedState.id);
        setIsCreatingTransition(true);
      } else if (transitionStart !== clickedState.id) {
        setPendingTransition({ from: transitionStart, to: clickedState.id });
        setShowSymbolInput(true);
        setTransitionStart(null);
        setIsCreatingTransition(false);
      }
      return;
    }

    if (clickedState) {
      if (event.ctrlKey || event.metaKey) {
        // Multi-select
        const newSelection = new Set(selectedStates);
        if (newSelection.has(clickedState.id)) {
          newSelection.delete(clickedState.id);
        } else {
          newSelection.add(clickedState.id);
        }
        setSelectedStates(newSelection);
      } else if (!selectedStates.has(clickedState.id)) {
        setSelectedStates(new Set([clickedState.id]));
      }

      if (currentTool === 'move' || currentTool === 'select') {
        setIsDragging(true);
        setDragStart(canvasPoint);
        
        // Calculate drag offsets for all selected states
        const offsets: { [stateId: string]: Point } = {};
        selectedStates.forEach(stateId => {
          const state = automaton.states.find(s => s.id === stateId);
          if (state) {
            offsets[stateId] = { x: state.x - canvasPoint.x, y: state.y - canvasPoint.y };
          }
        });
        if (!selectedStates.has(clickedState.id)) {
          offsets[clickedState.id] = { x: clickedState.x - canvasPoint.x, y: clickedState.y - canvasPoint.y };
        }
        setDragOffset(offsets);
      }
    } else {
      // Click on empty space
      if (currentTool === 'select' && !event.ctrlKey && !event.metaKey) {
        // Create new state
        const newStateId = `q${automaton.states.length}`;
        const position = snapToGridPoint(canvasPoint);
        const newState: State = {
          id: newStateId,
          x: position.x,
          y: position.y,
          is_start: automaton.states.length === 0,
          is_accept: false,
          label: newStateId,
        };

        executeCommand({
          type: 'add_state',
          data: { stateId: newStateId, state: newState },
          timestamp: Date.now()
        });

        onAutomatonChange({
          ...automaton,
          states: [...automaton.states, newState],
        });

        setSelectedStates(new Set([newStateId]));
      }
      setSelectedTransitions(new Set());
    }
  }, [canvasRef, screenToCanvas, currentTool, automaton, selectedStates, transitionStart, snapToGridPoint, executeCommand, onAutomatonChange]);

  const handleCanvasMouseMove = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const screenPoint = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const canvasPoint = screenToCanvas(screenPoint);

    if (isPanning) {
      const deltaX = screenPoint.x - panStart.x;
      const deltaY = screenPoint.y - panStart.y;
      setViewport(prev => ({
        ...prev,
        x: prev.x + deltaX,
        y: prev.y + deltaY
      }));
      setPanStart(screenPoint);
      return;
    }

    if (isLassoActive) {
      setLassoPath(prev => [...prev, canvasPoint]);
      return;
    }

    if (isDragging && selectedStates.size > 0) {
      const newStates = automaton.states.map(state => {
        if (selectedStates.has(state.id) && dragOffset[state.id]) {
          const newPosition = snapToGridPoint({
            x: canvasPoint.x + dragOffset[state.id].x,
            y: canvasPoint.y + dragOffset[state.id].y
          });
          return {
            ...state,
            x: Math.max(STATE_RADIUS, Math.min(CANVAS_WIDTH - STATE_RADIUS, newPosition.x)),
            y: Math.max(STATE_RADIUS, Math.min(CANVAS_HEIGHT - STATE_RADIUS, newPosition.y))
          };
        }
        return state;
      });

      onAutomatonChange({ ...automaton, states: newStates });
    }
  }, [canvasRef, screenToCanvas, isPanning, panStart, isLassoActive, isDragging, selectedStates, dragOffset, snapToGridPoint, automaton, onAutomatonChange]);

  const handleCanvasMouseUp = useCallback(() => {
    if (isPanning) {
      setIsPanning(false);
    }

    if (isLassoActive) {
      // Find states inside lasso
      const selectedStateIds = new Set<string>();
      automaton.states.forEach(state => {
        // Simple point-in-polygon test
        let inside = false;
        for (let i = 0, j = lassoPath.length - 1; i < lassoPath.length; j = i++) {
          if (((lassoPath[i].y > state.y) !== (lassoPath[j].y > state.y)) &&
              (state.x < (lassoPath[j].x - lassoPath[i].x) * (state.y - lassoPath[i].y) / (lassoPath[j].y - lassoPath[i].y) + lassoPath[i].x)) {
            inside = !inside;
          }
        }
        if (inside) selectedStateIds.add(state.id);
      });
      setSelectedStates(selectedStateIds);
      setIsLassoActive(false);
      setLassoPath([]);
    }

    if (isDragging) {
      setIsDragging(false);
      setDragOffset({});
    }
  }, [isPanning, isLassoActive, isDragging, automaton.states, lassoPath]);

  // Touch event handlers for mobile
  const handleTouchStart = useCallback((event: React.TouchEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const touches = Array.from(event.touches);
    setTouchState(prev => ({ ...prev, touches }));

    if (touches.length === 1) {
      // Single touch - treat as mouse down
      const touch = touches[0];
      const rect = canvasRef.current?.getBoundingClientRect();
      if (rect) {
        const mouseEvent = new MouseEvent('mousedown', {
          clientX: touch.clientX,
          clientY: touch.clientY,
          button: 0
        });
        handleCanvasMouseDown(mouseEvent as any);
      }
    } else if (touches.length === 2) {
      // Two finger gesture - zoom/pan
      const touch1 = touches[0];
      const touch2 = touches[1];
      const distance = Math.sqrt(
        Math.pow(touch2.clientX - touch1.clientX, 2) + 
        Math.pow(touch2.clientY - touch1.clientY, 2)
      );
      const center = {
        x: (touch1.clientX + touch2.clientX) / 2,
        y: (touch1.clientY + touch2.clientY) / 2
      };
      setTouchState(prev => ({ ...prev, lastDistance: distance, lastCenter: center }));
    }
  }, [handleCanvasMouseDown]);

  const handleTouchMove = useCallback((event: React.TouchEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const touches = Array.from(event.touches);

    if (touches.length === 1) {
      // Single touch - treat as mouse move
      const touch = touches[0];
      const rect = canvasRef.current?.getBoundingClientRect();
      if (rect) {
        const mouseEvent = new MouseEvent('mousemove', {
          clientX: touch.clientX,
          clientY: touch.clientY
        });
        handleCanvasMouseMove(mouseEvent as any);
      }
    } else if (touches.length === 2) {
      // Two finger gesture
      const touch1 = touches[0];
      const touch2 = touches[1];
      const distance = Math.sqrt(
        Math.pow(touch2.clientX - touch1.clientX, 2) + 
        Math.pow(touch2.clientY - touch1.clientY, 2)
      );
      const center = {
        x: (touch1.clientX + touch2.clientX) / 2,
        y: (touch1.clientY + touch2.clientY) / 2
      };

      if (touchState.lastDistance > 0) {
        // Zoom
        const zoomFactor = distance / touchState.lastDistance;
        const newScale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, viewport.scale * zoomFactor));
        
        // Pan
        const deltaX = center.x - touchState.lastCenter.x;
        const deltaY = center.y - touchState.lastCenter.y;
        
        setViewport(prev => ({
          scale: newScale,
          x: prev.x + deltaX,
          y: prev.y + deltaY
        }));
      }

      setTouchState(prev => ({ ...prev, lastDistance: distance, lastCenter: center }));
    }
  }, [handleCanvasMouseMove, touchState, viewport]);

  const handleTouchEnd = useCallback((event: React.TouchEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const touches = Array.from(event.touches);
    setTouchState(prev => ({ ...prev, touches }));

    if (touches.length === 0) {
      handleCanvasMouseUp();
    }
  }, [handleCanvasMouseUp]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.target !== document.body) return; // Only handle when not in input

      switch (event.key) {
        case 'Delete':
        case 'Backspace':
          deleteSelected();
          break;
        case 'c':
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            copySelected();
          }
          break;
        case 'v':
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            pasteSelected();
          }
          break;
        case 'z':
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            if (event.shiftKey) {
              redo();
            } else {
              undo();
            }
          }
          break;
        case 'a':
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            selectAll();
          }
          break;
        case 'g':
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            groupSelected();
          }
          break;
        case 'u':
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            ungroupSelected();
          }
          break;
        case '1':
          setCurrentTool('select');
          break;
        case '2':
          setCurrentTool('move');
          break;
        case '3':
          setCurrentTool('lasso');
          break;
        case '4':
          setCurrentTool('transition');
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedStates, selectedTransitions, copiedStates, copiedTransitions]);

  // Tool functions
  const deleteSelected = useCallback(() => {
    if (selectedStates.size > 0) {
      const statesToDelete = Array.from(selectedStates);
      const transitionsToDelete = automaton.transitions.filter(
        t => statesToDelete.includes(t.from_state) || statesToDelete.includes(t.to_state)
      );

      executeCommand({
        type: 'delete_state',
        data: {
          states: automaton.states.filter(s => statesToDelete.includes(s.id)),
          transitions: transitionsToDelete
        },
        timestamp: Date.now()
      });

      onAutomatonChange({
        ...automaton,
        states: automaton.states.filter(s => !statesToDelete.includes(s.id)),
        transitions: automaton.transitions.filter(
          t => !statesToDelete.includes(t.from_state) && !statesToDelete.includes(t.to_state)
        )
      });
      setSelectedStates(new Set());
    }

    if (selectedTransitions.size > 0) {
      const transitionIndicesToDelete = Array.from(selectedTransitions);
      onAutomatonChange({
        ...automaton,
        transitions: automaton.transitions.filter((_, index) => !transitionIndicesToDelete.includes(index))
      });
      setSelectedTransitions(new Set());
    }
  }, [selectedStates, selectedTransitions, automaton, executeCommand, onAutomatonChange]);

  const copySelected = useCallback(() => {
    const statesToCopy = automaton.states.filter(s => selectedStates.has(s.id));
    const transitionsToCopy = automaton.transitions.filter(
      t => selectedStates.has(t.from_state) && selectedStates.has(t.to_state)
    );
    
    setCopiedStates(statesToCopy);
    setCopiedTransitions(transitionsToCopy);
  }, [automaton, selectedStates]);

  const pasteSelected = useCallback(() => {
    if (copiedStates.length === 0) return;

    const stateIdMap: { [oldId: string]: string } = {};
    const newStates: State[] = [];
    const offsetX = 50;
    const offsetY = 50;

    copiedStates.forEach(state => {
      const newId = `q${automaton.states.length + newStates.length}`;
      stateIdMap[state.id] = newId;
      newStates.push({
        ...state,
        id: newId,
        x: state.x + offsetX,
        y: state.y + offsetY,
        is_start: false, // Don't copy start state property
        label: newId
      });
    });

    const newTransitions: Transition[] = copiedTransitions.map(transition => ({
      ...transition,
      from_state: stateIdMap[transition.from_state],
      to_state: stateIdMap[transition.to_state]
    }));

    onAutomatonChange({
      ...automaton,
      states: [...automaton.states, ...newStates],
      transitions: [...automaton.transitions, ...newTransitions]
    });

    setSelectedStates(new Set(newStates.map(s => s.id)));
  }, [copiedStates, copiedTransitions, automaton, onAutomatonChange]);

  const selectAll = useCallback(() => {
    setSelectedStates(new Set(automaton.states.map(s => s.id)));
    setSelectedTransitions(new Set(automaton.transitions.map((_, index) => index)));
  }, [automaton]);

  const groupSelected = useCallback(() => {
    if (selectedStates.size < 2) return;

    const groupId = `group_${stateGroups.length}`;
    const newGroup: AutomataGroup = {
      id: groupId,
      stateIds: Array.from(selectedStates),
      label: `Group ${stateGroups.length + 1}`,
      color: `hsl(${Math.random() * 360}, 70%, 50%)`
    };

    setStateGroups(prev => [...prev, newGroup]);
  }, [selectedStates, stateGroups]);

  const ungroupSelected = useCallback(() => {
    const statesToUngroup = Array.from(selectedStates);
    setStateGroups(prev => prev.filter(group => 
      !group.stateIds.some(stateId => statesToUngroup.includes(stateId))
    ));
  }, [selectedStates]);

  // Zoom functions
  const zoomIn = useCallback(() => {
    setViewport(prev => ({
      ...prev,
      scale: Math.min(MAX_ZOOM, prev.scale * 1.2)
    }));
  }, []);

  const zoomOut = useCallback(() => {
    setViewport(prev => ({
      ...prev,
      scale: Math.max(MIN_ZOOM, prev.scale / 1.2)
    }));
  }, []);

  const resetView = useCallback(() => {
    setViewport({ x: 0, y: 0, scale: 1 });
  }, []);

  const addTransition = useCallback(() => {
    if (!pendingTransition || !newSymbol.trim()) return;

    const newTransition: Transition = {
      from_state: pendingTransition.from,
      to_state: pendingTransition.to,
      symbol: newSymbol.trim(),
    };

    executeCommand({
      type: 'add_transition',
      data: { transition: newTransition },
      timestamp: Date.now()
    });

    onAutomatonChange({
      ...automaton,
      transitions: [...automaton.transitions, newTransition],
    });

    setNewSymbol('');
    setShowSymbolInput(false);
    setPendingTransition(null);
  }, [pendingTransition, newSymbol, automaton, executeCommand, onAutomatonChange]);

  // Use effect for drawing
  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  // Use effect for resizing
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        drawCanvas();
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [drawCanvas]);

  const toolButtons = useMemo(() => [
    { tool: 'select', icon: MousePointer, label: 'Select (1)', shortcut: '1' },
    { tool: 'move', icon: Move, label: 'Move (2)', shortcut: '2' },
    { tool: 'lasso', icon: Navigation, label: 'Lasso (3)', shortcut: '3' },
    { tool: 'transition', icon: Navigation, label: 'Transition (4)', shortcut: '4' }
  ], []);

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">Enhanced Automata Canvas</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Tool Selection */}
          <div className="flex flex-wrap gap-2">
            {enableAdvancedFeatures && toolButtons.map(({ tool, icon: Icon, label }) => (
              <Button
                key={tool}
                onClick={() => setCurrentTool(tool as any)}
                variant={currentTool === tool ? "default" : "outline"}
                size="sm"
                title={label}
              >
                <Icon className="w-4 h-4 mr-1" />
                {label.split(' ')[0]}
              </Button>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-2">
            <Button onClick={deleteSelected} disabled={selectedStates.size === 0 && selectedTransitions.size === 0} variant="destructive" size="sm">
              <Trash2 className="w-4 h-4 mr-1" />
              Delete
            </Button>
            
            {enableAdvancedFeatures && (
              <>
                <Button onClick={copySelected} disabled={selectedStates.size === 0} size="sm">
                  <Copy className="w-4 h-4 mr-1" />
                  Copy
                </Button>
                
                <Button onClick={pasteSelected} disabled={copiedStates.length === 0} size="sm">
                  <Paste className="w-4 h-4 mr-1" />
                  Paste
                </Button>
                
                <Button onClick={undo} disabled={historyIndex < 0} size="sm">
                  <Undo className="w-4 h-4 mr-1" />
                  Undo
                </Button>
                
                <Button onClick={redo} disabled={historyIndex >= commandHistory.length - 1} size="sm">
                  <Redo className="w-4 h-4 mr-1" />
                  Redo
                </Button>
              </>
            )}
          </div>

          {/* View Controls */}
          {enableAdvancedFeatures && (
            <div className="flex flex-wrap gap-2 items-center">
              <Button onClick={zoomIn} size="sm" variant="outline">
                <ZoomIn className="w-4 h-4 mr-1" />
                Zoom In
              </Button>
              
              <Button onClick={zoomOut} size="sm" variant="outline">
                <ZoomOut className="w-4 h-4 mr-1" />
                Zoom Out
              </Button>
              
              <Button onClick={resetView} size="sm" variant="outline">
                Reset View
              </Button>
              
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-600">Zoom:</span>
                <Slider
                  value={[viewport.scale]}
                  onValueChange={([scale]) => setViewport(prev => ({ ...prev, scale }))}
                  min={MIN_ZOOM}
                  max={MAX_ZOOM}
                  step={0.1}
                  className="w-20"
                />
                <span className="text-sm text-gray-600">{Math.round(viewport.scale * 100)}%</span>
              </div>
            </div>
          )}

          {/* Layout and Grouping */}
          {enableAdvancedFeatures && (
            <div className="flex flex-wrap gap-2">
              <Button onClick={applyForceDirectedLayout} disabled={isApplyingLayout} size="sm" variant="outline">
                Force Layout
              </Button>
              
              <Button onClick={applyHierarchicalLayout} disabled={isApplyingLayout} size="sm" variant="outline">
                Hierarchical Layout
              </Button>
              
              <Button onClick={groupSelected} disabled={selectedStates.size < 2} size="sm" variant="outline">
                <Group className="w-4 h-4 mr-1" />
                Group
              </Button>
              
              <Button onClick={ungroupSelected} disabled={selectedStates.size === 0} size="sm" variant="outline">
                <Ungroup className="w-4 h-4 mr-1" />
                Ungroup
              </Button>
              
              <Button
                onClick={() => setShowGrid(!showGrid)}
                variant={showGrid ? "default" : "outline"}
                size="sm"
              >
                <Grid className="w-4 h-4 mr-1" />
                Grid
              </Button>
              
              <Button
                onClick={() => setSnapToGrid(!snapToGrid)}
                variant={snapToGrid ? "default" : "outline"}
                size="sm"
              >
                Snap
              </Button>
            </div>
          )}

          {onRequestAIGuidance && (
            <Button
              onClick={onRequestAIGuidance}
              variant="secondary"
              size="sm"
              className="bg-gradient-to-r from-purple-500 to-blue-500 text-white"
            >
              AI Guidance
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Symbol Input Dialog */}
      {showSymbolInput && (
        <Card className="bg-blue-50">
          <CardContent className="pt-4">
            <div className="flex gap-2 items-center">
              <Input
                value={newSymbol}
                onChange={(e) => setNewSymbol(e.target.value)}
                placeholder="Enter transition symbol"
                className="w-40"
                onKeyPress={(e) => e.key === 'Enter' && addTransition()}
                autoFocus
              />
              <Button onClick={addTransition} size="sm">Add</Button>
              <Button onClick={() => setShowSymbolInput(false)} variant="outline" size="sm">Cancel</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Canvas Container */}
      <div className="border-2 border-gray-300 rounded-lg overflow-hidden relative bg-gray-50">
        <canvas
          ref={canvasRef}
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          className="cursor-crosshair touch-none"
          style={{ display: 'block', width: '100%', height: '100%', maxWidth: '100%' }}
        />
      </div>

      {/* Status Bar */}
      <div className="flex flex-wrap gap-2 justify-between items-center">
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline">States: {automaton.states.length}</Badge>
          <Badge variant="outline">Transitions: {automaton.transitions.length}</Badge>
          <Badge variant="outline">Selected: {selectedStates.size} states, {selectedTransitions.size} transitions</Badge>
          {enableAdvancedFeatures && (
            <>
              <Badge variant="outline">Groups: {stateGroups.length}</Badge>
              <Badge variant="outline">Tool: {currentTool}</Badge>
            </>
          )}
        </div>
        {enableAdvancedFeatures && (
          <div className="text-sm text-gray-600">
            Zoom: {Math.round(viewport.scale * 100)}% | Grid: {showGrid ? 'On' : 'Off'} | Snap: {snapToGrid ? 'On' : 'Off'}
          </div>
        )}
      </div>

      {/* Help Text */}
      <div className="text-sm text-gray-600 space-y-1">
        <p><strong>Basic Controls:</strong></p>
        <p>• Click empty space to create state • Drag states to move • Alt+Click to pan</p>
        {enableAdvancedFeatures && (
          <>
            <p><strong>Advanced Features:</strong></p>
            <p>• Ctrl+Click for multi-select • Ctrl+C/V to copy/paste • Ctrl+Z/Y for undo/redo</p>
            <p>• Use lasso tool for area selection • Mouse wheel to zoom • Touch gestures on mobile</p>
            <p><strong>Keyboard Shortcuts:</strong> 1-4 (tools), Delete (remove), Ctrl+A (select all), Ctrl+G (group)</p>
          </>
        )}
      </div>
    </div>
  );
};