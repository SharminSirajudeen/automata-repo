import { useRef, useEffect, useState, useCallback } from 'react';
import { State, Transition, Automaton } from '../types/automata';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Trash2, Play, Square, RotateCcw } from 'lucide-react';
import { InteractiveOverlay } from './InteractiveOverlay';

interface AutomataCanvasProps {
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
}

export const AutomataCanvas: React.FC<AutomataCanvasProps> = ({
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
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedState, setSelectedState] = useState<string | null>(null);
  const [selectedTransition, setSelectedTransition] = useState<number | null>(null);
  const [isCreatingTransition, setIsCreatingTransition] = useState(false);
  const [transitionStart, setTransitionStart] = useState<string | null>(null);
  const [newSymbol, setNewSymbol] = useState('');
  const [showSymbolInput, setShowSymbolInput] = useState(false);
  const [pendingTransition, setPendingTransition] = useState<{ from: string; to: string } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [draggedState, setDraggedState] = useState<string | null>(null);
  const [editingTransition, setEditingTransition] = useState<number | null>(null);
  const [editSymbol, setEditSymbol] = useState('');

  const STATE_RADIUS = 30;
  const CANVAS_WIDTH = 800;
  const CANVAS_HEIGHT = 500;

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    automaton.transitions.forEach((transition, index) => {
      const fromState = automaton.states.find(s => s.id === transition.from_state);
      const toState = automaton.states.find(s => s.id === transition.to_state);
      
      if (!fromState || !toState) return;

      ctx.strokeStyle = selectedTransition === index ? '#3b82f6' : '#64748b';
      ctx.lineWidth = selectedTransition === index ? 3 : 2;
      ctx.beginPath();

      if (fromState.id === toState.id) {
        const loopRadius = 25;
        ctx.arc(fromState.x, fromState.y - STATE_RADIUS - loopRadius, loopRadius, 0, 2 * Math.PI);
        ctx.stroke();
        
        ctx.fillStyle = '#1f2937';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(transition.symbol, fromState.x, fromState.y - STATE_RADIUS - loopRadius * 2 - 5);
      } else {
        const angle = Math.atan2(toState.y - fromState.y, toState.x - fromState.x);
        const startX = fromState.x + Math.cos(angle) * STATE_RADIUS;
        const startY = fromState.y + Math.sin(angle) * STATE_RADIUS;
        const endX = toState.x - Math.cos(angle) * STATE_RADIUS;
        const endY = toState.y - Math.sin(angle) * STATE_RADIUS;

        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        const arrowLength = 10;
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

        const midX = (startX + endX) / 2;
        const midY = (startY + endY) / 2;
        ctx.fillStyle = '#1f2937';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(transition.symbol, midX, midY - 5);
      }
    });

    automaton.states.forEach(state => {
      const isCurrentInSimulation = isSimulating && simulationPath[currentSimulationStep] === state.id;
      const wasVisitedInSimulation = isSimulating && simulationPath.slice(0, currentSimulationStep + 1).includes(state.id);
      
      ctx.fillStyle = isCurrentInSimulation ? '#ef4444' : 
                     wasVisitedInSimulation ? '#f59e0b' :
                     selectedState === state.id ? '#3b82f6' : '#ffffff';
      ctx.strokeStyle = state.is_accept ? '#10b981' : '#64748b';
      ctx.lineWidth = state.is_accept ? 4 : 2;
      
      ctx.beginPath();
      ctx.arc(state.x, state.y, STATE_RADIUS, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      if (state.is_accept) {
        ctx.beginPath();
        ctx.arc(state.x, state.y, STATE_RADIUS - 5, 0, 2 * Math.PI);
        ctx.stroke();
      }

      if (state.is_start) {
        ctx.strokeStyle = '#1f2937';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(state.x - STATE_RADIUS - 20, state.y);
        ctx.lineTo(state.x - STATE_RADIUS, state.y);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(state.x - STATE_RADIUS, state.y);
        ctx.lineTo(state.x - STATE_RADIUS - 8, state.y - 4);
        ctx.moveTo(state.x - STATE_RADIUS, state.y);
        ctx.lineTo(state.x - STATE_RADIUS - 8, state.y + 4);
        ctx.stroke();
      }

      ctx.fillStyle = '#1f2937';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(state.label || state.id, state.x, state.y + 5);
    });
  }, [automaton, selectedState, selectedTransition, isSimulating, simulationPath, currentSimulationStep]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const clickedState = automaton.states.find(state => {
      const distance = Math.sqrt((x - state.x) ** 2 + (y - state.y) ** 2);
      return distance <= STATE_RADIUS;
    });

    if (isCreatingTransition && transitionStart) {
      if (clickedState) {
        setPendingTransition({ from: transitionStart, to: clickedState.id });
        setShowSymbolInput(true);
      }
      setIsCreatingTransition(false);
      setTransitionStart(null);
    } else if (clickedState) {
      setSelectedState(clickedState.id);
      setSelectedTransition(null);
      setEditingTransition(null);
    } else {
      const clickedTransition = automaton.transitions.findIndex(transition => {
        const fromState = automaton.states.find(s => s.id === transition.from_state);
        const toState = automaton.states.find(s => s.id === transition.to_state);
        if (!fromState || !toState) return false;

        const dx = toState.x - fromState.x;
        const dy = toState.y - fromState.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        
        for (let t = 0.2; t <= 0.8; t += 0.1) {
          const checkX = fromState.x + dx * t;
          const checkY = fromState.y + dy * t;
          const distance = Math.sqrt((x - checkX) ** 2 + (y - checkY) ** 2);
          if (distance <= 25) return true;
        }
        
        return false;
      });

      if (clickedTransition !== -1) {
        setSelectedTransition(clickedTransition);
        setSelectedState(null);
        setEditingTransition(clickedTransition);
        setEditSymbol(automaton.transitions[clickedTransition].symbol);
      } else {
        const newStateId = `q${automaton.states.length}`;
        const newState: State = {
          id: newStateId,
          x,
          y,
          is_start: automaton.states.length === 0,
          is_accept: false,
          label: newStateId,
        };

        onAutomatonChange({
          ...automaton,
          states: [...automaton.states, newState],
        });

        setSelectedState(newStateId);
        setSelectedTransition(null);
        setEditingTransition(null);
      }
    }
  };

  const handleStateDoubleClick = (stateId: string) => {
    const state = automaton.states.find(s => s.id === stateId);
    if (!state) return;

    const newLabel = prompt('Enter state label:', state.label || state.id);
    if (newLabel !== null) {
      onAutomatonChange({
        ...automaton,
        states: automaton.states.map(s =>
          s.id === stateId ? { ...s, label: newLabel } : s
        ),
      });
    }
  };

  const toggleStateStart = () => {
    if (!selectedState) return;

    onAutomatonChange({
      ...automaton,
      states: automaton.states.map(state => ({
        ...state,
        is_start: state.id === selectedState ? !state.is_start : false,
      })),
    });
  };

  const toggleStateAccept = () => {
    if (!selectedState) return;

    onAutomatonChange({
      ...automaton,
      states: automaton.states.map(state =>
        state.id === selectedState ? { ...state, is_accept: !state.is_accept } : state
      ),
    });
  };

  const deleteSelected = () => {
    if (selectedState) {
      onAutomatonChange({
        ...automaton,
        states: automaton.states.filter(state => state.id !== selectedState),
        transitions: automaton.transitions.filter(
          transition => transition.from_state !== selectedState && transition.to_state !== selectedState
        ),
      });
      setSelectedState(null);
    } else if (selectedTransition !== null) {
      onAutomatonChange({
        ...automaton,
        transitions: automaton.transitions.filter((_, index) => index !== selectedTransition),
      });
      setSelectedTransition(null);
    }
  };

  const startTransition = () => {
    if (!selectedState) return;
    setIsCreatingTransition(true);
    setTransitionStart(selectedState);
  };

  const addTransition = () => {
    if (!pendingTransition || !newSymbol.trim()) return;

    const newTransition: Transition = {
      from_state: pendingTransition.from,
      to_state: pendingTransition.to,
      symbol: newSymbol.trim(),
    };

    onAutomatonChange({
      ...automaton,
      transitions: [...automaton.transitions, newTransition],
    });

    setNewSymbol('');
    setShowSymbolInput(false);
    setPendingTransition(null);
  };

  const updateTransitionSymbol = () => {
    if (editingTransition === null || !editSymbol.trim()) return;

    onAutomatonChange({
      ...automaton,
      transitions: automaton.transitions.map((transition, index) =>
        index === editingTransition
          ? { ...transition, symbol: editSymbol.trim() }
          : transition
      ),
    });

    setEditingTransition(null);
    setEditSymbol('');
    setSelectedTransition(null);
  };

  const clearCanvas = () => {
    onAutomatonChange({
      states: [],
      transitions: [],
      alphabet: automaton.alphabet,
    });
    setSelectedState(null);
    setSelectedTransition(null);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <Button
          onClick={toggleStateStart}
          disabled={!selectedState}
          variant={selectedState && automaton.states.find(s => s.id === selectedState)?.is_start ? "default" : "outline"}
          size="sm"
        >
          <Play className="w-4 h-4 mr-1" />
          Start State
        </Button>
        
        <Button
          onClick={toggleStateAccept}
          disabled={!selectedState}
          variant={selectedState && automaton.states.find(s => s.id === selectedState)?.is_accept ? "default" : "outline"}
          size="sm"
        >
          <Square className="w-4 h-4 mr-1" />
          Accept State
        </Button>
        
        <Button
          onClick={startTransition}
          disabled={!selectedState}
          variant="outline"
          size="sm"
        >
          Add Transition
        </Button>
        
        <Button
          onClick={deleteSelected}
          disabled={!selectedState && selectedTransition === null}
          variant="destructive"
          size="sm"
        >
          <Trash2 className="w-4 h-4 mr-1" />
          Delete
        </Button>
        
        <Button
          onClick={clearCanvas}
          variant="outline"
          size="sm"
        >
          <RotateCcw className="w-4 h-4 mr-1" />
          Clear
        </Button>

        {onRequestAIGuidance && (
          <Button
            onClick={onRequestAIGuidance}
            variant="secondary"
            size="sm"
            className="bg-gradient-to-r from-purple-500 to-blue-500 text-white"
          >
            🤖 AI Guidance
          </Button>
        )}
      </div>

      {showSymbolInput && (
        <div className="flex gap-2 items-center p-3 bg-blue-50 rounded-lg">
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
      )}

      {editingTransition !== null && (
        <div className="flex gap-2 items-center p-3 bg-yellow-50 rounded-lg">
          <span className="text-sm font-medium">Edit transition symbol:</span>
          <Input
            value={editSymbol}
            onChange={(e) => setEditSymbol(e.target.value)}
            placeholder="Enter new symbol"
            className="w-40"
            onKeyPress={(e) => e.key === 'Enter' && updateTransitionSymbol()}
            autoFocus
          />
          <Button onClick={updateTransitionSymbol} size="sm">Update</Button>
          <Button onClick={() => { setEditingTransition(null); setEditSymbol(''); setSelectedTransition(null); }} variant="outline" size="sm">Cancel</Button>
        </div>
      )}

      <div className="border-2 border-gray-300 rounded-lg overflow-hidden relative">
        <canvas
          ref={canvasRef}
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          onClick={handleCanvasClick}
          onMouseDown={(e) => {
            const canvas = canvasRef.current;
            if (!canvas) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const clickedState = automaton.states.find(state => {
              const distance = Math.sqrt((x - state.x) ** 2 + (y - state.y) ** 2);
              return distance <= STATE_RADIUS;
            });
            
            if (clickedState) {
              setIsDragging(true);
              setDragStart({ x, y });
              setDraggedState(clickedState.id);
              e.preventDefault();
            }
          }}
          onMouseMove={(e) => {
            if (!isDragging || !draggedState || !dragStart) return;
            
            const canvas = canvasRef.current;
            if (!canvas) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const newX = Math.max(STATE_RADIUS, Math.min(CANVAS_WIDTH - STATE_RADIUS, x));
            const newY = Math.max(STATE_RADIUS, Math.min(CANVAS_HEIGHT - STATE_RADIUS, y));
            
            onAutomatonChange({
              ...automaton,
              states: automaton.states.map(state =>
                state.id === draggedState
                  ? { ...state, x: newX, y: newY }
                  : state
              ),
            });
            
            e.preventDefault();
          }}
          onMouseUp={() => {
            setIsDragging(false);
            setDragStart(null);
            setDraggedState(null);
          }}
          onMouseLeave={() => {
            setIsDragging(false);
            setDragStart(null);
            setDraggedState(null);
          }}
          onDoubleClick={(e) => {
            const canvas = canvasRef.current;
            if (!canvas) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const clickedState = automaton.states.find(state => {
              const distance = Math.sqrt((x - state.x) ** 2 + (y - state.y) ** 2);
              return distance <= STATE_RADIUS;
            });
            if (clickedState) handleStateDoubleClick(clickedState.id);
          }}
          className="cursor-pointer bg-gray-50"
        />
        {showInteractiveOverlay && (
          <InteractiveOverlay
            states={automaton.states}
            transitions={automaton.transitions}
            canvasWidth={CANVAS_WIDTH}
            canvasHeight={CANVAS_HEIGHT}
            stateRadius={STATE_RADIUS}
            stepExplanations={stepExplanations}
            onStateHover={onStateHover}
            onTransitionHover={onTransitionHover}
          />
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        <Badge variant="outline">States: {automaton.states.length}</Badge>
        <Badge variant="outline">Transitions: {automaton.transitions.length}</Badge>
        <Badge variant="outline">Alphabet: {automaton.alphabet.join(', ')}</Badge>
      </div>

      <div className="text-sm text-gray-600 space-y-1">
        <p>• Click on empty space to create a new state</p>
        <p>• Click and drag states to move them around</p>
        <p>• Double-click a state to rename it</p>
        <p>• Select a state and click "Add Transition" to create transitions</p>
        <p>• Click on transition arrows to edit their symbols</p>
        <p>• Use Delete button to remove selected states or transitions</p>
      </div>
    </div>
  );
};
