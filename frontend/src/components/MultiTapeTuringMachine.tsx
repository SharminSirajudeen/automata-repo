import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Separator } from './ui/separator';
import { Plus, Minus, Play, Square, RotateCcw, Edit3 } from 'lucide-react';
import { MultiTapeTMAutomaton, TapeState, MultiTapeTransition, TapeOperation } from '../types/advanced-automata';
import { State } from '../types/automata';

interface MultiTapeTuringMachineProps {
  automaton: MultiTapeTMAutomaton;
  onAutomatonChange: (automaton: MultiTapeTMAutomaton) => void;
  isSimulating?: boolean;
  currentStep?: number;
  readOnly?: boolean;
}

export const MultiTapeTuringMachine: React.FC<MultiTapeTuringMachineProps> = ({
  automaton,
  onAutomatonChange,
  isSimulating = false,
  currentStep = 0,
  readOnly = false
}) => {
  const [selectedTape, setSelectedTape] = useState(0);
  const [tapeStates, setTapeStates] = useState<TapeState[]>([]);
  const [editingTransition, setEditingTransition] = useState<number | null>(null);
  const [newTransition, setNewTransition] = useState<Partial<MultiTapeTransition>>({});
  const [showTransitionEditor, setShowTransitionEditor] = useState(false);

  const TAPE_CELL_WIDTH = 40;
  const TAPE_HEIGHT = 60;
  const VISIBLE_CELLS = 15;

  useEffect(() => {
    // Initialize tape states
    const initialTapes: TapeState[] = Array.from({ length: automaton.num_tapes }, (_, index) => ({
      contents: Array(20).fill(automaton.blank_symbol),
      head_position: 10,
      tape_index: index
    }));
    setTapeStates(initialTapes);
  }, [automaton.num_tapes, automaton.blank_symbol]);

  const addTape = () => {
    if (automaton.num_tapes >= 5) return;
    
    const newAutomaton = {
      ...automaton,
      num_tapes: automaton.num_tapes + 1
    };
    
    onAutomatonChange(newAutomaton);
  };

  const removeTape = () => {
    if (automaton.num_tapes <= 1) return;
    
    const newAutomaton = {
      ...automaton,
      num_tapes: automaton.num_tapes - 1,
      transitions: automaton.transitions.map(t => ({
        ...t,
        tape_operations: t.tape_operations.filter(op => op.tape_index < automaton.num_tapes - 1)
      }))
    };
    
    onAutomatonChange(newAutomaton);
  };

  const renderTapeCell = (content: string, position: number, headPos: number, tapeIndex: number) => {
    const isHead = position === headPos;
    const isInputTape = tapeIndex === automaton.input_tape_index;
    
    return (
      <div
        key={position}
        className={`
          w-10 h-12 border border-gray-300 flex items-center justify-center font-mono text-sm
          ${isHead ? 'border-red-500 border-2 bg-red-50' : 'bg-white'}
          ${isInputTape ? 'border-blue-300' : ''}
        `}
      >
        {content === automaton.blank_symbol ? '□' : content}
      </div>
    );
  };

  const renderTape = (tapeState: TapeState) => {
    const { contents, head_position, tape_index } = tapeState;
    const startPos = Math.max(0, head_position - Math.floor(VISIBLE_CELLS / 2));
    const visibleContents = contents.slice(startPos, startPos + VISIBLE_CELLS);
    const isInputTape = tape_index === automaton.input_tape_index;

    return (
      <Card key={tape_index} className={`mb-4 ${isInputTape ? 'border-blue-400' : ''}`}>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm flex items-center gap-2">
              Tape {tape_index + 1}
              {isInputTape && <Badge variant="outline">Input</Badge>}
            </CardTitle>
            <div className="text-xs text-gray-500">
              Head: {head_position} | Position: {startPos}-{startPos + VISIBLE_CELLS - 1}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex gap-0 overflow-x-auto">
            {visibleContents.map((content, index) => 
              renderTapeCell(content, startPos + index, head_position, tape_index)
            )}
          </div>
          <div className="flex gap-0 mt-1">
            {visibleContents.map((_, index) => (
              <div
                key={index}
                className={`w-10 h-4 flex items-center justify-center text-xs ${
                  startPos + index === head_position ? 'text-red-600' : 'text-transparent'
                }`}
              >
                ↑
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderTransitionEditor = () => {
    if (!showTransitionEditor) return null;

    return (
      <Card className="mb-4 border-blue-300">
        <CardHeader>
          <CardTitle className="text-sm">
            {editingTransition !== null ? 'Edit Transition' : 'Add New Transition'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium">From State</label>
              <Select
                value={newTransition.from_state || ''}
                onValueChange={(value) => setNewTransition(prev => ({ ...prev, from_state: value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select state" />
                </SelectTrigger>
                <SelectContent>
                  {automaton.states.map(state => (
                    <SelectItem key={state.id} value={state.id}>{state.label || state.id}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-sm font-medium">To State</label>
              <Select
                value={newTransition.to_state || ''}
                onValueChange={(value) => setNewTransition(prev => ({ ...prev, to_state: value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select state" />
                </SelectTrigger>
                <SelectContent>
                  {automaton.states.map(state => (
                    <SelectItem key={state.id} value={state.id}>{state.label || state.id}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div>
            <label className="text-sm font-medium mb-2 block">Tape Operations</label>
            <div className="space-y-2 max-h-40 overflow-y-auto">
              {Array.from({ length: automaton.num_tapes }, (_, tapeIndex) => (
                <div key={tapeIndex} className="border rounded p-3 bg-gray-50">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm font-medium">Tape {tapeIndex + 1}:</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    <Input
                      placeholder="Read"
                      value={newTransition.tape_operations?.[tapeIndex]?.read_symbol || ''}
                      onChange={(e) => {
                        const operations = [...(newTransition.tape_operations || [])];
                        if (!operations[tapeIndex]) {
                          operations[tapeIndex] = { tape_index: tapeIndex, read_symbol: '', write_symbol: '', head_direction: 'S' };
                        }
                        operations[tapeIndex].read_symbol = e.target.value;
                        setNewTransition(prev => ({ ...prev, tape_operations: operations }));
                      }}
                    />
                    <Input
                      placeholder="Write"
                      value={newTransition.tape_operations?.[tapeIndex]?.write_symbol || ''}
                      onChange={(e) => {
                        const operations = [...(newTransition.tape_operations || [])];
                        if (!operations[tapeIndex]) {
                          operations[tapeIndex] = { tape_index: tapeIndex, read_symbol: '', write_symbol: '', head_direction: 'S' };
                        }
                        operations[tapeIndex].write_symbol = e.target.value;
                        setNewTransition(prev => ({ ...prev, tape_operations: operations }));
                      }}
                    />
                    <Select
                      value={newTransition.tape_operations?.[tapeIndex]?.head_direction || 'S'}
                      onValueChange={(value: 'L' | 'R' | 'S') => {
                        const operations = [...(newTransition.tape_operations || [])];
                        if (!operations[tapeIndex]) {
                          operations[tapeIndex] = { tape_index: tapeIndex, read_symbol: '', write_symbol: '', head_direction: 'S' };
                        }
                        operations[tapeIndex].head_direction = value;
                        setNewTransition(prev => ({ ...prev, tape_operations: operations }));
                      }}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="L">Left</SelectItem>
                        <SelectItem value="R">Right</SelectItem>
                        <SelectItem value="S">Stay</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="flex gap-2">
            <Button onClick={saveTransition} disabled={!canSaveTransition()}>
              {editingTransition !== null ? 'Update' : 'Add'}
            </Button>
            <Button variant="outline" onClick={cancelTransitionEdit}>
              Cancel
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  };

  const canSaveTransition = () => {
    return newTransition.from_state && 
           newTransition.to_state && 
           newTransition.tape_operations &&
           newTransition.tape_operations.length === automaton.num_tapes &&
           newTransition.tape_operations.every(op => op.read_symbol && op.write_symbol);
  };

  const saveTransition = () => {
    if (!canSaveTransition()) return;

    const transition: MultiTapeTransition = {
      from_state: newTransition.from_state!,
      to_state: newTransition.to_state!,
      tape_operations: newTransition.tape_operations!
    };

    let newTransitions;
    if (editingTransition !== null) {
      newTransitions = automaton.transitions.map((t, index) => 
        index === editingTransition ? transition : t
      );
    } else {
      newTransitions = [...automaton.transitions, transition];
    }

    onAutomatonChange({
      ...automaton,
      transitions: newTransitions
    });

    cancelTransitionEdit();
  };

  const cancelTransitionEdit = () => {
    setShowTransitionEditor(false);
    setEditingTransition(null);
    setNewTransition({});
  };

  const formatTransitionLabel = (transition: MultiTapeTransition) => {
    return transition.tape_operations
      .map(op => `${op.read_symbol};${op.write_symbol},${op.head_direction}`)
      .join('|');
  };

  return (
    <div className="space-y-4">
      {/* Tape Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Multi-Tape Turing Machine
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={removeTape}
                disabled={automaton.num_tapes <= 1 || readOnly}
              >
                <Minus className="w-4 h-4" />
              </Button>
              <Badge variant="outline">{automaton.num_tapes} Tapes</Badge>
              <Button
                size="sm"
                variant="outline"
                onClick={addTape}
                disabled={automaton.num_tapes >= 5 || readOnly}
              >
                <Plus className="w-4 h-4" />
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2 flex-wrap">
            <Select
              value={automaton.input_tape_index.toString()}
              onValueChange={(value) => onAutomatonChange({
                ...automaton,
                input_tape_index: parseInt(value)
              })}
              disabled={readOnly}
            >
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Array.from({ length: automaton.num_tapes }, (_, i) => (
                  <SelectItem key={i} value={i.toString()}>Input Tape: {i + 1}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            {!readOnly && (
              <Button
                size="sm"
                onClick={() => setShowTransitionEditor(true)}
                disabled={showTransitionEditor}
              >
                <Edit3 className="w-4 h-4 mr-1" />
                Add Transition
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Transition Editor */}
      {renderTransitionEditor()}

      {/* Tape Visualizations */}
      <div className="space-y-2">
        {tapeStates.map(tapeState => renderTape(tapeState))}
      </div>

      {/* Transition List */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Transitions ({automaton.transitions.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {automaton.transitions.map((transition, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-2 border rounded hover:bg-gray-50"
              >
                <div className="flex items-center gap-2">
                  <Badge variant="outline">{transition.from_state}</Badge>
                  <span className="text-sm">→</span>
                  <Badge variant="outline">{transition.to_state}</Badge>
                  <span className="text-xs font-mono text-gray-600">
                    {formatTransitionLabel(transition)}
                  </span>
                </div>
                {!readOnly && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setEditingTransition(index);
                      setNewTransition(transition);
                      setShowTransitionEditor(true);
                    }}
                  >
                    <Edit3 className="w-3 h-3" />
                  </Button>
                )}
              </div>
            ))}
            {automaton.transitions.length === 0 && (
              <div className="text-center text-gray-500 text-sm py-4">
                No transitions defined. Click "Add Transition" to start.
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};