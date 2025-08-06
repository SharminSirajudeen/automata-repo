import React, { useState, useRef, useCallback } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Textarea } from './ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Alert, AlertDescription } from './ui/alert';
import { Separator } from './ui/separator';
import { Play, Pause, RotateCcw, Download, Upload, Code, Zap, ArrowLeftRight } from 'lucide-react';
import { UniversalTM, TMAutomaton } from '../types/advanced-automata';

interface UniversalTuringMachineProps {
  universalTM: UniversalTM;
  onUniversalTMChange: (utm: UniversalTM) => void;
  readOnly?: boolean;
}

interface EncodingRule {
  symbol: string;
  encoding: string;
  description: string;
}

interface SimulationState {
  isRunning: boolean;
  step: number;
  currentState: string;
  tapeContents: string[];
  headPosition: number;
  encodedInput: string;
  decodedOutput?: string;
}

export const UniversalTuringMachine: React.FC<UniversalTuringMachineProps> = ({
  universalTM,
  onUniversalTMChange,
  readOnly = false
}) => {
  const [inputTM, setInputTM] = useState<TMAutomaton | null>(null);
  const [inputString, setInputString] = useState('');
  const [encodedTM, setEncodedTM] = useState('');
  const [simulationState, setSimulationState] = useState<SimulationState>({
    isRunning: false,
    step: 0,
    currentState: '',
    tapeContents: [],
    headPosition: 0,
    encodedInput: ''
  });
  const [validationError, setValidationError] = useState<string | null>(null);
  
  const encodingRules: { [scheme: string]: EncodingRule[] } = {
    standard: [
      { symbol: 'q0', encoding: '1', description: 'State q0' },
      { symbol: 'q1', encoding: '11', description: 'State q1' },
      { symbol: 'q2', encoding: '111', description: 'State q2' },
      { symbol: '0', encoding: '1', description: 'Tape symbol 0' },
      { symbol: '1', encoding: '11', description: 'Tape symbol 1' },
      { symbol: 'B', encoding: '111', description: 'Blank symbol' },
      { symbol: 'L', encoding: '1', description: 'Move Left' },
      { symbol: 'R', encoding: '11', description: 'Move Right' },
      { symbol: 'S', encoding: '111', description: 'Stay' },
    ],
    binary: [
      { symbol: 'q0', encoding: '00', description: 'State q0' },
      { symbol: 'q1', encoding: '01', description: 'State q1' },
      { symbol: 'q2', encoding: '10', description: 'State q2' },
      { symbol: '0', encoding: '0', description: 'Tape symbol 0' },
      { symbol: '1', encoding: '1', description: 'Tape symbol 1' },
      { symbol: 'B', encoding: '00', description: 'Blank symbol' },
      { symbol: 'L', encoding: '0', description: 'Move Left' },
      { symbol: 'R', encoding: '1', description: 'Move Right' },
    ],
    decimal: [
      { symbol: 'q0', encoding: '0', description: 'State q0' },
      { symbol: 'q1', encoding: '1', description: 'State q1' },
      { symbol: 'q2', encoding: '2', description: 'State q2' },
      { symbol: '0', encoding: '0', description: 'Tape symbol 0' },
      { symbol: '1', encoding: '1', description: 'Tape symbol 1' },
      { symbol: 'B', encoding: '9', description: 'Blank symbol' },
      { symbol: 'L', encoding: '7', description: 'Move Left' },
      { symbol: 'R', encoding: '8', description: 'Move Right' },
      { symbol: 'S', encoding: '9', description: 'Stay' },
    ]
  };

  const encodeTuringMachine = useCallback((tm: TMAutomaton): string => {
    const rules = encodingRules[universalTM.encoding_scheme];
    if (!rules) return '';

    const separator = universalTM.encoding_scheme === 'standard' ? '0' : 
                     universalTM.encoding_scheme === 'binary' ? '11' : ',';

    const encodedTransitions = tm.transitions.map(transition => {
      const fromState = rules.find(r => r.symbol === transition.from_state)?.encoding || '';
      const toState = rules.find(r => r.symbol === transition.to_state)?.encoding || '';
      const readSymbol = rules.find(r => r.symbol === transition.read_symbol)?.encoding || '';
      const writeSymbol = rules.find(r => r.symbol === transition.write_symbol)?.encoding || '';
      const direction = rules.find(r => r.symbol === transition.head_direction)?.encoding || '';
      
      return [fromState, readSymbol, toState, writeSymbol, direction].join(separator);
    });

    return encodedTransitions.join(separator + separator);
  }, [universalTM.encoding_scheme]);

  const decodeTuringMachine = useCallback((encoded: string): TMAutomaton | null => {
    try {
      const rules = encodingRules[universalTM.encoding_scheme];
      if (!rules) return null;

      const separator = universalTM.encoding_scheme === 'standard' ? '0' : 
                       universalTM.encoding_scheme === 'binary' ? '11' : ',';

      // This is a simplified decoder - real implementation would be more complex
      const transitionStrings = encoded.split(separator + separator);
      const transitions = transitionStrings.map(transStr => {
        const parts = transStr.split(separator);
        if (parts.length !== 5) throw new Error('Invalid transition encoding');

        const fromState = rules.find(r => r.encoding === parts[0])?.symbol || 'q0';
        const readSymbol = rules.find(r => r.encoding === parts[1])?.symbol || '0';
        const toState = rules.find(r => r.encoding === parts[2])?.symbol || 'q0';
        const writeSymbol = rules.find(r => r.encoding === parts[3])?.symbol || '0';
        const direction = rules.find(r => r.encoding === parts[4])?.symbol as 'L' | 'R' | 'S' || 'R';

        return {
          from_state: fromState,
          to_state: toState,
          read_symbol: readSymbol,
          write_symbol: writeSymbol,
          head_direction: direction
        };
      });

      // Extract unique states and symbols
      const states = Array.from(new Set([
        ...transitions.map(t => t.from_state),
        ...transitions.map(t => t.to_state)
      ])).map((id, index) => ({
        id,
        x: 100 + (index % 5) * 80,
        y: 100 + Math.floor(index / 5) * 80,
        is_start: id === 'q0',
        is_accept: false, // Would need to be encoded separately
        label: id
      }));

      const tape_alphabet = Array.from(new Set([
        ...transitions.map(t => t.read_symbol),
        ...transitions.map(t => t.write_symbol)
      ]));

      return {
        type: 'tm',
        states,
        transitions,
        tape_alphabet,
        blank_symbol: 'B'
      };
    } catch (error) {
      setValidationError(`Decoding error: ${error}`);
      return null;
    }
  }, [universalTM.encoding_scheme]);

  const handleEncodeCurrentTM = () => {
    if (!inputTM) return;
    const encoded = encodeTuringMachine(inputTM);
    setEncodedTM(encoded);
    onUniversalTMChange({
      ...universalTM,
      encoded_tm: encoded
    });
  };

  const handleDecodeToTM = () => {
    if (!encodedTM.trim()) return;
    const decoded = decodeTuringMachine(encodedTM);
    if (decoded) {
      setInputTM(decoded);
      setValidationError(null);
    }
  };

  const startSimulation = async () => {
    if (!universalTM.encoded_tm || !inputString) return;

    try {
      setSimulationState(prev => ({ ...prev, isRunning: true }));
      
      const response = await fetch('/api/universal-tm-simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          encoded_tm: universalTM.encoded_tm,
          input_string: inputString,
          encoding_scheme: universalTM.encoding_scheme
        })
      });

      if (response.ok) {
        const result = await response.json();
        setSimulationState({
          isRunning: false,
          step: result.steps,
          currentState: result.final_state,
          tapeContents: result.tape_contents,
          headPosition: result.head_position,
          encodedInput: result.encoded_input,
          decodedOutput: result.decoded_output
        });
      }
    } catch (error) {
      console.error('Simulation error:', error);
      setSimulationState(prev => ({ ...prev, isRunning: false }));
    }
  };

  const renderEncodingTable = () => {
    const rules = encodingRules[universalTM.encoding_scheme];
    
    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse border border-gray-300">
          <thead>
            <tr className="bg-gray-50">
              <th className="border border-gray-300 px-3 py-2 text-left">Symbol</th>
              <th className="border border-gray-300 px-3 py-2 text-left">Encoding</th>
              <th className="border border-gray-300 px-3 py-2 text-left">Description</th>
            </tr>
          </thead>
          <tbody>
            {rules.map((rule, index) => (
              <tr key={index} className="hover:bg-gray-50">
                <td className="border border-gray-300 px-3 py-2 font-mono">{rule.symbol}</td>
                <td className="border border-gray-300 px-3 py-2 font-mono">{rule.encoding}</td>
                <td className="border border-gray-300 px-3 py-2">{rule.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const renderTMEditor = () => {
    if (!inputTM) {
      return (
        <div className="text-center py-8 text-gray-500">
          <p>No Turing Machine loaded</p>
          <p className="text-sm">Import a TM or decode from an encoded string</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">States ({inputTM.states.length})</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-1">
                {inputTM.states.map(state => (
                  <Badge 
                    key={state.id} 
                    variant={state.is_start ? "default" : state.is_accept ? "secondary" : "outline"}
                  >
                    {state.label || state.id}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Tape Alphabet</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-1">
                {inputTM.tape_alphabet.map(symbol => (
                  <Badge key={symbol} variant="outline" className="font-mono">
                    {symbol === ' ' ? '□' : symbol}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Transitions ({inputTM.transitions.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {inputTM.transitions.map((transition, index) => (
                <div key={index} className="font-mono text-xs bg-gray-50 p-2 rounded">
                  δ({transition.from_state}, {transition.read_symbol}) = 
                  ({transition.to_state}, {transition.write_symbol}, {transition.head_direction})
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderSimulationResults = () => {
    if (!simulationState.encodedInput) return null;

    return (
      <div className="space-y-4">
        <Alert>
          <Zap className="h-4 w-4" />
          <AlertDescription>
            Simulation completed in {simulationState.step} steps
          </AlertDescription>
        </Alert>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Input Encoding</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="font-mono text-xs bg-gray-50 p-2 rounded break-all">
                {simulationState.encodedInput}
              </div>
            </CardContent>
          </Card>

          {simulationState.decodedOutput && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Decoded Output</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="font-mono text-xs bg-gray-50 p-2 rounded">
                  {simulationState.decodedOutput}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Final Tape State</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-1 overflow-x-auto">
              {simulationState.tapeContents.map((symbol, index) => (
                <div
                  key={index}
                  className={`min-w-[2rem] h-8 border-2 flex items-center justify-center font-mono text-xs ${
                    index === simulationState.headPosition
                      ? 'border-red-500 bg-red-100'
                      : 'border-gray-300 bg-white'
                  }`}
                >
                  {symbol === ' ' ? '□' : symbol}
                </div>
              ))}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Head Position: {simulationState.headPosition}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Universal Turing Machine
            <div className="flex items-center gap-2">
              <Badge variant="outline">{universalTM.encoding_scheme}</Badge>
              <Select
                value={universalTM.encoding_scheme}
                onValueChange={(value: 'standard' | 'binary' | 'decimal') =>
                  onUniversalTMChange({ ...universalTM, encoding_scheme: value })
                }
                disabled={readOnly}
              >
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="standard">Standard</SelectItem>
                  <SelectItem value="binary">Binary</SelectItem>
                  <SelectItem value="decimal">Decimal</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="encoding" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="encoding">Encoding</TabsTrigger>
              <TabsTrigger value="machine">Machine</TabsTrigger>
              <TabsTrigger value="simulation">Simulation</TabsTrigger>
              <TabsTrigger value="results">Results</TabsTrigger>
            </TabsList>

            <TabsContent value="encoding" className="space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Encoding Rules</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {renderEncodingTable()}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Encoded Turing Machine</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Textarea
                      value={encodedTM}
                      onChange={(e) => setEncodedTM(e.target.value)}
                      placeholder={`Enter encoded TM using ${universalTM.encoding_scheme} scheme...`}
                      className="font-mono text-xs"
                      rows={6}
                      readOnly={readOnly}
                    />
                    {!readOnly && (
                      <div className="flex gap-2">
                        <Button size="sm" onClick={handleDecodeToTM}>
                          <ArrowLeftRight className="w-4 h-4 mr-1" />
                          Decode
                        </Button>
                        <Button size="sm" variant="outline" onClick={handleEncodeCurrentTM}>
                          <Code className="w-4 h-4 mr-1" />
                          Encode Current
                        </Button>
                      </div>
                    )}
                    {validationError && (
                      <Alert variant="destructive">
                        <AlertDescription>{validationError}</AlertDescription>
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="machine" className="space-y-4">
              <div className="flex gap-2 mb-4">
                <Button size="sm" variant="outline">
                  <Upload className="w-4 h-4 mr-1" />
                  Import TM
                </Button>
                <Button size="sm" variant="outline" disabled={!inputTM}>
                  <Download className="w-4 h-4 mr-1" />
                  Export TM
                </Button>
              </div>
              {renderTMEditor()}
            </TabsContent>

            <TabsContent value="simulation" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Simulation Controls</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex gap-2">
                    <Input
                      value={inputString}
                      onChange={(e) => setInputString(e.target.value)}
                      placeholder="Enter input string..."
                      className="flex-1"
                      readOnly={readOnly}
                    />
                    <Button 
                      onClick={startSimulation}
                      disabled={!universalTM.encoded_tm || !inputString || simulationState.isRunning}
                    >
                      {simulationState.isRunning ? (
                        <><Pause className="w-4 h-4 mr-1" />Running</>
                      ) : (
                        <><Play className="w-4 h-4 mr-1" />Simulate</>
                      )}
                    </Button>
                  </div>
                  
                  <div className="text-sm text-gray-600">
                    <p>• The Universal TM will simulate the encoded machine on your input</p>
                    <p>• Input will be encoded according to the selected scheme</p>
                    <p>• Output will be decoded back to readable format</p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="results">
              {simulationState.encodedInput ? renderSimulationResults() : (
                <div className="text-center py-8 text-gray-500">
                  <p>No simulation results yet</p>
                  <p className="text-sm">Run a simulation to see results here</p>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};