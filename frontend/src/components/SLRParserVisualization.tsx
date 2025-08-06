import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Alert, AlertDescription } from './ui/alert';
import { Play, Pause, SkipForward, SkipBack, RotateCcw, Eye, Grid } from 'lucide-react';
import { SLRParser, SLRParserState, LRItem, Action, UnrestrictedGrammar } from '../types/advanced-automata';

interface SLRParserVisualizationProps {
  parser: SLRParser;
  onParserChange?: (parser: SLRParser) => void;
  readOnly?: boolean;
}

interface ParseStep {
  step: number;
  stack: string[];
  input: string[];
  action: Action;
  production?: string;
  currentState: string;
}

export const SLRParserVisualization: React.FC<SLRParserVisualizationProps> = ({
  parser,
  onParserChange,
  readOnly = false
}) => {
  const [inputString, setInputString] = useState('');
  const [parseSteps, setParseSteps] = useState<ParseStep[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedState, setSelectedState] = useState<string | null>(null);
  const [showDFA, setShowDFA] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const STATE_RADIUS = 40;
  const CANVAS_WIDTH = 1000;
  const CANVAS_HEIGHT = 600;

  useEffect(() => {
    drawDFA();
  }, [parser, selectedState]);

  useEffect(() => {
    let interval: any;
    if (isPlaying && currentStep < parseSteps.length - 1) {
      interval = setInterval(() => {
        setCurrentStep(prev => prev + 1);
      }, 1000);
    } else if (currentStep >= parseSteps.length - 1) {
      setIsPlaying(false);
    }
    return () => clearInterval(interval);
  }, [isPlaying, currentStep, parseSteps.length]);

  const drawDFA = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Position states in a grid layout
    const cols = Math.ceil(Math.sqrt(parser.states.length));
    const rows = Math.ceil(parser.states.length / cols);
    const spacingX = (CANVAS_WIDTH - 100) / cols;
    const spacingY = (CANVAS_HEIGHT - 100) / rows;

    // Draw transitions first
    parser.states.forEach((state, stateIndex) => {
      const col = stateIndex % cols;
      const row = Math.floor(stateIndex / cols);
      const fromX = 50 + col * spacingX + spacingX / 2;
      const fromY = 50 + row * spacingY + spacingY / 2;

      Object.entries(state.transitions).forEach(([symbol, toStateId]) => {
        const toStateIndex = parser.states.findIndex(s => s.id === toStateId);
        if (toStateIndex === -1) return;

        const toCol = toStateIndex % cols;
        const toRow = Math.floor(toStateIndex / cols);
        const toX = 50 + toCol * spacingX + spacingX / 2;
        const toY = 50 + toRow * spacingY + spacingY / 2;

        // Skip self-loops for clarity in compact view
        if (fromX === toX && fromY === toY) return;

        const angle = Math.atan2(toY - fromY, toX - fromX);
        const startX = fromX + Math.cos(angle) * STATE_RADIUS;
        const startY = fromY + Math.sin(angle) * STATE_RADIUS;
        const endX = toX - Math.cos(angle) * STATE_RADIUS;
        const endY = toY - Math.sin(angle) * STATE_RADIUS;

        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        // Draw arrow head
        const arrowLength = 8;
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

        // Draw transition label
        const midX = (startX + endX) / 2;
        const midY = (startY + endY) / 2;
        ctx.fillStyle = '#374151';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(symbol, midX, midY - 5);
      });
    });

    // Draw states
    parser.states.forEach((state, index) => {
      const col = index % cols;
      const row = Math.floor(index / cols);
      const x = 50 + col * spacingX + spacingX / 2;
      const y = 50 + row * spacingY + spacingY / 2;

      const isSelected = selectedState === state.id;
      const isCurrentStep = parseSteps[currentStep]?.currentState === state.id;

      ctx.fillStyle = isCurrentStep ? '#ef4444' : 
                     isSelected ? '#3b82f6' : '#ffffff';
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = isSelected ? 3 : 2;

      ctx.beginPath();
      ctx.arc(x, y, STATE_RADIUS, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();

      // State label
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(state.id, x, y + 5);
    });
  }, [parser, selectedState, parseSteps, currentStep]);

  const startParsing = async () => {
    if (!inputString.trim()) return;

    try {
      const response = await fetch('/api/slr-parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parser,
          input_string: inputString
        })
      });

      if (response.ok) {
        const result = await response.json();
        setParseSteps(result.steps);
        setCurrentStep(0);
        setIsPlaying(false);
      }
    } catch (error) {
      console.error('Parse error:', error);
    }
  };

  const formatAction = (action: Action) => {
    switch (action.type) {
      case 'shift':
        return `S${action.value}`;
      case 'reduce':
        return `R${action.value}`;
      case 'accept':
        return 'ACC';
      case 'error':
        return 'ERR';
      default:
        return '';
    }
  };

  const renderActionTable = () => {
    const terminals = parser.grammar.terminals.concat(['$']);
    
    return (
      <div className="overflow-x-auto">
        <Table className="text-xs">
          <TableHeader>
            <TableRow>
              <TableHead className="w-16">State</TableHead>
              {terminals.map(terminal => (
                <TableHead key={terminal} className="text-center min-w-16">
                  {terminal}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {parser.states.map(state => (
              <TableRow 
                key={state.id}
                className={selectedState === state.id ? 'bg-blue-50' : ''}
              >
                <TableCell 
                  className="font-medium cursor-pointer"
                  onClick={() => setSelectedState(selectedState === state.id ? null : state.id)}
                >
                  {state.id}
                </TableCell>
                {terminals.map(terminal => (
                  <TableCell key={terminal} className="text-center">
                    {parser.action_table[state.id]?.[terminal] ? 
                      formatAction(parser.action_table[state.id][terminal]) : ''}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    );
  };

  const renderGotoTable = () => {
    return (
      <div className="overflow-x-auto">
        <Table className="text-xs">
          <TableHeader>
            <TableRow>
              <TableHead className="w-16">State</TableHead>
              {parser.grammar.non_terminals.map(nonTerminal => (
                <TableHead key={nonTerminal} className="text-center min-w-16">
                  {nonTerminal}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {parser.states.map(state => (
              <TableRow 
                key={state.id}
                className={selectedState === state.id ? 'bg-blue-50' : ''}
              >
                <TableCell 
                  className="font-medium cursor-pointer"
                  onClick={() => setSelectedState(selectedState === state.id ? null : state.id)}
                >
                  {state.id}
                </TableCell>
                {parser.grammar.non_terminals.map(nonTerminal => (
                  <TableCell key={nonTerminal} className="text-center">
                    {parser.goto_table[state.id]?.[nonTerminal] || ''}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    );
  };

  const renderLRItems = (state: SLRParserState) => {
    return (
      <div className="space-y-1">
        {state.items.map((item, index) => {
          const leftSide = item.production.left_side.join('');
          const rightSide = item.production.right_side;
          const beforeDot = rightSide.slice(0, item.dot_position).join('');
          const afterDot = rightSide.slice(item.dot_position).join('');
          
          return (
            <div key={index} className="font-mono text-xs bg-gray-50 p-1 rounded">
              {leftSide} → {beforeDot}•{afterDot}
              {item.lookahead && item.lookahead.length > 0 && (
                <span className="text-gray-500 ml-2">, {item.lookahead.join('/')}</span>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const renderParseTrace = () => {
    if (parseSteps.length === 0) return null;

    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 mb-4">
          <Button
            size="sm"
            variant="outline"
            onClick={() => setCurrentStep(0)}
            disabled={currentStep === 0}
          >
            <RotateCcw className="w-3 h-3" />
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setCurrentStep(prev => Math.max(0, prev - 1))}
            disabled={currentStep === 0}
          >
            <SkipBack className="w-3 h-3" />
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setIsPlaying(!isPlaying)}
          >
            {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setCurrentStep(prev => Math.min(parseSteps.length - 1, prev + 1))}
            disabled={currentStep >= parseSteps.length - 1}
          >
            <SkipForward className="w-3 h-3" />
          </Button>
          <Badge variant="outline">
            Step {currentStep + 1} of {parseSteps.length}
          </Badge>
        </div>

        <div className="overflow-x-auto">
          <Table className="text-sm">
            <TableHeader>
              <TableRow>
                <TableHead>Step</TableHead>
                <TableHead>Stack</TableHead>
                <TableHead>Input</TableHead>
                <TableHead>Action</TableHead>
                <TableHead>Production</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {parseSteps.slice(0, currentStep + 1).map((step, index) => (
                <TableRow 
                  key={index}
                  className={index === currentStep ? 'bg-blue-50' : ''}
                >
                  <TableCell>{step.step}</TableCell>
                  <TableCell className="font-mono">
                    {step.stack.join(' ')}
                  </TableCell>
                  <TableCell className="font-mono">
                    {step.input.join('')}
                  </TableCell>
                  <TableCell>
                    <Badge variant={
                      step.action.type === 'accept' ? 'default' :
                      step.action.type === 'error' ? 'destructive' : 'secondary'
                    }>
                      {formatAction(step.action)}
                    </Badge>
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {step.production || ''}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            SLR(1) Parser Visualization
            <Badge variant="outline">
              {parser.states.length} States
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2 mb-4">
            <Input
              value={inputString}
              onChange={(e) => setInputString(e.target.value)}
              placeholder="Enter string to parse..."
              className="flex-1"
            />
            <Button onClick={startParsing} disabled={!inputString.trim()}>
              <Play className="w-4 h-4 mr-1" />
              Parse
            </Button>
          </div>

          <Tabs defaultValue="dfa" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="dfa">DFA</TabsTrigger>
              <TabsTrigger value="action">ACTION Table</TabsTrigger>
              <TabsTrigger value="goto">GOTO Table</TabsTrigger>
              <TabsTrigger value="trace">Parse Trace</TabsTrigger>
            </TabsList>

            <TabsContent value="dfa" className="space-y-4">
              <div className="flex items-center gap-2 mb-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setShowDFA(!showDFA)}
                >
                  <Eye className="w-4 h-4 mr-1" />
                  {showDFA ? 'Hide' : 'Show'} DFA
                </Button>
                {selectedState && (
                  <Badge variant="outline">
                    Selected: {selectedState}
                  </Badge>
                )}
              </div>

              {showDFA && (
                <div className="border rounded-lg overflow-hidden">
                  <canvas
                    ref={canvasRef}
                    width={CANVAS_WIDTH}
                    height={CANVAS_HEIGHT}
                    className="w-full bg-gray-50 cursor-pointer"
                    onClick={(e) => {
                      const canvas = canvasRef.current;
                      if (!canvas) return;

                      const rect = canvas.getBoundingClientRect();
                      const x = e.clientX - rect.left;
                      const y = e.clientY - rect.top;

                      const cols = Math.ceil(Math.sqrt(parser.states.length));
                      const spacingX = (CANVAS_WIDTH - 100) / cols;
                      const spacingY = (CANVAS_HEIGHT - 100) / Math.ceil(parser.states.length / cols);

                      parser.states.forEach((state, index) => {
                        const col = index % cols;
                        const row = Math.floor(index / cols);
                        const stateX = 50 + col * spacingX + spacingX / 2;
                        const stateY = 50 + row * spacingY + spacingY / 2;

                        const distance = Math.sqrt((x - stateX) ** 2 + (y - stateY) ** 2);
                        if (distance <= STATE_RADIUS) {
                          setSelectedState(selectedState === state.id ? null : state.id);
                        }
                      });
                    }}
                  />
                </div>
              )}

              {selectedState && (
                <Card className="mt-4">
                  <CardHeader>
                    <CardTitle className="text-sm">State {selectedState} - LR Items</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {renderLRItems(parser.states.find(s => s.id === selectedState)!)}
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="action">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Grid className="w-4 h-4" />
                    ACTION Table
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {renderActionTable()}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="goto">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Grid className="w-4 h-4" />
                    GOTO Table
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {renderGotoTable()}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="trace">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Parse Trace</CardTitle>
                </CardHeader>
                <CardContent>
                  {parseSteps.length > 0 ? renderParseTrace() : (
                    <div className="text-center text-gray-500 py-8">
                      Enter a string and click "Parse" to see the step-by-step trace
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};