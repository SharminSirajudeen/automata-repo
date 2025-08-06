import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Slider } from './ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Separator } from './ui/separator';
import { Switch } from './ui/switch';
import { Play, Download, RotateCcw, Settings, Palette, Eye, Layers3 } from 'lucide-react';
import { LSystem, TurtleCommand } from '../types/advanced-automata';

interface LSystemRendererProps {
  lsystem: LSystem;
  onLSystemChange: (lsystem: LSystem) => void;
  readOnly?: boolean;
}

interface TurtleState {
  x: number;
  y: number;
  z: number;
  angle: number;
  elevation: number;
  penDown: boolean;
}

interface RenderSettings {
  lineWidth: number;
  colorScheme: string;
  backgroundColor: string;
  animate: boolean;
  animationSpeed: number;
  show3D: boolean;
  perspective: number;
}

const presetLSystems: { [key: string]: LSystem } = {
  sierpinski: {
    type: 'l-system',
    axiom: 'F-G-G',
    productions: {
      'F': 'F-G+F+G-F',
      'G': 'GG'
    },
    iterations: 4,
    angle: 120,
    turtle_commands: [
      { symbol: 'F', action: 'forward', value: 10 },
      { symbol: 'G', action: 'forward', value: 10 },
      { symbol: '+', action: 'turn_left' },
      { symbol: '-', action: 'turn_right' }
    ],
    render_3d: false
  },
  plant: {
    type: 'l-system',
    axiom: 'X',
    productions: {
      'X': 'F+[[X]-X]-F[-FX]+X',
      'F': 'FF'
    },
    iterations: 5,
    angle: 25,
    turtle_commands: [
      { symbol: 'F', action: 'forward', value: 8 },
      { symbol: '+', action: 'turn_left' },
      { symbol: '-', action: 'turn_right' },
      { symbol: '[', action: 'push' },
      { symbol: ']', action: 'pop' }
    ],
    render_3d: false
  },
  tree3d: {
    type: 'l-system',
    axiom: 'A',
    productions: {
      'A': '[&FL!A]/////[&FL!A]///////[&FL!A]',
      'F': 'S ///// F',
      'S': 'FL'
    },
    iterations: 4,
    angle: 22.5,
    turtle_commands: [
      { symbol: 'F', action: 'forward', value: 12 },
      { symbol: 'S', action: 'forward', value: 6 },
      { symbol: '+', action: 'turn_left' },
      { symbol: '-', action: 'turn_right' },
      { symbol: '&', action: 'down' },
      { symbol: '^', action: 'up' },
      { symbol: '/', action: 'turn_left' },
      { symbol: '\\', action: 'turn_right' },
      { symbol: '[', action: 'push' },
      { symbol: ']', action: 'pop' }
    ],
    render_3d: true
  }
};

export const LSystemRenderer: React.FC<LSystemRendererProps> = ({
  lsystem,
  onLSystemChange,
  readOnly = false
}) => {
  const [generatedString, setGeneratedString] = useState('');
  const [renderSettings, setRenderSettings] = useState<RenderSettings>({
    lineWidth: 2,
    colorScheme: 'gradient',
    backgroundColor: '#ffffff',
    animate: false,
    animationSpeed: 50,
    show3D: false,
    perspective: 500
  });
  const [isRendering, setIsRendering] = useState(false);
  const [renderProgress, setRenderProgress] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvas3DRef = useRef<HTMLCanvasElement>(null);

  const CANVAS_WIDTH = 800;
  const CANVAS_HEIGHT = 600;

  useEffect(() => {
    generateString();
  }, [lsystem.axiom, lsystem.productions, lsystem.iterations]);

  useEffect(() => {
    if (generatedString) {
      renderLSystem();
    }
  }, [generatedString, renderSettings, lsystem.turtle_commands, lsystem.angle]);

  const generateString = () => {
    let current = lsystem.axiom;
    
    for (let i = 0; i < lsystem.iterations; i++) {
      let next = '';
      for (const char of current) {
        next += lsystem.productions[char] || char;
      }
      current = next;
    }
    
    setGeneratedString(current);
  };

  const renderLSystem = useCallback(async () => {
    const canvas = lsystem.render_3d ? canvas3DRef.current : canvasRef.current;
    if (!canvas || !generatedString) return;

    setIsRendering(true);
    setRenderProgress(0);

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    ctx.fillStyle = renderSettings.backgroundColor;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Initialize turtle
    const turtle: TurtleState = {
      x: CANVAS_WIDTH / 2,
      y: CANVAS_HEIGHT - 50,
      z: 0,
      angle: -90, // Start pointing up
      elevation: 0,
      penDown: true
    };

    const stateStack: TurtleState[] = [];
    const lines: Array<{
      x1: number; y1: number; z1: number;
      x2: number; y2: number; z2: number;
      generation: number;
    }> = [];

    let currentGeneration = 0;
    let stepCount = 0;
    const totalSteps = generatedString.length;

    // Process each symbol
    for (const symbol of generatedString) {
      const command = lsystem.turtle_commands.find(cmd => cmd.symbol === symbol);
      
      if (command) {
        const oldX = turtle.x;
        const oldY = turtle.y;
        const oldZ = turtle.z;

        switch (command.action) {
          case 'forward':
            const distance = command.value || 10;
            if (lsystem.render_3d) {
              turtle.x += distance * Math.cos(turtle.angle * Math.PI / 180) * Math.cos(turtle.elevation * Math.PI / 180);
              turtle.y += distance * Math.sin(turtle.angle * Math.PI / 180) * Math.cos(turtle.elevation * Math.PI / 180);
              turtle.z += distance * Math.sin(turtle.elevation * Math.PI / 180);
            } else {
              turtle.x += distance * Math.cos(turtle.angle * Math.PI / 180);
              turtle.y += distance * Math.sin(turtle.angle * Math.PI / 180);
            }
            
            if (turtle.penDown) {
              lines.push({
                x1: oldX, y1: oldY, z1: oldZ,
                x2: turtle.x, y2: turtle.y, z2: turtle.z,
                generation: currentGeneration
              });
            }
            break;

          case 'turn_left':
            turtle.angle += lsystem.angle;
            break;

          case 'turn_right':
            turtle.angle -= lsystem.angle;
            break;

          case 'up':
            turtle.elevation += lsystem.angle;
            break;

          case 'down':
            turtle.elevation -= lsystem.angle;
            break;

          case 'push':
            stateStack.push({ ...turtle });
            currentGeneration++;
            break;

          case 'pop':
            if (stateStack.length > 0) {
              const savedState = stateStack.pop()!;
              turtle.x = savedState.x;
              turtle.y = savedState.y;
              turtle.z = savedState.z;
              turtle.angle = savedState.angle;
              turtle.elevation = savedState.elevation;
              turtle.penDown = savedState.penDown;
              currentGeneration = Math.max(0, currentGeneration - 1);
            }
            break;
        }
      }

      stepCount++;
      setRenderProgress((stepCount / totalSteps) * 100);

      // Yield control periodically for smooth animation
      if (renderSettings.animate && stepCount % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, renderSettings.animationSpeed));
      }
    }

    // Render all lines
    renderLines(ctx, lines);
    setIsRendering(false);
    setRenderProgress(100);
  }, [generatedString, lsystem, renderSettings]);

  const renderLines = (ctx: CanvasRenderingContext2D, lines: Array<any>) => {
    ctx.lineWidth = renderSettings.lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    lines.forEach((line, index) => {
      // Apply color scheme
      if (renderSettings.colorScheme === 'gradient') {
        const hue = (line.generation * 30) % 360;
        ctx.strokeStyle = `hsl(${hue}, 70%, 50%)`;
      } else if (renderSettings.colorScheme === 'depth') {
        const intensity = Math.max(0, Math.min(255, 128 + line.z));
        ctx.strokeStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
      } else {
        ctx.strokeStyle = '#2563eb';
      }

      if (lsystem.render_3d && renderSettings.show3D) {
        // Simple 3D projection
        const projectedX1 = line.x1 + (line.z1 / renderSettings.perspective) * 100;
        const projectedY1 = line.y1 + (line.z1 / renderSettings.perspective) * 50;
        const projectedX2 = line.x2 + (line.z2 / renderSettings.perspective) * 100;
        const projectedY2 = line.y2 + (line.z2 / renderSettings.perspective) * 50;

        ctx.beginPath();
        ctx.moveTo(projectedX1, projectedY1);
        ctx.lineTo(projectedX2, projectedY2);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.moveTo(line.x1, line.y1);
        ctx.lineTo(line.x2, line.y2);
        ctx.stroke();
      }
    });
  };

  const loadPreset = (presetName: string) => {
    const preset = presetLSystems[presetName];
    if (preset) {
      onLSystemChange(preset);
    }
  };

  const exportImage = () => {
    const canvas = lsystem.render_3d ? canvas3DRef.current : canvasRef.current;
    if (!canvas) return;

    const link = document.createElement('a');
    link.download = `lsystem-${Date.now()}.png`;
    link.href = canvas.toDataURL();
    link.click();
  };

  const addProduction = (symbol: string, rule: string) => {
    if (!symbol || !rule) return;
    
    onLSystemChange({
      ...lsystem,
      productions: {
        ...lsystem.productions,
        [symbol]: rule
      }
    });
  };

  const removeProduction = (symbol: string) => {
    const newProductions = { ...lsystem.productions };
    delete newProductions[symbol];
    onLSystemChange({
      ...lsystem,
      productions: newProductions
    });
  };

  const addTurtleCommand = (symbol: string, action: string, value?: number) => {
    if (!symbol || !action) return;

    const newCommand: TurtleCommand = {
      symbol,
      action: action as any,
      value
    };

    onLSystemChange({
      ...lsystem,
      turtle_commands: [...lsystem.turtle_commands, newCommand]
    });
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            L-System Graphics Renderer
            <div className="flex items-center gap-2">
              <Badge variant="outline">
                Gen: {lsystem.iterations}
              </Badge>
              <Badge variant="outline">
                {generatedString.length} symbols
              </Badge>
              {lsystem.render_3d && (
                <Badge variant="secondary">3D</Badge>
              )}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="system" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="system">System</TabsTrigger>
              <TabsTrigger value="render">Render</TabsTrigger>
              <TabsTrigger value="settings">Settings</TabsTrigger>
              <TabsTrigger value="presets">Presets</TabsTrigger>
            </TabsList>

            <TabsContent value="system" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">L-System Definition</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <label className="text-sm font-medium">Axiom</label>
                      <Input
                        value={lsystem.axiom}
                        onChange={(e) => onLSystemChange({ ...lsystem, axiom: e.target.value })}
                        placeholder="F"
                        readOnly={readOnly}
                      />
                    </div>
                    
                    <div>
                      <label className="text-sm font-medium">Iterations</label>
                      <Slider
                        value={[lsystem.iterations]}
                        onValueChange={([value]) => onLSystemChange({ ...lsystem, iterations: value })}
                        min={1}
                        max={8}
                        step={1}
                        disabled={readOnly}
                      />
                      <div className="text-xs text-gray-500 mt-1">
                        Current: {lsystem.iterations}
                      </div>
                    </div>

                    <div>
                      <label className="text-sm font-medium">Angle (degrees)</label>
                      <Slider
                        value={[lsystem.angle]}
                        onValueChange={([value]) => onLSystemChange({ ...lsystem, angle: value })}
                        min={1}
                        max={180}
                        step={1}
                        disabled={readOnly}
                      />
                      <div className="text-xs text-gray-500 mt-1">
                        Current: {lsystem.angle}°
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={lsystem.render_3d}
                        onCheckedChange={(checked) => onLSystemChange({ ...lsystem, render_3d: checked })}
                        disabled={readOnly}
                      />
                      <label className="text-sm font-medium">3D Rendering</label>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Production Rules</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {Object.entries(lsystem.productions).map(([symbol, rule]) => (
                      <div key={symbol} className="flex items-center gap-2 p-2 border rounded">
                        <span className="font-mono text-sm w-8">{symbol}</span>
                        <span className="text-sm">→</span>
                        <code className="flex-1 text-xs bg-gray-50 p-1 rounded">{rule}</code>
                        {!readOnly && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => removeProduction(symbol)}
                          >
                            ×
                          </Button>
                        )}
                      </div>
                    ))}
                    
                    {!readOnly && (
                      <div className="flex gap-2 pt-2">
                        <Input
                          placeholder="Symbol"
                          className="w-20"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              const symbol = e.currentTarget.value;
                              const rule = (e.currentTarget.nextSibling as HTMLInputElement)?.value;
                              if (symbol && rule) {
                                addProduction(symbol, rule);
                                e.currentTarget.value = '';
                                (e.currentTarget.nextSibling as HTMLInputElement).value = '';
                              }
                            }
                          }}
                        />
                        <Input
                          placeholder="Rule"
                          className="flex-1"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              const rule = e.currentTarget.value;
                              const symbol = (e.currentTarget.previousSibling as HTMLInputElement)?.value;
                              if (symbol && rule) {
                                addProduction(symbol, rule);
                                e.currentTarget.value = '';
                                (e.currentTarget.previousSibling as HTMLInputElement).value = '';
                              }
                            }
                          }}
                        />
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Turtle Commands</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {lsystem.turtle_commands.map((cmd, index) => (
                      <div key={index} className="flex items-center gap-2 p-2 border rounded text-sm">
                        <span className="font-mono w-8">{cmd.symbol}</span>
                        <span className="flex-1">{cmd.action}</span>
                        {cmd.value && <span className="text-gray-500">{cmd.value}</span>}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="render" className="space-y-4">
              <div className="flex items-center gap-2 mb-4">
                <Button onClick={renderLSystem} disabled={isRendering}>
                  {isRendering ? (
                    <>Rendering... {Math.round(renderProgress)}%</>
                  ) : (
                    <><Play className="w-4 h-4 mr-1" />Render</>
                  )}
                </Button>
                <Button variant="outline" onClick={exportImage}>
                  <Download className="w-4 h-4 mr-1" />
                  Export
                </Button>
                <Button variant="outline" onClick={() => setRenderProgress(0)}>
                  <RotateCcw className="w-4 h-4 mr-1" />
                  Clear
                </Button>
                {lsystem.render_3d && (
                  <Button
                    variant="outline"
                    onClick={() => setRenderSettings(prev => ({ ...prev, show3D: !prev.show3D }))}
                  >
                    <Layers3 className="w-4 h-4 mr-1" />
                    {renderSettings.show3D ? '2D View' : '3D View'}
                  </Button>
                )}
              </div>

              <div className="border rounded-lg overflow-hidden">
                <canvas
                  ref={canvasRef}
                  width={CANVAS_WIDTH}
                  height={CANVAS_HEIGHT}
                  className={`w-full bg-white ${lsystem.render_3d && renderSettings.show3D ? 'hidden' : ''}`}
                />
                {lsystem.render_3d && (
                  <canvas
                    ref={canvas3DRef}
                    width={CANVAS_WIDTH}
                    height={CANVAS_HEIGHT}
                    className={`w-full bg-white ${!renderSettings.show3D ? 'hidden' : ''}`}
                  />
                )}
              </div>

              {generatedString && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Generated String</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="font-mono text-xs bg-gray-50 p-2 rounded max-h-20 overflow-y-auto break-all">
                      {generatedString.length > 1000 ? 
                        `${generatedString.substring(0, 1000)}... (${generatedString.length} total characters)` : 
                        generatedString
                      }
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="settings" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Settings className="w-4 h-4" />
                      Render Settings
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <label className="text-sm font-medium">Line Width</label>
                      <Slider
                        value={[renderSettings.lineWidth]}
                        onValueChange={([value]) => setRenderSettings(prev => ({ ...prev, lineWidth: value }))}
                        min={0.5}
                        max={5}
                        step={0.5}
                      />
                    </div>

                    <div>
                      <label className="text-sm font-medium">Color Scheme</label>
                      <Select
                        value={renderSettings.colorScheme}
                        onValueChange={(value) => setRenderSettings(prev => ({ ...prev, colorScheme: value }))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="gradient">Gradient</SelectItem>
                          <SelectItem value="depth">Depth</SelectItem>
                          <SelectItem value="single">Single Color</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={renderSettings.animate}
                        onCheckedChange={(checked) => setRenderSettings(prev => ({ ...prev, animate: checked }))}
                      />
                      <label className="text-sm font-medium">Animate Rendering</label>
                    </div>

                    {renderSettings.animate && (
                      <div>
                        <label className="text-sm font-medium">Animation Speed (ms)</label>
                        <Slider
                          value={[renderSettings.animationSpeed]}
                          onValueChange={([value]) => setRenderSettings(prev => ({ ...prev, animationSpeed: value }))}
                          min={10}
                          max={200}
                          step={10}
                        />
                      </div>
                    )}
                  </CardContent>
                </Card>

                {lsystem.render_3d && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm flex items-center gap-2">
                        <Layers3 className="w-4 h-4" />
                        3D Settings
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div>
                        <label className="text-sm font-medium">Perspective</label>
                        <Slider
                          value={[renderSettings.perspective]}
                          onValueChange={([value]) => setRenderSettings(prev => ({ ...prev, perspective: value }))}
                          min={100}
                          max={1000}
                          step={50}
                        />
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </TabsContent>

            <TabsContent value="presets" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(presetLSystems).map(([name, preset]) => (
                  <Card key={name} className="cursor-pointer hover:shadow-md transition-shadow">
                    <CardHeader>
                      <CardTitle className="text-sm capitalize">{name}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2 text-xs">
                        <div><strong>Axiom:</strong> <code>{preset.axiom}</code></div>
                        <div><strong>Rules:</strong> {Object.keys(preset.productions).length}</div>
                        <div><strong>Angle:</strong> {preset.angle}°</div>
                        <div><strong>3D:</strong> {preset.render_3d ? 'Yes' : 'No'}</div>
                      </div>
                      <Button
                        size="sm"
                        className="w-full mt-3"
                        onClick={() => loadPreset(name)}
                        disabled={readOnly}
                      >
                        Load Preset
                      </Button>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};