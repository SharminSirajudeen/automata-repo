import React, { useState, useRef, useCallback } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Alert, AlertDescription } from './ui/alert';
import { Plus, Trash2, Edit3, CheckCircle, AlertCircle, Eye } from 'lucide-react';
import { UnrestrictedGrammar, UnrestrictedProduction, ParseNode } from '../types/advanced-automata';

interface AdvancedGrammarEditorProps {
  grammar: UnrestrictedGrammar;
  onGrammarChange: (grammar: UnrestrictedGrammar) => void;
  readOnly?: boolean;
}

export const AdvancedGrammarEditor: React.FC<AdvancedGrammarEditorProps> = ({
  grammar,
  onGrammarChange,
  readOnly = false
}) => {
  const [newProduction, setNewProduction] = useState<Partial<UnrestrictedProduction & {
    left_side: string | string[];
    right_side: string | string[];
  }>>({});
  const [, setEditingProduction] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [parseString, setParseString] = useState('');
  const [parseTree, setParseTree] = useState<ParseNode | null>(null);
  const [showBracketedView, setShowBracketedView] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const validateGrammar = useCallback(() => {
    const errors: string[] = [];
    
    // Check if start symbol exists in non-terminals
    if (!grammar.non_terminals.includes(grammar.start_symbol)) {
      errors.push('Start symbol must be a non-terminal');
    }
    
    // Check productions
    grammar.productions.forEach((prod, index) => {
      // Left side validation for unrestricted grammars
      if (prod.left_side.length === 0) {
        errors.push(`Production ${index + 1}: Left side cannot be empty`);
      }
      
      // Check if left side contains at least one non-terminal for context-sensitive
      if (prod.context_sensitive) {
        const hasNonTerminal = prod.left_side.some(symbol => 
          grammar.non_terminals.includes(symbol)
        );
        if (!hasNonTerminal) {
          errors.push(`Production ${index + 1}: Context-sensitive productions must have at least one non-terminal on left side`);
        }
        
        // Context-sensitive: |α| ≤ |β| (except for S → ε)
        if (prod.right_side.length < prod.left_side.length && 
            !(prod.left_side.length === 1 && 
              prod.left_side[0] === grammar.start_symbol && 
              prod.right_side.length === 0)) {
          errors.push(`Production ${index + 1}: Context-sensitive rule violated (right side must be at least as long as left side)`);
        }
      }
      
      // Check symbols exist in alphabet
      [...prod.left_side, ...prod.right_side].forEach(symbol => {
        if (symbol !== '' && 
            !grammar.terminals.includes(symbol) && 
            !grammar.non_terminals.includes(symbol)) {
          errors.push(`Production ${index + 1}: Unknown symbol '${symbol}'`);
        }
      });
    });
    
    setValidationErrors(errors);
    return errors.length === 0;
  }, [grammar]);

  const addProduction = () => {
    if (!newProduction.left_side || !newProduction.right_side) return;
    
    const leftSide = typeof newProduction.left_side === 'string' 
      ? newProduction.left_side.split('').filter(s => s.trim())
      : newProduction.left_side;
    
    const rightSide = typeof newProduction.right_side === 'string'
      ? (newProduction.right_side === 'ε' ? [] : newProduction.right_side.split('').filter(s => s.trim()))
      : newProduction.right_side;
    
    const production: UnrestrictedProduction = {
      id: `prod_${Date.now()}`,
      left_side: leftSide,
      right_side: rightSide,
      context_sensitive: newProduction.context_sensitive || false
    };
    
    onGrammarChange({
      ...grammar,
      productions: [...grammar.productions, production]
    });
    
    setNewProduction({});
  };

  // const updateProduction = (id: string, updates: Partial<UnrestrictedProduction>) => {
  //   onGrammarChange({
  //     ...grammar,
  //     productions: grammar.productions.map(prod =>
  //       prod.id === id ? { ...prod, ...updates } : prod
  //     )
  //   });
  // };

  const deleteProduction = (id: string) => {
    onGrammarChange({
      ...grammar,
      productions: grammar.productions.filter(prod => prod.id !== id)
    });
  };

  const formatProductionString = (production: UnrestrictedProduction) => {
    const leftSide = production.left_side.join('');
    const rightSide = production.right_side.length === 0 ? 'ε' : production.right_side.join('');
    return `${leftSide} → ${rightSide}`;
  };

  const parseStringWithGrammar = async () => {
    if (!parseString.trim()) return;
    
    try {
      // This would typically call a parser service
      const response = await fetch('/api/parse-unrestricted', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          grammar,
          input_string: parseString
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        setParseTree(result.parse_tree);
      }
    } catch (error) {
      console.error('Parse error:', error);
    }
  };

  const drawParseTree = useCallback(() => {
    if (!parseTree || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const drawNode = (node: ParseNode, x: number, y: number, level: number) => {
      const nodeRadius = 25;
      const levelHeight = 80;
      
      // Draw node
      ctx.fillStyle = node.is_terminal ? '#dbeafe' : '#fef3c7';
      ctx.strokeStyle = node.is_bracketed ? '#ef4444' : '#64748b';
      ctx.lineWidth = node.is_bracketed ? 3 : 2;
      
      ctx.beginPath();
      ctx.arc(x, y, nodeRadius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      
      // Draw symbol
      ctx.fillStyle = '#1f2937';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(node.symbol, x, y + 5);
      
      // Draw children
      const childrenWidth = node.children.length * 100;
      const startX = x - childrenWidth / 2 + 50;
      
      node.children.forEach((child, index) => {
        const childX = startX + index * 100;
        const childY = y + levelHeight;
        
        // Draw edge
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y + nodeRadius);
        ctx.lineTo(childX, childY - nodeRadius);
        ctx.stroke();
        
        // Recursively draw child
        drawNode(child, childX, childY, level + 1);
      });
    };
    
    drawNode(parseTree, canvas.width / 2, 50, 0);
  }, [parseTree]);

  React.useEffect(() => {
    drawParseTree();
  }, [drawParseTree]);

  React.useEffect(() => {
    validateGrammar();
  }, [validateGrammar]);

  const renderBracketedString = () => {
    if (!parseTree) return null;
    
    const generateBracketedString = (node: ParseNode): string => {
      if (node.is_terminal) {
        return node.symbol;
      }
      
      const childStrings = node.children.map(generateBracketedString);
      return `[${node.symbol} ${childStrings.join(' ')}]`;
    };
    
    return (
      <div className="font-mono text-sm p-3 bg-gray-50 rounded border">
        {generateBracketedString(parseTree)}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Advanced Grammar Editor
            <div className="flex items-center gap-2">
              <Badge variant={validationErrors.length === 0 ? "default" : "destructive"}>
                {validationErrors.length === 0 ? (
                  <><CheckCircle className="w-3 h-3 mr-1" />Valid</>
                ) : (
                  <><AlertCircle className="w-3 h-3 mr-1" />{validationErrors.length} Errors</>
                )}
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="productions">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="productions">Productions</TabsTrigger>
              <TabsTrigger value="symbols">Symbols</TabsTrigger>
              <TabsTrigger value="parser">Parser</TabsTrigger>
              <TabsTrigger value="validation">Validation</TabsTrigger>
            </TabsList>
            
            <TabsContent value="productions" className="space-y-4">
              {!readOnly && (
                <Card className="border-dashed">
                  <CardHeader>
                    <CardTitle className="text-sm">Add New Production</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="text-sm font-medium">Left Side</label>
                        <Input
                          value={newProduction.left_side || ''}
                          onChange={(e) => setNewProduction(prev => ({ ...prev, left_side: e.target.value }))}
                          placeholder="αAβ"
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium">Right Side</label>
                        <Input
                          value={newProduction.right_side || ''}
                          onChange={(e) => setNewProduction(prev => ({ ...prev, right_side: e.target.value }))}
                          placeholder="γδε or ε"
                        />
                      </div>
                      <div className="flex items-end">
                        <Button onClick={addProduction} className="w-full">
                          <Plus className="w-4 h-4 mr-1" />
                          Add
                        </Button>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="context-sensitive"
                        checked={newProduction.context_sensitive || false}
                        onChange={(e) => setNewProduction(prev => ({ ...prev, context_sensitive: e.target.checked }))}
                      />
                      <label htmlFor="context-sensitive" className="text-sm">
                        Context-sensitive (length-preserving)
                      </label>
                    </div>
                  </CardContent>
                </Card>
              )}
              
              <div className="space-y-2">
                {grammar.productions.map((production) => (
                  <Card key={production.id} className="relative">
                    <CardContent className="pt-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="font-mono text-lg">
                            {formatProductionString(production)}
                          </span>
                          {production.context_sensitive && (
                            <Badge variant="secondary" className="text-xs">CS</Badge>
                          )}
                        </div>
                        {!readOnly && (
                          <div className="flex gap-1">
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => setEditingProduction(production.id)}
                            >
                              <Edit3 className="w-3 h-3" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => deleteProduction(production.id)}
                            >
                              <Trash2 className="w-3 h-3" />
                            </Button>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>
            
            <TabsContent value="symbols" className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Non-terminals</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-1">
                      {grammar.non_terminals.map(symbol => (
                        <Badge key={symbol} variant="outline">{symbol}</Badge>
                      ))}
                    </div>
                    {!readOnly && (
                      <Input
                        className="mt-2"
                        placeholder="Add non-terminal..."
                        onKeyPress={(e) => {
                          if (e.key === 'Enter' && e.currentTarget.value.trim()) {
                            const newSymbol = e.currentTarget.value.trim();
                            if (!grammar.non_terminals.includes(newSymbol)) {
                              onGrammarChange({
                                ...grammar,
                                non_terminals: [...grammar.non_terminals, newSymbol]
                              });
                            }
                            e.currentTarget.value = '';
                          }
                        }}
                      />
                    )}
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Terminals</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-1">
                      {grammar.terminals.map(symbol => (
                        <Badge key={symbol} variant="secondary">{symbol}</Badge>
                      ))}
                    </div>
                    {!readOnly && (
                      <Input
                        className="mt-2"
                        placeholder="Add terminal..."
                        onKeyPress={(e) => {
                          if (e.key === 'Enter' && e.currentTarget.value.trim()) {
                            const newSymbol = e.currentTarget.value.trim();
                            if (!grammar.terminals.includes(newSymbol)) {
                              onGrammarChange({
                                ...grammar,
                                terminals: [...grammar.terminals, newSymbol]
                              });
                            }
                            e.currentTarget.value = '';
                          }
                        }}
                      />
                    )}
                  </CardContent>
                </Card>
              </div>
              
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Start Symbol</CardTitle>
                </CardHeader>
                <CardContent>
                  <Select
                    value={grammar.start_symbol}
                    onValueChange={(value) => onGrammarChange({ ...grammar, start_symbol: value })}
                    disabled={readOnly}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {grammar.non_terminals.map(symbol => (
                        <SelectItem key={symbol} value={symbol}>{symbol}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="parser" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Parse String</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex gap-2">
                    <Input
                      value={parseString}
                      onChange={(e) => setParseString(e.target.value)}
                      placeholder="Enter string to parse..."
                      className="flex-1"
                    />
                    <Button onClick={parseStringWithGrammar}>Parse</Button>
                  </div>
                  
                  {parseTree && (
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setShowBracketedView(!showBracketedView)}
                        >
                          <Eye className="w-4 h-4 mr-1" />
                          {showBracketedView ? 'Tree View' : 'Bracketed View'}
                        </Button>
                      </div>
                      
                      {showBracketedView ? (
                        renderBracketedString()
                      ) : (
                        <div className="border rounded-lg overflow-hidden">
                          <canvas
                            ref={canvasRef}
                            width={800}
                            height={400}
                            className="w-full bg-gray-50"
                          />
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="validation" className="space-y-4">
              {validationErrors.length > 0 ? (
                <div className="space-y-2">
                  {validationErrors.map((error, index) => (
                    <Alert key={index} variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  ))}
                </div>
              ) : (
                <Alert>
                  <CheckCircle className="h-4 w-4" />
                  <AlertDescription>
                    Grammar is valid! All productions follow the rules for unrestricted/context-sensitive grammars.
                  </AlertDescription>
                </Alert>
              )}
              
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Grammar Statistics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Type:</span>
                      <span className="ml-2">
                        {grammar.productions.every(p => p.context_sensitive) ? 'Context-Sensitive' : 'Unrestricted'}
                      </span>
                    </div>
                    <div>
                      <span className="font-medium">Productions:</span>
                      <span className="ml-2">{grammar.productions.length}</span>
                    </div>
                    <div>
                      <span className="font-medium">Non-terminals:</span>
                      <span className="ml-2">{grammar.non_terminals.length}</span>
                    </div>
                    <div>
                      <span className="font-medium">Terminals:</span>
                      <span className="ml-2">{grammar.terminals.length}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};