import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Search, AlertTriangle, CheckCircle, Minimize2, GitBranch } from 'lucide-react';
import { ExtendedAutomaton, AutomataType, State, Transition } from '../types/automata';

interface AutomataInspectorProps {
  automaton: ExtendedAutomaton;
  automatonType: AutomataType;
  onHighlightStates?: (stateIds: string[]) => void;
}

interface InspectionResult {
  unreachable_states: string[];
  dead_states: string[];
  equivalent_states: string[][];
  minimization_suggestions: string[];
  ambiguous_productions?: string[];
  left_recursion?: string[];
}

export const AutomataInspector: React.FC<AutomataInspectorProps> = ({
  automaton,
  automatonType,
  onHighlightStates
}) => {
  const [inspectionResult, setInspectionResult] = useState<InspectionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [minimizedAutomaton, setMinimizedAutomaton] = useState<ExtendedAutomaton | null>(null);

  useEffect(() => {
    if (automaton) {
      analyzeAutomaton();
    }
  }, [automaton]);

  const analyzeAutomaton = async () => {
    setIsAnalyzing(true);
    try {
      const result = await performStaticAnalysis(automaton, automatonType);
      setInspectionResult(result);
      
      if (automatonType === 'dfa' && result.equivalent_states.length > 0) {
        const minimized = await minimizeAutomaton();
        setMinimizedAutomaton(minimized);
      }
    } catch (error) {
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const performStaticAnalysis = async (automaton: ExtendedAutomaton, type: AutomataType): Promise<InspectionResult> => {
    if (type === 'dfa' || type === 'nfa') {
      return analyzeFSM(automaton as any);
    } else if (type === 'cfg') {
      return analyzeCFG(automaton as any);
    } else if (type === 'pda') {
      return analyzePDA();
    } else if (type === 'tm') {
      return analyzeTM();
    }
    
    return {
      unreachable_states: [],
      dead_states: [],
      equivalent_states: [],
      minimization_suggestions: []
    };
  };

  const analyzeFSM = (automaton: { states: State[], transitions: Transition[] }): InspectionResult => {
    const unreachable = findUnreachableStates(automaton);
    const dead = findDeadStates(automaton);
    const equivalent = findEquivalentStates();
    
    const suggestions = [];
    if (unreachable.length > 0) {
      suggestions.push(`Remove ${unreachable.length} unreachable state(s): ${unreachable.join(', ')}`);
    }
    if (dead.length > 0) {
      suggestions.push(`Consider removing ${dead.length} dead state(s): ${dead.join(', ')}`);
    }
    if (equivalent.length > 0) {
      suggestions.push(`Merge ${equivalent.length} group(s) of equivalent states`);
    }
    
    return {
      unreachable_states: unreachable,
      dead_states: dead,
      equivalent_states: equivalent,
      minimization_suggestions: suggestions
    };
  };

  const analyzeCFG = (automaton: any): InspectionResult => {
    const suggestions = [];
    const ambiguous: string[] = [];
    const leftRecursive: string[] = [];
    
    if (automaton.productions) {
      for (const production of automaton.productions) {
        if (production.left_side === production.right_side.split(' ')[0]) {
          leftRecursive.push(production.id);
        }
      }
    }
    
    if (leftRecursive.length > 0) {
      suggestions.push(`Remove left recursion from ${leftRecursive.length} production(s)`);
    }
    
    return {
      unreachable_states: [],
      dead_states: [],
      equivalent_states: [],
      minimization_suggestions: suggestions,
      ambiguous_productions: ambiguous,
      left_recursion: leftRecursive
    };
  };

  const analyzePDA = (): InspectionResult => {
    return {
      unreachable_states: [],
      dead_states: [],
      equivalent_states: [],
      minimization_suggestions: ['PDA analysis not yet implemented']
    };
  };

  const analyzeTM = (): InspectionResult => {
    return {
      unreachable_states: [],
      dead_states: [],
      equivalent_states: [],
      minimization_suggestions: ['TM analysis not yet implemented']
    };
  };

  const findUnreachableStates = (automaton: { states: State[], transitions: Transition[] }): string[] => {
    const startStates = automaton.states.filter(s => s.is_start).map(s => s.id);
    if (startStates.length === 0) return [];
    
    const reachable = new Set<string>();
    const queue = [...startStates];
    
    while (queue.length > 0) {
      const current = queue.shift()!;
      if (reachable.has(current)) continue;
      
      reachable.add(current);
      
      for (const transition of automaton.transitions) {
        if (transition.from_state === current && !reachable.has(transition.to_state)) {
          queue.push(transition.to_state);
        }
      }
    }
    
    return automaton.states
      .filter(s => !reachable.has(s.id))
      .map(s => s.id);
  };

  const findDeadStates = (automaton: { states: State[], transitions: Transition[] }): string[] => {
    const acceptStates = new Set(automaton.states.filter(s => s.is_accept).map(s => s.id));
    const canReachAccept = new Set<string>();
    
    const queue = Array.from(acceptStates);
    while (queue.length > 0) {
      const current = queue.shift()!;
      if (canReachAccept.has(current)) continue;
      
      canReachAccept.add(current);
      
      for (const transition of automaton.transitions) {
        if (transition.to_state === current && !canReachAccept.has(transition.from_state)) {
          queue.push(transition.from_state);
        }
      }
    }
    
    return automaton.states
      .filter(s => !canReachAccept.has(s.id) && !s.is_accept)
      .map(s => s.id);
  };

  const findEquivalentStates = (): string[][] => {
    return [];
  };

  const minimizeAutomaton = async (): Promise<ExtendedAutomaton | null> => {
    try {
      const response = await fetch('/api/minimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ automaton })
      });

      if (response.ok) {
        const data = await response.json();
        return data.minimized_automaton;
      }
    } catch (error) {
      console.error('Minimization error:', error);
    }
    return null;
  };

  const handleHighlightIssue = (stateIds: string[]) => {
    onHighlightStates?.(stateIds);
  };

  const getTypeDisplayName = (type: AutomataType) => {
    const typeNames: { [key in AutomataType]: string } = {
      'dfa': 'DFA',
      'nfa': 'NFA',
      'enfa': 'ε-NFA',
      'pda': 'PDA',
      'cfg': 'CFG',
      'tm': 'Turing Machine',
      'regex': 'Regular Expression',
      'pumping': 'Pumping Lemma'
    };
    return typeNames[type];
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Search className="w-5 h-5 text-orange-600" />
          Automata Inspector
          <Badge variant="outline" className="ml-auto">
            {getTypeDisplayName(automatonType)}
          </Badge>
        </CardTitle>
      </CardHeader>
      
      <CardContent>
        <Tabs defaultValue="analysis" className="h-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
            <TabsTrigger value="optimization">Optimization</TabsTrigger>
            <TabsTrigger value="properties">Properties</TabsTrigger>
          </TabsList>
          
          <TabsContent value="analysis" className="space-y-4 mt-4">
            {isAnalyzing ? (
              <div className="flex items-center justify-center py-8">
                <div className="w-6 h-6 border-2 border-orange-600 border-t-transparent rounded-full animate-spin mr-2" />
                <span>Analyzing automaton...</span>
              </div>
            ) : inspectionResult ? (
              <div className="space-y-4">
                {inspectionResult.unreachable_states.length > 0 && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-4 h-4 text-red-600" />
                      <span className="font-medium text-red-800">Unreachable States</span>
                    </div>
                    <p className="text-sm text-red-700 mb-2">
                      These states cannot be reached from the start state:
                    </p>
                    <div className="flex gap-1 mb-2">
                      {inspectionResult.unreachable_states.map(stateId => (
                        <Badge key={stateId} variant="destructive" className="text-xs">
                          {stateId}
                        </Badge>
                      ))}
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleHighlightIssue(inspectionResult.unreachable_states)}
                    >
                      Highlight on Canvas
                    </Button>
                  </div>
                )}

                {inspectionResult.dead_states.length > 0 && (
                  <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-4 h-4 text-yellow-600" />
                      <span className="font-medium text-yellow-800">Dead States</span>
                    </div>
                    <p className="text-sm text-yellow-700 mb-2">
                      These states cannot reach any accept state:
                    </p>
                    <div className="flex gap-1 mb-2">
                      {inspectionResult.dead_states.map(stateId => (
                        <Badge key={stateId} variant="secondary" className="text-xs">
                          {stateId}
                        </Badge>
                      ))}
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleHighlightIssue(inspectionResult.dead_states)}
                    >
                      Highlight on Canvas
                    </Button>
                  </div>
                )}

                {inspectionResult.equivalent_states.length > 0 && (
                  <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <GitBranch className="w-4 h-4 text-blue-600" />
                      <span className="font-medium text-blue-800">Equivalent States</span>
                    </div>
                    <p className="text-sm text-blue-700 mb-2">
                      These state groups can be merged:
                    </p>
                    {inspectionResult.equivalent_states.map((group, index) => (
                      <div key={index} className="flex gap-1 mb-2">
                        {group.map(stateId => (
                          <Badge key={stateId} variant="outline" className="text-xs">
                            {stateId}
                          </Badge>
                        ))}
                      </div>
                    ))}
                  </div>
                )}

                {inspectionResult.unreachable_states.length === 0 && 
                 inspectionResult.dead_states.length === 0 && 
                 inspectionResult.equivalent_states.length === 0 && (
                  <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-green-600" />
                      <span className="font-medium text-green-800">No Issues Found</span>
                    </div>
                    <p className="text-sm text-green-700 mt-1">
                      Your automaton appears to be well-formed with no obvious structural issues.
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>Click analyze to inspect your automaton</p>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="optimization" className="space-y-4 mt-4">
            {inspectionResult?.minimization_suggestions && (
              <div className="space-y-3">
                <h3 className="font-medium text-gray-900">Optimization Suggestions</h3>
                {inspectionResult.minimization_suggestions.map((suggestion, index) => (
                  <div key={index} className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-700">{suggestion}</p>
                  </div>
                ))}
                
                {automatonType === 'dfa' && (
                  <Button
                    onClick={minimizeAutomaton}
                    className="w-full"
                    disabled={isAnalyzing}
                  >
                    <Minimize2 className="w-4 h-4 mr-2" />
                    Minimize DFA
                  </Button>
                )}
              </div>
            )}

            {minimizedAutomaton && (
              <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                <h4 className="font-medium text-green-800 mb-2">Minimized Automaton</h4>
                <p className="text-sm text-green-700">
                  Original states: {(automaton as any).states?.length || 0} → 
                  Minimized states: {(minimizedAutomaton as any).states?.length || 0}
                </p>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="properties" className="space-y-4 mt-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-gray-50 rounded-lg">
                <h4 className="font-medium text-gray-900 mb-1">States</h4>
                <p className="text-2xl font-bold text-blue-600">
                  {(automaton as any).states?.length || 0}
                </p>
              </div>
              
              <div className="p-3 bg-gray-50 rounded-lg">
                <h4 className="font-medium text-gray-900 mb-1">Transitions</h4>
                <p className="text-2xl font-bold text-green-600">
                  {(automaton as any).transitions?.length || 0}
                </p>
              </div>
              
              <div className="p-3 bg-gray-50 rounded-lg">
                <h4 className="font-medium text-gray-900 mb-1">Alphabet Size</h4>
                <p className="text-2xl font-bold text-purple-600">
                  {(automaton as any).alphabet?.length || 0}
                </p>
              </div>
              
              <div className="p-3 bg-gray-50 rounded-lg">
                <h4 className="font-medium text-gray-900 mb-1">Accept States</h4>
                <p className="text-2xl font-bold text-orange-600">
                  {(automaton as any).states?.filter((s: State) => s.is_accept).length || 0}
                </p>
              </div>
            </div>

            {automatonType === 'cfg' && inspectionResult?.left_recursion && (
              <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                <h4 className="font-medium text-yellow-800 mb-2">Left Recursion Detected</h4>
                <p className="text-sm text-yellow-700">
                  Productions with left recursion: {inspectionResult.left_recursion.length}
                </p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};
