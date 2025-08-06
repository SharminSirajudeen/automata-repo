import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Progress } from './ui/progress';
import { Separator } from './ui/separator';
import { ScrollArea } from './ui/scroll-area';
import { Alert, AlertDescription } from './ui/alert';
import { 
  GitCompare, 
  CheckCircle, 
  XCircle, 
  ArrowRight, 
  Minimize2, 
  Target, 
  AlertTriangle,
  Eye,
  Split,
  RefreshCw,
  Lightbulb,
  PlayCircle
} from 'lucide-react';
import { ExtendedAutomaton, Automaton, ValidationResult } from '../types/automata';
import { API_BASE_URL } from '../config/api';

interface EquivalenceResult {
  are_equivalent: boolean;
  explanation: string;
  counter_example?: string;
  witness_string?: string;
  proof_steps?: string[];
}

interface MinimizationStep {
  step_number: number;
  description: string;
  states_before: string[];
  states_after: string[];
  merged_states?: { [key: string]: string[] };
  partition?: string[][];
}

interface MinimizationResult {
  minimized_automaton: Automaton;
  steps: MinimizationStep[];
  removed_states: string[];
  state_mapping: { [key: string]: string };
}

interface ContainmentResult {
  is_contained: boolean;
  explanation: string;
  counter_example?: string;
  inclusion_proof?: string[];
}

interface FormalVerificationProps {
  primaryAutomaton?: ExtendedAutomaton;
  secondaryAutomaton?: ExtendedAutomaton;
  onResultsUpdate?: (results: any) => void;
}

export const FormalVerification: React.FC<FormalVerificationProps> = ({
  primaryAutomaton,
  secondaryAutomaton,
  onResultsUpdate
}) => {
  const [activeTab, setActiveTab] = useState<'equivalence' | 'minimization' | 'containment'>('equivalence');
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  
  // Equivalence checking state
  const [equivalenceResult, setEquivalenceResult] = useState<EquivalenceResult | null>(null);
  const [selectedAutomata, setSelectedAutomata] = useState<{ first: string; second: string }>({
    first: 'primary',
    second: 'secondary'
  });
  
  // Minimization state
  const [minimizationResult, setMinimizationResult] = useState<MinimizationResult | null>(null);
  const [currentMinimizationStep, setCurrentMinimizationStep] = useState(0);
  const [showMinimizationAnimation, setShowMinimizationAnimation] = useState(false);
  
  // Containment checking state
  const [containmentResult, setContainmentResult] = useState<ContainmentResult | null>(null);
  const [containmentDirection, setContainmentDirection] = useState<'subset' | 'superset'>('subset');

  const checkEquivalence = async () => {
    if (!primaryAutomaton || !secondaryAutomaton) {
      return;
    }

    setIsProcessing(true);
    setProcessingProgress(0);
    
    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProcessingProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const response = await fetch(`${API_BASE_URL}/api/check-equivalence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          automaton1: primaryAutomaton,
          automaton2: secondaryAutomaton,
          method: 'product_construction'
        })
      });

      clearInterval(progressInterval);
      setProcessingProgress(100);

      const result = await response.json();
      setEquivalenceResult(result);
      
      if (onResultsUpdate) {
        onResultsUpdate({ equivalence: result });
      }
    } catch (error) {
      console.error('Failed to check equivalence:', error);
      setEquivalenceResult({
        are_equivalent: false,
        explanation: 'Failed to perform equivalence check',
        counter_example: undefined
      });
    } finally {
      setIsProcessing(false);
      setTimeout(() => setProcessingProgress(0), 1000);
    }
  };

  const minimizeAutomaton = async (automaton: ExtendedAutomaton) => {
    setIsProcessing(true);
    setProcessingProgress(0);
    
    try {
      const progressInterval = setInterval(() => {
        setProcessingProgress(prev => Math.min(prev + 15, 90));
      }, 300);

      const response = await fetch(`${API_BASE_URL}/api/minimize-automaton`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          automaton,
          algorithm: 'hopcroft'
        })
      });

      clearInterval(progressInterval);
      setProcessingProgress(100);

      const result = await response.json();
      setMinimizationResult(result);
      setCurrentMinimizationStep(0);
      
      if (onResultsUpdate) {
        onResultsUpdate({ minimization: result });
      }
    } catch (error) {
      console.error('Failed to minimize automaton:', error);
    } finally {
      setIsProcessing(false);
      setTimeout(() => setProcessingProgress(0), 1000);
    }
  };

  const checkContainment = async () => {
    if (!primaryAutomaton || !secondaryAutomaton) {
      return;
    }

    setIsProcessing(true);
    setProcessingProgress(0);
    
    try {
      const progressInterval = setInterval(() => {
        setProcessingProgress(prev => Math.min(prev + 12, 90));
      }, 250);

      const response = await fetch(`${API_BASE_URL}/api/check-containment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          automaton1: containmentDirection === 'subset' ? primaryAutomaton : secondaryAutomaton,
          automaton2: containmentDirection === 'subset' ? secondaryAutomaton : primaryAutomaton,
          direction: containmentDirection
        })
      });

      clearInterval(progressInterval);
      setProcessingProgress(100);

      const result = await response.json();
      setContainmentResult(result);
      
      if (onResultsUpdate) {
        onResultsUpdate({ containment: result });
      }
    } catch (error) {
      console.error('Failed to check containment:', error);
      setContainmentResult({
        is_contained: false,
        explanation: 'Failed to perform containment check',
        counter_example: undefined
      });
    } finally {
      setIsProcessing(false);
      setTimeout(() => setProcessingProgress(0), 1000);
    }
  };

  const playMinimizationAnimation = () => {
    if (!minimizationResult) return;
    
    setShowMinimizationAnimation(true);
    setCurrentMinimizationStep(0);
    
    const interval = setInterval(() => {
      setCurrentMinimizationStep(prev => {
        if (prev >= minimizationResult.steps.length - 1) {
          clearInterval(interval);
          setShowMinimizationAnimation(false);
          return prev;
        }
        return prev + 1;
      });
    }, 2000);
  };

  const renderEquivalenceTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="border-2 border-blue-200 bg-blue-50/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              Automaton A
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="text-xs text-gray-600">
                States: {primaryAutomaton?.type === 'dfa' || primaryAutomaton?.type === 'nfa' ? 
                  (primaryAutomaton as Automaton).states.length : 'N/A'}
              </div>
              <div className="text-xs text-gray-600">
                Type: {primaryAutomaton?.type || 'None'}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-2 border-green-200 bg-green-50/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              Automaton B
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="text-xs text-gray-600">
                States: {secondaryAutomaton?.type === 'dfa' || secondaryAutomaton?.type === 'nfa' ? 
                  (secondaryAutomaton as Automaton).states.length : 'N/A'}
              </div>
              <div className="text-xs text-gray-600">
                Type: {secondaryAutomaton?.type || 'None'}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="flex gap-3">
        <Button 
          onClick={checkEquivalence}
          disabled={!primaryAutomaton || !secondaryAutomaton || isProcessing}
          className="flex-1"
        >
          <GitCompare className="w-4 h-4 mr-2" />
          Check Equivalence
        </Button>
        
        <Button variant="outline" size="sm">
          <Eye className="w-4 h-4 mr-2" />
          Preview Product
        </Button>
      </div>

      {isProcessing && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Checking equivalence...</span>
            <span>{processingProgress}%</span>
          </div>
          <Progress value={processingProgress} className="h-2" />
        </div>
      )}

      {equivalenceResult && (
        <Card className={`border-2 ${equivalenceResult.are_equivalent ? 'border-green-200 bg-green-50/30' : 'border-red-200 bg-red-50/30'}`}>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              {equivalenceResult.are_equivalent ? (
                <CheckCircle className="w-4 h-4 text-green-600" />
              ) : (
                <XCircle className="w-4 h-4 text-red-600" />
              )}
              Equivalence Result
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm">{equivalenceResult.explanation}</p>
            
            {equivalenceResult.counter_example && (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <strong>Counter-example:</strong> "{equivalenceResult.counter_example}"
                  {equivalenceResult.witness_string && (
                    <span className="block mt-1">
                      Witness string: "{equivalenceResult.witness_string}"
                    </span>
                  )}
                </AlertDescription>
              </Alert>
            )}

            {equivalenceResult.proof_steps && equivalenceResult.proof_steps.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Proof Steps:</h4>
                <ScrollArea className="h-32">
                  <div className="space-y-1">
                    {equivalenceResult.proof_steps.map((step, index) => (
                      <div key={index} className="text-xs p-2 bg-white rounded border-l-2 border-blue-200">
                        {index + 1}. {step}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );

  const renderMinimizationTab = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">Automaton Minimization</h3>
        <Select 
          value={primaryAutomaton ? 'primary' : 'secondary'} 
          onValueChange={() => {}}
        >
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Select automaton" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="primary">Primary Automaton</SelectItem>
            <SelectItem value="secondary">Secondary Automaton</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="flex gap-3">
        <Button 
          onClick={() => minimizeAutomaton(primaryAutomaton!)}
          disabled={!primaryAutomaton || isProcessing}
          className="flex-1"
        >
          <Minimize2 className="w-4 h-4 mr-2" />
          Minimize Automaton
        </Button>
        
        {minimizationResult && (
          <Button 
            variant="outline" 
            onClick={playMinimizationAnimation}
            disabled={showMinimizationAnimation}
          >
            <PlayCircle className="w-4 h-4 mr-2" />
            Animate Steps
          </Button>
        )}
      </div>

      {isProcessing && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Minimizing automaton...</span>
            <span>{processingProgress}%</span>
          </div>
          <Progress value={processingProgress} className="h-2" />
        </div>
      )}

      {minimizationResult && (
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center justify-between">
                <span>Minimization Summary</span>
                <Badge variant="outline">
                  {minimizationResult.removed_states.length} states removed
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium">Original states:</span>
                  <span className="ml-2">{Object.keys(minimizationResult.state_mapping).length}</span>
                </div>
                <div>
                  <span className="font-medium">Minimized states:</span>
                  <span className="ml-2">{minimizationResult.minimized_automaton.states.length}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center justify-between">
                <span>Minimization Steps</span>
                {showMinimizationAnimation && (
                  <Badge className="bg-blue-100 text-blue-800">
                    Step {currentMinimizationStep + 1}/{minimizationResult.steps.length}
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-64">
                <div className="space-y-3">
                  {minimizationResult.steps.map((step, index) => (
                    <div 
                      key={index}
                      className={`p-3 rounded border-l-4 transition-all ${
                        showMinimizationAnimation && index === currentMinimizationStep
                          ? 'border-blue-500 bg-blue-50 shadow-sm'
                          : index <= currentMinimizationStep || !showMinimizationAnimation
                          ? 'border-green-500 bg-green-50'
                          : 'border-gray-200 bg-gray-50 opacity-50'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Step {step.step_number}</span>
                        {step.merged_states && Object.keys(step.merged_states).length > 0 && (
                          <Badge variant="outline" className="text-xs">
                            {Object.keys(step.merged_states).length} merges
                          </Badge>
                        )}
                      </div>
                      
                      <p className="text-sm text-gray-700 mb-2">{step.description}</p>
                      
                      {step.merged_states && Object.keys(step.merged_states).length > 0 && (
                        <div className="text-xs space-y-1">
                          {Object.entries(step.merged_states).map(([target, sources]) => (
                            <div key={target} className="flex items-center gap-2">
                              <span className="font-mono bg-gray-100 px-1 rounded">
                                {sources.join(', ')}
                              </span>
                              <ArrowRight className="w-3 h-3" />
                              <span className="font-mono bg-blue-100 px-1 rounded">
                                {target}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );

  const renderContainmentTab = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium">Language Containment</h3>
          <Select 
            value={containmentDirection} 
            onValueChange={(value: any) => setContainmentDirection(value)}
          >
            <SelectTrigger className="w-48">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="subset">L(A) ⊆ L(B)</SelectItem>
              <SelectItem value="superset">L(A) ⊇ L(B)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className={`border-2 ${containmentDirection === 'subset' ? 'border-blue-200 bg-blue-50/30' : 'border-green-200 bg-green-50/30'}`}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                {containmentDirection === 'subset' ? 'Subset Language' : 'Superset Language'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xs text-gray-600">
                Automaton: {containmentDirection === 'subset' ? 'A' : 'B'}
              </div>
            </CardContent>
          </Card>

          <Card className={`border-2 ${containmentDirection === 'subset' ? 'border-green-200 bg-green-50/30' : 'border-blue-200 bg-blue-50/30'}`}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                {containmentDirection === 'subset' ? 'Superset Language' : 'Subset Language'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xs text-gray-600">
                Automaton: {containmentDirection === 'subset' ? 'B' : 'A'}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <Button 
        onClick={checkContainment}
        disabled={!primaryAutomaton || !secondaryAutomaton || isProcessing}
        className="w-full"
      >
        <Target className="w-4 h-4 mr-2" />
        Check Containment
      </Button>

      {isProcessing && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Checking containment...</span>
            <span>{processingProgress}%</span>
          </div>
          <Progress value={processingProgress} className="h-2" />
        </div>
      )}

      {containmentResult && (
        <Card className={`border-2 ${containmentResult.is_contained ? 'border-green-200 bg-green-50/30' : 'border-red-200 bg-red-50/30'}`}>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              {containmentResult.is_contained ? (
                <CheckCircle className="w-4 h-4 text-green-600" />
              ) : (
                <XCircle className="w-4 h-4 text-red-600" />
              )}
              Containment Result
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm">{containmentResult.explanation}</p>
            
            {containmentResult.counter_example && (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <strong>Counter-example:</strong> "{containmentResult.counter_example}"
                </AlertDescription>
              </Alert>
            )}

            {containmentResult.inclusion_proof && containmentResult.inclusion_proof.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Inclusion Proof:</h4>
                <ScrollArea className="h-32">
                  <div className="space-y-1">
                    {containmentResult.inclusion_proof.map((step, index) => (
                      <div key={index} className="text-xs p-2 bg-white rounded border-l-2 border-green-200">
                        {index + 1}. {step}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Split className="w-5 h-5 text-purple-600" />
          Formal Verification
        </CardTitle>
      </CardHeader>
      
      <CardContent>
        <Tabs value={activeTab} onValueChange={(value: any) => setActiveTab(value)}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="equivalence">Equivalence</TabsTrigger>
            <TabsTrigger value="minimization">Minimization</TabsTrigger>
            <TabsTrigger value="containment">Containment</TabsTrigger>
          </TabsList>

          <TabsContent value="equivalence" className="mt-6">
            {renderEquivalenceTab()}
          </TabsContent>

          <TabsContent value="minimization" className="mt-6">
            {renderMinimizationTab()}
          </TabsContent>

          <TabsContent value="containment" className="mt-6">
            {renderContainmentTab()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};