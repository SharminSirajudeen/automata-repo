import React, { useState, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Textarea } from './ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Progress } from './ui/progress';
import { Separator } from './ui/separator';
import { ScrollArea } from './ui/scroll-area';
import { BookOpen, CheckCircle, XCircle, HelpCircle, Lightbulb, GripVertical, TreePine, Move, Target, ArrowRight, ChevronDown, ChevronRight } from 'lucide-react';
import { ExtendedAutomaton } from '../types/automata';
import { API_BASE_URL } from '../config/api';

interface ProofStep {
  id: string;
  description: string;
  type: 'assumption' | 'definition' | 'lemma' | 'conclusion' | 'contradiction' | 'induction_base' | 'induction_step' | 'construction';
  isValid: boolean;
  explanation?: string;
  dependencies?: string[];
  children?: string[];
  level: number;
  isExpanded?: boolean;
  validationStatus: 'pending' | 'validating' | 'valid' | 'invalid';
  hints?: string[];
}

interface ProofTree {
  root: string;
  nodes: { [key: string]: ProofStep };
  edges: { [parent: string]: string[] };
}

interface DragItem {
  id: string;
  type: 'step';
  index: number;
}

interface ProofAssistantProps {
  automaton: ExtendedAutomaton;
  onProofComplete?: (proof: ProofStep[]) => void;
}

export const ProofAssistant: React.FC<ProofAssistantProps> = ({ 
  automaton,
  onProofComplete
}) => {
  const [proofSteps, setProofSteps] = useState<ProofStep[]>([]);
  const [currentStep, setCurrentStep] = useState('');
  const [stepType, setStepType] = useState<ProofStep['type']>('assumption');
  const [proofType, setProofType] = useState<'equivalence' | 'pumping' | 'closure' | 'contradiction' | 'induction' | 'construction'>('equivalence');
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeTab, setActiveTab] = useState<'linear' | 'tree'>('linear');
  const [draggedItem, setDraggedItem] = useState<DragItem | null>(null);
  const [proofTree, setProofTree] = useState<ProofTree>({ root: '', nodes: {}, edges: {} });
  const [validationProgress, setValidationProgress] = useState(0);
  const dragRef = useRef<HTMLDivElement>(null);

  const addProofStep = useCallback(() => {
    if (!currentStep.trim()) return;

    const newStep: ProofStep = {
      id: Date.now().toString(),
      description: currentStep.trim(),
      type: stepType,
      isValid: false,
      explanation: undefined,
      dependencies: [],
      children: [],
      level: proofSteps.length,
      isExpanded: true,
      validationStatus: 'pending',
      hints: []
    };

    setProofSteps(prev => [...prev, newStep]);
    updateProofTree(newStep);
    setCurrentStep('');
  }, [currentStep, stepType, proofSteps.length]);

  const updateProofTree = (step: ProofStep) => {
    setProofTree(prev => ({
      ...prev,
      nodes: { ...prev.nodes, [step.id]: step },
      edges: { ...prev.edges, [step.id]: step.children || [] }
    }));
  };

  const validateProofStep = async (stepId: string) => {
    setProofSteps(prev => prev.map(step => 
      step.id === stepId 
        ? { ...step, validationStatus: 'validating' }
        : step
    ));
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/validate-proof-step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          automaton,
          proof_type: proofType,
          step_id: stepId,
          steps: proofSteps
        })
      });

      const result = await response.json();
      
      setProofSteps(prev => prev.map(step => 
        step.id === stepId 
          ? { 
              ...step, 
              isValid: result.is_valid, 
              explanation: result.explanation,
              validationStatus: result.is_valid ? 'valid' : 'invalid',
              hints: result.hints || []
            }
          : step
      ));
    } catch (error) {
      console.error('Failed to validate proof step:', error);
      setProofSteps(prev => prev.map(step => 
        step.id === stepId 
          ? { ...step, validationStatus: 'invalid', explanation: 'Validation failed' }
          : step
      ));
    }
  };

  const validateAllSteps = async () => {
    setIsGenerating(true);
    setValidationProgress(0);
    
    for (let i = 0; i < proofSteps.length; i++) {
      await validateProofStep(proofSteps[i].id);
      setValidationProgress(((i + 1) / proofSteps.length) * 100);
    }
    
    setIsGenerating(false);
  };

  const generateProofSuggestion = async () => {
    setIsGenerating(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-proof`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          automaton,
          proof_type: proofType,
          current_steps: proofSteps
        })
      });

      const result = await response.json();
      
      if (result.suggested_steps) {
        const newSteps = result.suggested_steps.map((step: any, index: number) => ({
          id: `suggested-${Date.now()}-${index}`,
          description: step.description,
          type: step.type,
          isValid: false,
          explanation: step.explanation,
          dependencies: step.dependencies || [],
          children: [],
          level: proofSteps.length + index,
          isExpanded: true,
          validationStatus: 'pending' as const,
          hints: step.hints || []
        }));
        
        setProofSteps(prev => [...prev, ...newSteps]);
        newSteps.forEach(updateProofTree);
      }
    } catch (error) {
      console.error('Failed to generate proof suggestion:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const moveStep = useCallback((dragIndex: number, dropIndex: number) => {
    const draggedStep = proofSteps[dragIndex];
    const updatedSteps = [...proofSteps];
    updatedSteps.splice(dragIndex, 1);
    updatedSteps.splice(dropIndex, 0, draggedStep);
    
    // Update levels
    const stepsWithUpdatedLevels = updatedSteps.map((step, index) => ({
      ...step,
      level: index
    }));
    
    setProofSteps(stepsWithUpdatedLevels);
  }, [proofSteps]);

  const handleDragStart = (e: React.DragEvent, item: DragItem) => {
    setDraggedItem(item);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e: React.DragEvent, dropIndex: number) => {
    e.preventDefault();
    if (draggedItem && draggedItem.type === 'step') {
      moveStep(draggedItem.index, dropIndex);
    }
    setDraggedItem(null);
  };

  const getProofTypeDescription = () => {
    switch (proofType) {
      case 'equivalence':
        return 'Prove that two automata accept the same language';
      case 'pumping':
        return 'Apply the pumping lemma to prove a language is not regular/context-free';
      case 'closure':
        return 'Prove closure properties of language classes';
      case 'contradiction':
        return 'Prove by assuming the opposite and deriving a contradiction';
      case 'induction':
        return 'Prove using mathematical induction (base case + inductive step)';
      case 'construction':
        return 'Prove by constructing an example or counter-example';
      default:
        return '';
    }
  };

  const getStepTypeColor = (type: string) => {
    switch (type) {
      case 'assumption': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'definition': return 'bg-green-100 text-green-800 border-green-200';
      case 'lemma': return 'bg-purple-100 text-purple-800 border-purple-200';
      case 'conclusion': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'contradiction': return 'bg-red-100 text-red-800 border-red-200';
      case 'induction_base': return 'bg-cyan-100 text-cyan-800 border-cyan-200';
      case 'induction_step': return 'bg-teal-100 text-teal-800 border-teal-200';
      case 'construction': return 'bg-indigo-100 text-indigo-800 border-indigo-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getValidationStatusIcon = (status: ProofStep['validationStatus']) => {
    switch (status) {
      case 'valid': return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'invalid': return <XCircle className="w-4 h-4 text-red-600" />;
      case 'validating': return <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />;
      case 'pending': return <HelpCircle className="w-4 h-4 text-gray-400" />;
      default: return null;
    }
  };

  const renderProofTree = () => {
    const renderNode = (stepId: string, depth = 0) => {
      const step = proofSteps.find(s => s.id === stepId);
      if (!step) return null;

      return (
        <div key={step.id} className="ml-4" style={{ marginLeft: `${depth * 20}px` }}>
          <div className="flex items-center gap-2 p-2 border rounded bg-white hover:bg-gray-50 transition-colors">
            <button
              onClick={() => {
                setProofSteps(prev => prev.map(s => 
                  s.id === step.id ? { ...s, isExpanded: !s.isExpanded } : s
                ));
              }}
              className="p-1 hover:bg-gray-100 rounded"
            >
              {step.isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            </button>
            
            <Badge className={`text-xs ${getStepTypeColor(step.type)}`}>
              {step.type}
            </Badge>
            
            <span className="text-sm flex-1">{step.description}</span>
            
            {getValidationStatusIcon(step.validationStatus)}
          </div>
          
          {step.isExpanded && step.children?.map(childId => renderNode(childId, depth + 1))}
        </div>
      );
    };

    return (
      <div className="space-y-2">
        {proofSteps.filter(step => step.level === 0).map(step => renderNode(step.id))}
      </div>
    );
  };

  const isProofComplete = proofSteps.length > 0 && proofSteps.every(step => step.validationStatus === 'valid');

  React.useEffect(() => {
    if (isProofComplete && onProofComplete) {
      onProofComplete(proofSteps);
    }
  }, [isProofComplete, proofSteps, onProofComplete]);

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-indigo-600" />
            Proof Assistant
          </div>
          {isProofComplete && (
            <Badge className="bg-green-100 text-green-800 border-green-200">
              <CheckCircle className="w-3 h-3 mr-1" />
              Complete
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Proof Type</label>
            <Select value={proofType} onValueChange={(value: any) => setProofType(value)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="equivalence">Language Equivalence</SelectItem>
                <SelectItem value="pumping">Pumping Lemma</SelectItem>
                <SelectItem value="closure">Closure Properties</SelectItem>
                <SelectItem value="contradiction">Proof by Contradiction</SelectItem>
                <SelectItem value="induction">Mathematical Induction</SelectItem>
                <SelectItem value="construction">Proof by Construction</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-gray-600">{getProofTypeDescription()}</p>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Step Type</label>
            <Select value={stepType} onValueChange={(value: any) => setStepType(value)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="assumption">Assumption</SelectItem>
                <SelectItem value="definition">Definition</SelectItem>
                <SelectItem value="lemma">Lemma</SelectItem>
                <SelectItem value="conclusion">Conclusion</SelectItem>
                <SelectItem value="contradiction">Contradiction</SelectItem>
                <SelectItem value="induction_base">Induction Base</SelectItem>
                <SelectItem value="induction_step">Induction Step</SelectItem>
                <SelectItem value="construction">Construction</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Add Proof Step</label>
          <Textarea
            placeholder="Enter your proof step..."
            value={currentStep}
            onChange={(e) => setCurrentStep(e.target.value)}
            className="min-h-[80px]"
          />
          <div className="flex gap-2">
            <Button onClick={addProofStep} size="sm" className="flex-1">
              <Target className="w-4 h-4 mr-2" />
              Add Step
            </Button>
            <Button 
              onClick={generateProofSuggestion}
              variant="outline" 
              size="sm"
              disabled={isGenerating}
            >
              <Lightbulb className="w-4 h-4 mr-2" />
              Suggest
            </Button>
            <Button 
              onClick={validateAllSteps}
              variant="outline" 
              size="sm"
              disabled={isGenerating || proofSteps.length === 0}
            >
              <CheckCircle className="w-4 h-4 mr-2" />
              Validate All
            </Button>
          </div>
        </div>

        {isGenerating && validationProgress > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span>Validating steps...</span>
              <span>{Math.round(validationProgress)}%</span>
            </div>
            <Progress value={validationProgress} className="h-2" />
          </div>
        )}

        <Tabs value={activeTab} onValueChange={(value: any) => setActiveTab(value)}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="linear" className="flex items-center gap-2">
              <Move className="w-4 h-4" />
              Linear View
            </TabsTrigger>
            <TabsTrigger value="tree" className="flex items-center gap-2">
              <TreePine className="w-4 h-4" />
              Tree View
            </TabsTrigger>
          </TabsList>

          <TabsContent value="linear" className="mt-4">
            <ScrollArea className="h-96">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium">Proof Steps</h3>
                  <Badge variant="outline" className="text-xs">
                    {proofSteps.length} steps
                  </Badge>
                </div>

                {proofSteps.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <BookOpen className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No proof steps yet</p>
                    <p className="text-xs">Add your first step to begin the proof</p>
                  </div>
                ) : (
                  proofSteps.map((step, index) => (
                    <div 
                      key={step.id} 
                      className="group p-3 border rounded-lg space-y-2 hover:shadow-sm transition-shadow"
                      draggable
                      onDragStart={(e) => handleDragStart(e, { id: step.id, type: 'step', index })}
                      onDragOver={handleDragOver}
                      onDrop={(e) => handleDrop(e, index)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-2">
                          <GripVertical className="w-4 h-4 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity cursor-move" />
                          <span className="text-sm font-medium text-gray-500">
                            {index + 1}.
                          </span>
                          <Badge className={`text-xs border ${getStepTypeColor(step.type)}`}>
                            {step.type.replace('_', ' ')}
                          </Badge>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          {getValidationStatusIcon(step.validationStatus)}
                          <Button
                            onClick={() => validateProofStep(step.id)}
                            size="sm"
                            variant="outline"
                            className="h-6 px-2 opacity-0 group-hover:opacity-100 transition-opacity"
                            disabled={step.validationStatus === 'validating'}
                          >
                            <HelpCircle className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>
                      
                      <p className="text-sm text-gray-700 pl-6">{step.description}</p>
                      
                      {step.explanation && (
                        <div className="pl-6">
                          <p className="text-xs text-gray-600 bg-gray-50 p-2 rounded border-l-2 border-blue-200">
                            {step.explanation}
                          </p>
                        </div>
                      )}
                      
                      {step.hints && step.hints.length > 0 && (
                        <div className="pl-6">
                          <details className="text-xs">
                            <summary className="cursor-pointer text-blue-600 hover:text-blue-800">
                              Show hints ({step.hints.length})
                            </summary>
                            <div className="mt-1 space-y-1">
                              {step.hints.map((hint, hintIndex) => (
                                <p key={hintIndex} className="text-gray-600 bg-blue-50 p-2 rounded">
                                  💡 {hint}
                                </p>
                              ))}
                            </div>
                          </details>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="tree" className="mt-4">
            <ScrollArea className="h-96">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium">Proof Tree</h3>
                  <Badge variant="outline" className="text-xs">
                    {proofSteps.length} nodes
                  </Badge>
                </div>

                {proofSteps.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <TreePine className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No proof tree yet</p>
                    <p className="text-xs">Add steps to see the proof structure</p>
                  </div>
                ) : (
                  renderProofTree()
                )}
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>

        {proofSteps.length > 0 && (
          <>
            <Separator />
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">
                  Proof Status: {isProofComplete ? 'Complete & Valid' : 'In Progress'}
                </span>
                <div className="flex items-center gap-2">
                  <Badge variant={isProofComplete ? 'default' : 'outline'} className="text-xs">
                    {proofSteps.filter(s => s.validationStatus === 'valid').length}/{proofSteps.length} validated
                  </Badge>
                  {isProofComplete && (
                    <CheckCircle className="w-5 h-5 text-green-600" />
                  )}
                </div>
              </div>
              
              {!isProofComplete && proofSteps.length > 0 && (
                <div className="text-xs text-amber-600 bg-amber-50 p-2 rounded border border-amber-200">
                  <div className="flex items-center gap-1">
                    <ArrowRight className="w-3 h-3" />
                    <span>Next: {proofSteps.find(s => s.validationStatus === 'pending') ? 'Validate pending steps' : 'Add concluding step'}</span>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};
