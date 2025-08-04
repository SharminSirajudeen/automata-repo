import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Textarea } from './ui/textarea';
import { BookOpen, CheckCircle, XCircle, HelpCircle, Lightbulb } from 'lucide-react';
import { ExtendedAutomaton } from '../types/automata';

interface ProofStep {
  id: string;
  description: string;
  type: 'assumption' | 'definition' | 'lemma' | 'conclusion';
  isValid: boolean;
  explanation?: string;
}

interface ProofAssistantProps {
  automaton: ExtendedAutomaton;
}

export const ProofAssistant: React.FC<ProofAssistantProps> = ({ 
  automaton
}) => {
  const [proofSteps, setProofSteps] = useState<ProofStep[]>([]);
  const [currentStep, setCurrentStep] = useState('');
  const [proofType, setProofType] = useState<'equivalence' | 'pumping' | 'closure'>('equivalence');
  const [isGenerating, setIsGenerating] = useState(false);

  const addProofStep = () => {
    if (!currentStep.trim()) return;

    const newStep: ProofStep = {
      id: Date.now().toString(),
      description: currentStep.trim(),
      type: 'assumption',
      isValid: true,
      explanation: 'Step added by user'
    };

    setProofSteps(prev => [...prev, newStep]);
    setCurrentStep('');
  };

  const validateProofStep = async (stepId: string) => {
    setIsGenerating(true);
    
    try {
      const response = await fetch('/api/validate-proof-step', {
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
          ? { ...step, isValid: result.is_valid, explanation: result.explanation }
          : step
      ));
    } catch (error) {
      console.error('Failed to validate proof step:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const generateProofSuggestion = async () => {
    setIsGenerating(true);
    
    try {
      const response = await fetch('/api/generate-proof', {
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
          isValid: true,
          explanation: step.explanation
        }));
        
        setProofSteps(prev => [...prev, ...newSteps]);
      }
    } catch (error) {
      console.error('Failed to generate proof suggestion:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const getProofTypeDescription = () => {
    switch (proofType) {
      case 'equivalence':
        return 'Prove that two automata accept the same language';
      case 'pumping':
        return 'Apply the pumping lemma to prove a language is not regular/context-free';
      case 'closure':
        return 'Prove closure properties of language classes';
      default:
        return '';
    }
  };

  const getStepTypeColor = (type: string) => {
    switch (type) {
      case 'assumption': return 'bg-blue-100 text-blue-800';
      case 'definition': return 'bg-green-100 text-green-800';
      case 'lemma': return 'bg-purple-100 text-purple-800';
      case 'conclusion': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-indigo-600" />
          Proof Assistant
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="space-y-3">
          <div>
            <label className="text-sm font-medium mb-2 block">Proof Type</label>
            <select
              value={proofType}
              onChange={(e) => setProofType(e.target.value as any)}
              className="w-full px-3 py-2 border rounded text-sm"
            >
              <option value="equivalence">Language Equivalence</option>
              <option value="pumping">Pumping Lemma</option>
              <option value="closure">Closure Properties</option>
            </select>
            <p className="text-xs text-gray-600 mt-1">{getProofTypeDescription()}</p>
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
            </div>
          </div>
        </div>

        <div className="space-y-3 max-h-96 overflow-y-auto">
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
              <div key={step.id} className="p-3 border rounded-lg space-y-2">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-500">
                      {index + 1}.
                    </span>
                    <Badge className={`text-xs ${getStepTypeColor(step.type)}`}>
                      {step.type}
                    </Badge>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {step.isValid ? (
                      <CheckCircle className="w-4 h-4 text-green-600" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-600" />
                    )}
                    <Button
                      onClick={() => validateProofStep(step.id)}
                      size="sm"
                      variant="outline"
                      className="h-6 px-2"
                      disabled={isGenerating}
                    >
                      <HelpCircle className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
                
                <p className="text-sm text-gray-700 pl-6">{step.description}</p>
                
                {step.explanation && (
                  <div className="pl-6">
                    <p className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                      {step.explanation}
                    </p>
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {proofSteps.length > 0 && (
          <div className="pt-3 border-t">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">
                Proof Status: {proofSteps.every(s => s.isValid) ? 'Valid' : 'Needs Review'}
              </span>
              {proofSteps.every(s => s.isValid) && (
                <CheckCircle className="w-5 h-5 text-green-600" />
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
