import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Slider } from './ui/slider';
import { Progress } from './ui/progress';
import { Separator } from './ui/separator';
import { ScrollArea } from './ui/scroll-area';
import { Alert, AlertDescription } from './ui/alert';
import { 
  Zap, 
  Play, 
  Pause, 
  RotateCcw, 
  ChevronLeft, 
  ChevronRight,
  Calculator,
  Target,
  AlertTriangle,
  CheckCircle,
  Lightbulb,
  ArrowDown,
  ArrowUp,
  Repeat
} from 'lucide-react';
import { PumpingLemmaAutomaton } from '../types/automata';
import { API_BASE_URL } from '../config/api';

interface StringDecomposition {
  x: string;
  y: string;
  z: string;
  is_valid: boolean;
  explanation: string;
}

interface PumpingStep {
  i: number;
  pumped_string: string;
  length: number;
  is_accepted: boolean;
  explanation: string;
}

interface PumpingExample {
  language_description: string;
  example_string: string;
  pumping_length: number;
  decompositions: StringDecomposition[];
  pumping_steps: PumpingStep[];
  proof_conclusion: string;
}

interface NonRegularityProof {
  steps: string[];
  contradiction_found: boolean;
  witness_decomposition?: StringDecomposition;
  counter_example?: PumpingStep;
}

interface PumpingLemmaVisualizerProps {
  language?: PumpingLemmaAutomaton;
  onProofComplete?: (proof: NonRegularityProof) => void;
}

export const PumpingLemmaVisualizer: React.FC<PumpingLemmaVisualizerProps> = ({
  language,
  onProofComplete
}) => {
  const [activeTab, setActiveTab] = useState<'explore' | 'prove' | 'examples'>('explore');
  const [testString, setTestString] = useState('');
  const [pumpingLength, setPumpingLength] = useState([3]);
  const [currentDecomposition, setCurrentDecomposition] = useState<StringDecomposition | null>(null);
  const [pumpingSteps, setPumpingSteps] = useState<PumpingStep[]>([]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [animationSpeed, setAnimationSpeed] = useState([1000]);
  
  // Proof building state
  const [proofSteps, setProofSteps] = useState<string[]>([]);
  const [selectedLanguage, setSelectedLanguage] = useState('custom');
  const [languageDescription, setLanguageDescription] = useState('');
  const [proofResult, setProofResult] = useState<NonRegularityProof | null>(null);
  
  // Examples state
  const [predefinedExamples, setPredefinedExamples] = useState<PumpingExample[]>([]);
  const [selectedExample, setSelectedExample] = useState<PumpingExample | null>(null);

  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadPredefinedExamples();
  }, []);

  const loadPredefinedExamples = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/pumping-examples`);
      const examples = await response.json();
      setPredefinedExamples(examples);
    } catch (error) {
      console.error('Failed to load examples:', error);
    }
  };

  const decomposeString = useCallback(async (str: string, p: number) => {
    if (str.length < p) {
      setCurrentDecomposition({
        x: '',
        y: '',
        z: str,
        is_valid: false,
        explanation: 'String is shorter than pumping length'
      });
      return;
    }

    setIsLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/decompose-string`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          string: str,
          pumping_length: p,
          language_type: language?.language_type || 'regular'
        })
      });

      const result = await response.json();
      setCurrentDecomposition(result);
    } catch (error) {
      console.error('Failed to decompose string:', error);
      setCurrentDecomposition({
        x: '',
        y: str.substring(0, Math.min(1, str.length)),
        z: str.substring(Math.min(1, str.length)),
        is_valid: false,
        explanation: 'Failed to decompose string'
      });
    } finally {
      setIsLoading(false);
    }
  }, [language]);

  const generatePumpingSteps = useCallback(async (decomposition: StringDecomposition) => {
    setIsLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-pumping-steps`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          decomposition,
          max_pumps: 5,
          language_description: languageDescription || language?.language_description
        })
      });

      const result = await response.json();
      setPumpingSteps(result.steps || []);
      setCurrentStep(0);
    } catch (error) {
      console.error('Failed to generate pumping steps:', error);
      setPumpingSteps([]);
    } finally {
      setIsLoading(false);
    }
  }, [languageDescription, language]);

  const startPumpingAnimation = () => {
    if (pumpingSteps.length === 0) return;
    
    setIsAnimating(true);
    setCurrentStep(0);
    
    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= pumpingSteps.length - 1) {
          clearInterval(interval);
          setIsAnimating(false);
          return prev;
        }
        return prev + 1;
      });
    }, animationSpeed[0]);
  };

  const buildNonRegularityProof = async () => {
    if (!currentDecomposition || !languageDescription) return;

    setIsLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/build-non-regularity-proof`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          language_description: languageDescription,
          test_string: testString,
          decomposition: currentDecomposition,
          pumping_steps: pumpingSteps
        })
      });

      const result = await response.json();
      setProofResult(result);
      
      if (result.contradiction_found && onProofComplete) {
        onProofComplete(result);
      }
    } catch (error) {
      console.error('Failed to build proof:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const renderStringVisualization = () => {
    if (!currentDecomposition) return null;

    const { x, y, z } = currentDecomposition;
    const currentPumpedStep = pumpingSteps[currentStep];
    
    return (
      <div className="space-y-4">
        <div className="text-center">
          <h4 className="text-sm font-medium mb-3">String Decomposition: s = xyz</h4>
          <div className="flex items-center justify-center gap-1 text-lg font-mono bg-gray-50 p-4 rounded border">
            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded border-2 border-blue-200">
              {x || 'ε'}
            </span>
            <span className="px-3 py-1 bg-red-100 text-red-800 rounded border-2 border-red-200">
              {y || 'ε'}
            </span>
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded border-2 border-green-200">
              {z || 'ε'}
            </span>
          </div>
          
          <div className="flex items-center justify-center gap-4 mt-2 text-xs text-gray-600">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-blue-100 border border-blue-200 rounded" />
              <span>x (prefix)</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-red-100 border border-red-200 rounded" />
              <span>y (pump)</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-green-100 border border-green-200 rounded" />
              <span>z (suffix)</span>
            </div>
          </div>
        </div>

        {currentPumpedStep && (
          <div className="text-center">
            <h4 className="text-sm font-medium mb-3">
              Pumping Step i = {currentPumpedStep.i}
            </h4>
            <div className="text-lg font-mono bg-gray-50 p-4 rounded border">
              <span className="text-blue-800">{x}</span>
              <span className="text-red-800 relative">
                {Array(currentPumpedStep.i + 1).fill(y).join('')}
                {isAnimating && (
                  <div className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                )}
              </span>
              <span className="text-green-800">{z}</span>
            </div>
            
            <div className="flex items-center justify-center gap-4 mt-3">
              <Badge className={`${currentPumpedStep.is_accepted ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                Length: {currentPumpedStep.length}
              </Badge>
              <Badge className={`${currentPumpedStep.is_accepted ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                {currentPumpedStep.is_accepted ? 'Accepted' : 'Rejected'}
              </Badge>
            </div>
            
            {currentPumpedStep.explanation && (
              <p className="text-xs text-gray-600 mt-2 bg-yellow-50 p-2 rounded">
                {currentPumpedStep.explanation}
              </p>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderExploreTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-3">
          <label className="text-sm font-medium">Test String</label>
          <Input
            value={testString}
            onChange={(e) => setTestString(e.target.value)}
            placeholder="Enter string to test (e.g., aabbcc)"
            className="font-mono"
          />
        </div>
        
        <div className="space-y-3">
          <label className="text-sm font-medium">
            Pumping Length: {pumpingLength[0]}
          </label>
          <Slider
            value={pumpingLength}
            onValueChange={setPumpingLength}
            max={Math.max(10, testString.length)}
            min={1}
            step={1}
            className="w-full"
          />
        </div>
      </div>

      <div className="flex gap-3">
        <Button 
          onClick={() => decomposeString(testString, pumpingLength[0])}
          disabled={!testString || isLoading}
          className="flex-1"
        >
          <Calculator className="w-4 h-4 mr-2" />
          Decompose String
        </Button>
        
        {currentDecomposition && (
          <Button 
            onClick={() => generatePumpingSteps(currentDecomposition)}
            disabled={isLoading}
            variant="outline"
          >
            <Target className="w-4 h-4 mr-2" />
            Generate Steps
          </Button>
        )}
      </div>

      {isLoading && (
        <div className="flex items-center justify-center p-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-sm">Processing...</span>
        </div>
      )}

      {currentDecomposition && (
        <Card className={`border-2 ${currentDecomposition.is_valid ? 'border-green-200 bg-green-50/30' : 'border-red-200 bg-red-50/30'}`}>
          <CardContent className="pt-4">
            {renderStringVisualization()}
          </CardContent>
        </Card>
      )}

      {pumpingSteps.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center justify-between">
              <span>Pumping Animation</span>
              <div className="flex items-center gap-2">
                <span className="text-xs">Speed:</span>
                <Slider
                  value={animationSpeed}
                  onValueChange={setAnimationSpeed}
                  max={2000}
                  min={500}
                  step={250}
                  className="w-20"
                />
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-center gap-3">
              <Button 
                onClick={startPumpingAnimation}
                disabled={isAnimating}
                size="sm"
              >
                <Play className="w-4 h-4 mr-2" />
                Animate
              </Button>
              
              <Button 
                onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                disabled={currentStep === 0 || isAnimating}
                size="sm"
                variant="outline"
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
              
              <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">
                {currentStep + 1} / {pumpingSteps.length}
              </span>
              
              <Button 
                onClick={() => setCurrentStep(Math.min(pumpingSteps.length - 1, currentStep + 1))}
                disabled={currentStep === pumpingSteps.length - 1 || isAnimating}
                size="sm"
                variant="outline"
              >
                <ChevronRight className="w-4 h-4" />
              </Button>
              
              <Button 
                onClick={() => setCurrentStep(0)}
                disabled={isAnimating}
                size="sm"
                variant="outline"
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>

            <ScrollArea className="h-32">
              <div className="space-y-2">
                {pumpingSteps.map((step, index) => (
                  <div 
                    key={index}
                    className={`p-2 rounded text-sm border transition-all ${
                      index === currentStep 
                        ? 'border-blue-500 bg-blue-50 shadow-sm' 
                        : 'border-gray-200 hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono">
                        i = {step.i}: {step.pumped_string}
                      </span>
                      <Badge className={`text-xs ${step.is_accepted ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                        {step.is_accepted ? 'Accept' : 'Reject'}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}
    </div>
  );

  const renderProveTab = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Language to Prove Non-Regular</label>
          <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="custom">Custom Language</SelectItem>
              <SelectItem value="anbn">L = {`{a^n b^n | n ≥ 0}`}</SelectItem>
              <SelectItem value="palindromes">L = {`{palindromes over {a,b}}`}</SelectItem>
              <SelectItem value="equal_ab">L = {`{w | #a(w) = #b(w)}`}</SelectItem>
              <SelectItem value="anbncn">L = {`{a^n b^n c^n | n ≥ 0}`}</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {selectedLanguage === 'custom' && (
          <div className="space-y-2">
            <label className="text-sm font-medium">Language Description</label>
            <Input
              value={languageDescription}
              onChange={(e) => setLanguageDescription(e.target.value)}
              placeholder="Describe the language (e.g., strings with equal a's and b's)"
            />
          </div>
        )}
      </div>

      <Button 
        onClick={buildNonRegularityProof}
        disabled={!currentDecomposition || !testString || isLoading || (!languageDescription && selectedLanguage === 'custom')}
        className="w-full"
      >
        <Lightbulb className="w-4 h-4 mr-2" />
        Build Non-Regularity Proof
      </Button>

      {proofResult && (
        <Card className={`border-2 ${proofResult.contradiction_found ? 'border-green-200 bg-green-50/30' : 'border-yellow-200 bg-yellow-50/30'}`}>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              {proofResult.contradiction_found ? (
                <CheckCircle className="w-4 h-4 text-green-600" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-yellow-600" />
              )}
              Proof Result
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <ScrollArea className="h-64">
              <div className="space-y-3">
                {proofResult.steps.map((step, index) => (
                  <div key={index} className="p-3 bg-white rounded border-l-4 border-blue-200">
                    <div className="flex items-start gap-2">
                      <span className="text-sm font-medium text-gray-500 mt-0.5">
                        {index + 1}.
                      </span>
                      <p className="text-sm">{step}</p>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>

            {proofResult.contradiction_found && proofResult.counter_example && (
              <Alert>
                <CheckCircle className="h-4 w-4" />
                <AlertDescription>
                  <strong>Contradiction found!</strong> The string "{proofResult.counter_example.pumped_string}" 
                  with i = {proofResult.counter_example.i} violates the pumping lemma conditions.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );

  const renderExamplesTab = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Predefined Examples</h3>
        
        <div className="grid gap-4">
          {predefinedExamples.map((example, index) => (
            <Card 
              key={index}
              className={`cursor-pointer transition-all hover:shadow-md ${selectedExample === example ? 'border-blue-500 bg-blue-50/30' : ''}`}
              onClick={() => setSelectedExample(example)}
            >
              <CardContent className="p-4">
                <div className="space-y-2">
                  <h4 className="font-medium">{example.language_description}</h4>
                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <span>Example: {example.example_string}</span>
                    <Badge variant="outline">p = {example.pumping_length}</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {selectedExample && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Interactive Example</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <p className="text-sm"><strong>Language:</strong> {selectedExample.language_description}</p>
              <p className="text-sm"><strong>Example String:</strong> {selectedExample.example_string}</p>
              <p className="text-sm"><strong>Pumping Length:</strong> {selectedExample.pumping_length}</p>
            </div>

            <Separator />

            <div className="space-y-3">
              <h4 className="text-sm font-medium">Available Decompositions:</h4>
              <div className="grid gap-2">
                {selectedExample.decompositions.map((dec, index) => (
                  <div 
                    key={index}
                    className={`p-3 rounded border cursor-pointer transition-all ${dec.is_valid ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}`}
                    onClick={() => {
                      setCurrentDecomposition(dec);
                      setTestString(selectedExample.example_string);
                      setPumpingLength([selectedExample.pumping_length]);
                      setPumpingSteps(selectedExample.pumping_steps);
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-sm">
                        x="{dec.x}", y="{dec.y}", z="{dec.z}"
                      </span>
                      <Badge className={`text-xs ${dec.is_valid ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                        {dec.is_valid ? 'Valid' : 'Invalid'}
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-600 mt-1">{dec.explanation}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="text-sm p-3 bg-blue-50 rounded border border-blue-200">
              <strong>Conclusion:</strong> {selectedExample.proof_conclusion}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-600" />
          Pumping Lemma Visualizer
        </CardTitle>
      </CardHeader>
      
      <CardContent>
        <Tabs value={activeTab} onValueChange={(value: any) => setActiveTab(value)}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="explore">Explore</TabsTrigger>
            <TabsTrigger value="prove">Prove</TabsTrigger>
            <TabsTrigger value="examples">Examples</TabsTrigger>
          </TabsList>

          <TabsContent value="explore" className="mt-6">
            {renderExploreTab()}
          </TabsContent>

          <TabsContent value="prove" className="mt-6">
            {renderProveTab()}
          </TabsContent>

          <TabsContent value="examples" className="mt-6">
            {renderExamplesTab()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};