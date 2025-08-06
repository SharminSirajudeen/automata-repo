import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Textarea } from './ui/textarea';
import { Separator } from './ui/separator';
import { ScrollArea } from './ui/scroll-area';
import { Alert, AlertDescription } from './ui/alert';
import { Progress } from './ui/progress';
import { 
  Network, 
  ArrowRight, 
  ArrowDown,
  Timer,
  Target,
  CheckCircle,
  XCircle,
  Lightbulb,
  Zap,
  GitBranch,
  Calculator,
  Eye,
  Play,
  Settings,
  TrendingUp,
  AlertTriangle
} from 'lucide-react';
import { API_BASE_URL } from '../config/api';

interface ComplexityClass {
  name: string;
  description: string;
  examples: string[];
  relationships: string[];
  color: string;
  position: { x: number; y: number };
}

interface ReductionStep {
  from_problem: string;
  to_problem: string;
  reduction_type: 'polynomial' | 'logarithmic' | 'many_one' | 'turing';
  description: string;
  time_complexity: string;
  space_complexity: string;
  construction: string[];
}

interface Algorithm {
  name: string;
  problem: string;
  time_complexity: string;
  space_complexity: string;
  description: string;
  pseudocode: string[];
  analysis_steps: string[];
}

interface NPCompletenessProof {
  problem_name: string;
  steps: string[];
  reduction_from: string;
  reduction_construction: string[];
  correctness_proof: string[];
  polynomial_time_proof: string[];
  conclusion: string;
}

interface ComplexityTheoryProps {
  onAnalysisComplete?: (analysis: any) => void;
}

export const ComplexityTheory: React.FC<ComplexityTheoryProps> = ({
  onAnalysisComplete
}) => {
  const [activeTab, setActiveTab] = useState<'hierarchy' | 'reductions' | 'analysis' | 'proofs'>('hierarchy');
  const canvasRef = useRef<HTMLDivElement>(null);
  
  // Hierarchy state
  const [complexityClasses, setComplexityClasses] = useState<ComplexityClass[]>([]);
  const [selectedClass, setSelectedClass] = useState<ComplexityClass | null>(null);
  const [showRelationships, setShowRelationships] = useState(true);
  
  // Reductions state
  const [reductionSteps, setReductionSteps] = useState<ReductionStep[]>([]);
  const [currentReduction, setCurrentReduction] = useState<ReductionStep | null>(null);
  const [fromProblem, setFromProblem] = useState('');
  const [toProblem, setToProblem] = useState('');
  const [reductionType, setReductionType] = useState<ReductionStep['reduction_type']>('polynomial');
  
  // Analysis state
  const [algorithmToAnalyze, setAlgorithmToAnalyze] = useState('');
  const [analysisResult, setAnalysisResult] = useState<Algorithm | null>(null);
  const [customAlgorithm, setCustomAlgorithm] = useState('');
  
  // NP-Completeness proofs state
  const [proofTarget, setProofTarget] = useState('');
  const [npProof, setNpProof] = useState<NPCompletenessProof | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    initializeComplexityClasses();
    loadKnownReductions();
  }, []);

  const initializeComplexityClasses = () => {
    const classes: ComplexityClass[] = [
      {
        name: 'P',
        description: 'Problems solvable in polynomial time',
        examples: ['Sorting', 'Graph traversal', 'Linear programming'],
        relationships: ['P ⊆ NP', 'P ⊆ PSPACE'],
        color: 'bg-green-100 border-green-500 text-green-800',
        position: { x: 200, y: 300 }
      },
      {
        name: 'NP',
        description: 'Problems verifiable in polynomial time',
        examples: ['SAT', 'Hamiltonian Path', 'Vertex Cover'],
        relationships: ['P ⊆ NP', 'NP ⊆ PSPACE', 'NP ⊆ EXPTIME'],
        color: 'bg-blue-100 border-blue-500 text-blue-800',
        position: { x: 350, y: 200 }
      },
      {
        name: 'NP-Complete',
        description: 'Hardest problems in NP',
        examples: ['SAT', '3-SAT', 'Clique', 'Knapsack'],
        relationships: ['NP-Complete ⊆ NP', 'All NP-Complete problems are polynomial-time equivalent'],
        color: 'bg-red-100 border-red-500 text-red-800',
        position: { x: 500, y: 150 }
      },
      {
        name: 'PSPACE',
        description: 'Problems solvable in polynomial space',
        examples: ['QBF', 'Geography game', 'Regular expression equivalence'],
        relationships: ['NP ⊆ PSPACE', 'PSPACE ⊆ EXPTIME'],
        color: 'bg-purple-100 border-purple-500 text-purple-800',
        position: { x: 350, y: 100 }
      },
      {
        name: 'EXPTIME',
        description: 'Problems solvable in exponential time',
        examples: ['Generalized Chess', 'Presburger arithmetic'],
        relationships: ['PSPACE ⊆ EXPTIME'],
        color: 'bg-orange-100 border-orange-500 text-orange-800',
        position: { x: 200, y: 50 }
      }
    ];
    
    setComplexityClasses(classes);
  };

  const loadKnownReductions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/complexity-reductions`);
      const reductions = await response.json();
      setReductionSteps(reductions || []);
    } catch (error) {
      console.error('Failed to load reductions:', error);
    }
  };

  const buildReduction = async () => {
    if (!fromProblem || !toProblem) return;

    setIsProcessing(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/build-reduction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          from_problem: fromProblem,
          to_problem: toProblem,
          reduction_type: reductionType
        })
      });

      const result = await response.json();
      setCurrentReduction(result);
      setReductionSteps(prev => [...prev, result]);
    } catch (error) {
      console.error('Failed to build reduction:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const analyzeAlgorithm = async () => {
    if (!algorithmToAnalyze && !customAlgorithm) return;

    setIsProcessing(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze-algorithm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          algorithm: algorithmToAnalyze || 'custom',
          code: customAlgorithm
        })
      });

      const result = await response.json();
      setAnalysisResult(result);
      
      if (onAnalysisComplete) {
        onAnalysisComplete(result);
      }
    } catch (error) {
      console.error('Failed to analyze algorithm:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const proveNPCompleteness = async () => {
    if (!proofTarget) return;

    setIsProcessing(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/prove-np-complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          problem: proofTarget
        })
      });

      const result = await response.json();
      setNpProof(result);
    } catch (error) {
      console.error('Failed to prove NP-completeness:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const renderHierarchyDiagram = () => (
    <div className="relative w-full h-96 bg-gray-50 rounded border overflow-hidden">
      <div ref={canvasRef} className="relative w-full h-full">
        {/* Render relationship arrows */}
        {showRelationships && (
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                      refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
              </marker>
            </defs>
            
            {/* P ⊆ NP */}
            <line x1="250" y1="300" x2="350" y2="220" 
                  stroke="#666" strokeWidth="2" markerEnd="url(#arrowhead)" />
            
            {/* NP ⊆ PSPACE */}
            <line x1="350" y1="180" x2="350" y2="130" 
                  stroke="#666" strokeWidth="2" markerEnd="url(#arrowhead)" />
            
            {/* PSPACE ⊆ EXPTIME */}
            <line x1="320" y1="100" x2="250" y2="80" 
                  stroke="#666" strokeWidth="2" markerEnd="url(#arrowhead)" />
            
            {/* P ⊆ PSPACE */}
            <line x1="230" y1="280" x2="320" y2="120" 
                  stroke="#666" strokeWidth="2" strokeDasharray="5,5" markerEnd="url(#arrowhead)" />
          </svg>
        )}
        
        {/* Render complexity classes */}
        {complexityClasses.map((cls) => (
          <div
            key={cls.name}
            className={`absolute p-4 rounded-lg border-2 cursor-pointer transition-all hover:shadow-lg ${cls.color} ${
              selectedClass?.name === cls.name ? 'shadow-lg scale-105' : ''
            }`}
            style={{
              left: cls.position.x,
              top: cls.position.y,
              transform: 'translate(-50%, -50%)'
            }}
            onClick={() => setSelectedClass(cls)}
          >
            <div className="text-center">
              <h3 className="font-bold text-lg">{cls.name}</h3>
              <p className="text-xs mt-1 opacity-75">
                {cls.examples.slice(0, 2).join(', ')}
              </p>
            </div>
          </div>
        ))}
      </div>
      
      {/* Controls */}
      <div className="absolute top-4 right-4 flex gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() => setShowRelationships(!showRelationships)}
        >
          <GitBranch className="w-4 h-4 mr-2" />
          {showRelationships ? 'Hide' : 'Show'} Relations
        </Button>
      </div>
    </div>
  );

  const renderHierarchyTab = () => (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-lg font-medium mb-2">Complexity Class Hierarchy</h3>
        <p className="text-sm text-gray-600">
          Click on any complexity class to explore its properties and relationships
        </p>
      </div>

      {renderHierarchyDiagram()}

      {selectedClass && (
        <Card className={`border-2 ${selectedClass.color.replace('bg-', 'border-').replace('-100', '-200')}`}>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2">
              <Network className="w-5 h-5" />
              {selectedClass.name}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm">{selectedClass.description}</p>
            
            <div className="space-y-3">
              <div>
                <h4 className="text-sm font-medium mb-2">Example Problems:</h4>
                <div className="flex flex-wrap gap-2">
                  {selectedClass.examples.map((example, index) => (
                    <Badge key={index} variant="outline" className="text-xs">
                      {example}
                    </Badge>
                  ))}
                </div>
              </div>
              
              <div>
                <h4 className="text-sm font-medium mb-2">Relationships:</h4>
                <div className="space-y-1">
                  {selectedClass.relationships.map((rel, index) => (
                    <div key={index} className="text-sm p-2 bg-white rounded border-l-2 border-blue-200">
                      {rel}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );

  const renderReductionsTab = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Reduction Builder</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">From Problem</label>
            <Input
              value={fromProblem}
              onChange={(e) => setFromProblem(e.target.value)}
              placeholder="e.g., 3-SAT"
            />
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">To Problem</label>
            <Input
              value={toProblem}
              onChange={(e) => setToProblem(e.target.value)}
              placeholder="e.g., Clique"
            />
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Reduction Type</label>
            <Select value={reductionType} onValueChange={(value: any) => setReductionType(value)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="polynomial">Polynomial-time</SelectItem>
                <SelectItem value="logarithmic">Logarithmic-space</SelectItem>
                <SelectItem value="many_one">Many-one</SelectItem>
                <SelectItem value="turing">Turing</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <Button 
          onClick={buildReduction}
          disabled={!fromProblem || !toProblem || isProcessing}
          className="w-full"
        >
          <ArrowRight className="w-4 h-4 mr-2" />
          Build Reduction
        </Button>
      </div>

      {isProcessing && (
        <div className="flex items-center justify-center p-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-sm">Building reduction...</span>
        </div>
      )}

      {currentReduction && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Target className="w-4 h-4" />
              Reduction: {currentReduction.from_problem} → {currentReduction.to_problem}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">Time Complexity:</span>
                <span className="ml-2 font-mono">{currentReduction.time_complexity}</span>
              </div>
              <div>
                <span className="font-medium">Space Complexity:</span>
                <span className="ml-2 font-mono">{currentReduction.space_complexity}</span>
              </div>
            </div>
            
            <div>
              <h4 className="text-sm font-medium mb-2">Description:</h4>
              <p className="text-sm text-gray-700">{currentReduction.description}</p>
            </div>
            
            <div>
              <h4 className="text-sm font-medium mb-2">Construction Steps:</h4>
              <ScrollArea className="h-32">
                <div className="space-y-2">
                  {currentReduction.construction.map((step, index) => (
                    <div key={index} className="text-sm p-2 bg-gray-50 rounded border-l-2 border-blue-200">
                      {index + 1}. {step}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          </CardContent>
        </Card>
      )}

      {reductionSteps.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Known Reductions</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-64">
              <div className="space-y-2">
                {reductionSteps.map((reduction, index) => (
                  <div 
                    key={index}
                    className="p-3 border rounded hover:bg-gray-50 cursor-pointer transition-colors"
                    onClick={() => setCurrentReduction(reduction)}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-sm">
                        {reduction.from_problem} → {reduction.to_problem}
                      </span>
                      <Badge variant="outline" className="text-xs">
                        {reduction.reduction_type}
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-600 mt-1">{reduction.description}</p>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}
    </div>
  );

  const renderAnalysisTab = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Algorithm Complexity Analysis</h3>
        
        <div className="space-y-2">
          <label className="text-sm font-medium">Select Algorithm</label>
          <Select value={algorithmToAnalyze} onValueChange={setAlgorithmToAnalyze}>
            <SelectTrigger>
              <SelectValue placeholder="Choose an algorithm to analyze" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="bubble_sort">Bubble Sort</SelectItem>
              <SelectItem value="merge_sort">Merge Sort</SelectItem>
              <SelectItem value="quick_sort">Quick Sort</SelectItem>
              <SelectItem value="dijkstra">Dijkstra's Algorithm</SelectItem>
              <SelectItem value="floyd_warshall">Floyd-Warshall</SelectItem>
              <SelectItem value="custom">Custom Algorithm</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {algorithmToAnalyze === 'custom' && (
          <div className="space-y-2">
            <label className="text-sm font-medium">Algorithm Code/Pseudocode</label>
            <Textarea
              value={customAlgorithm}
              onChange={(e) => setCustomAlgorithm(e.target.value)}
              placeholder="Enter your algorithm code or pseudocode..."
              className="min-h-32 font-mono text-sm"
            />
          </div>
        )}

        <Button 
          onClick={analyzeAlgorithm}
          disabled={!algorithmToAnalyze || (algorithmToAnalyze === 'custom' && !customAlgorithm) || isProcessing}
          className="w-full"
        >
          <Calculator className="w-4 h-4 mr-2" />
          Analyze Complexity
        </Button>
      </div>

      {isProcessing && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Analyzing algorithm...</span>
          </div>
          <Progress value={60} className="h-2" />
        </div>
      )}

      {analysisResult && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Analysis: {analysisResult.name}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Time Complexity</h4>
                <Badge className="bg-blue-100 text-blue-800 font-mono">
                  {analysisResult.time_complexity}
                </Badge>
              </div>
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Space Complexity</h4>
                <Badge className="bg-green-100 text-green-800 font-mono">
                  {analysisResult.space_complexity}
                </Badge>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium mb-2">Description:</h4>
              <p className="text-sm text-gray-700">{analysisResult.description}</p>
            </div>

            <div>
              <h4 className="text-sm font-medium mb-2">Pseudocode:</h4>
              <ScrollArea className="h-32">
                <div className="space-y-1">
                  {analysisResult.pseudocode.map((line, index) => (
                    <div key={index} className="text-sm font-mono p-2 bg-gray-50 rounded">
                      {line}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>

            <div>
              <h4 className="text-sm font-medium mb-2">Analysis Steps:</h4>
              <ScrollArea className="h-32">
                <div className="space-y-2">
                  {analysisResult.analysis_steps.map((step, index) => (
                    <div key={index} className="text-sm p-2 bg-blue-50 rounded border-l-2 border-blue-200">
                      {index + 1}. {step}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );

  const renderProofsTab = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium">NP-Completeness Proof Assistant</h3>
        
        <div className="space-y-2">
          <label className="text-sm font-medium">Problem to Prove NP-Complete</label>
          <Select value={proofTarget} onValueChange={setProofTarget}>
            <SelectTrigger>
              <SelectValue placeholder="Select a problem" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="vertex_cover">Vertex Cover</SelectItem>
              <SelectItem value="hamiltonian_path">Hamiltonian Path</SelectItem>
              <SelectItem value="3_coloring">3-Coloring</SelectItem>
              <SelectItem value="subset_sum">Subset Sum</SelectItem>
              <SelectItem value="knapsack">0/1 Knapsack</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Button 
          onClick={proveNPCompleteness}
          disabled={!proofTarget || isProcessing}
          className="w-full"
        >
          <Lightbulb className="w-4 h-4 mr-2" />
          Generate NP-Completeness Proof
        </Button>
      </div>

      {isProcessing && (
        <div className="flex items-center justify-center p-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-sm">Generating proof...</span>
        </div>
      )}

      {npProof && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-600" />
              NP-Completeness Proof: {npProof.problem_name}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-3 bg-blue-50 rounded border border-blue-200">
              <h4 className="text-sm font-medium mb-2">Proof Strategy:</h4>
              <p className="text-sm">
                Reduction from <strong>{npProof.reduction_from}</strong> to <strong>{npProof.problem_name}</strong>
              </p>
            </div>

            <div>
              <h4 className="text-sm font-medium mb-2">Proof Steps:</h4>
              <ScrollArea className="h-40">
                <div className="space-y-3">
                  {npProof.steps.map((step, index) => (
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
            </div>

            <Separator />

            <div>
              <h4 className="text-sm font-medium mb-2">Reduction Construction:</h4>
              <ScrollArea className="h-32">
                <div className="space-y-2">
                  {npProof.reduction_construction.map((step, index) => (
                    <div key={index} className="text-sm p-2 bg-yellow-50 rounded border-l-2 border-yellow-200">
                      {index + 1}. {step}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>

            <div>
              <h4 className="text-sm font-medium mb-2">Correctness Proof:</h4>
              <ScrollArea className="h-32">
                <div className="space-y-2">
                  {npProof.correctness_proof.map((step, index) => (
                    <div key={index} className="text-sm p-2 bg-green-50 rounded border-l-2 border-green-200">
                      {index + 1}. {step}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>

            <div>
              <h4 className="text-sm font-medium mb-2">Polynomial-Time Proof:</h4>
              <ScrollArea className="h-24">
                <div className="space-y-2">
                  {npProof.polynomial_time_proof.map((step, index) => (
                    <div key={index} className="text-sm p-2 bg-purple-50 rounded border-l-2 border-purple-200">
                      {index + 1}. {step}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>

            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertDescription>
                <strong>Conclusion:</strong> {npProof.conclusion}
              </AlertDescription>
            </Alert>
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
          Complexity Theory
        </CardTitle>
      </CardHeader>
      
      <CardContent>
        <Tabs value={activeTab} onValueChange={(value: any) => setActiveTab(value)}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="hierarchy">Hierarchy</TabsTrigger>
            <TabsTrigger value="reductions">Reductions</TabsTrigger>
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
            <TabsTrigger value="proofs">Proofs</TabsTrigger>
          </TabsList>

          <TabsContent value="hierarchy" className="mt-6">
            {renderHierarchyTab()}
          </TabsContent>

          <TabsContent value="reductions" className="mt-6">
            {renderReductionsTab()}
          </TabsContent>

          <TabsContent value="analysis" className="mt-6">
            {renderAnalysisTab()}
          </TabsContent>

          <TabsContent value="proofs" className="mt-6">
            {renderProofsTab()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};