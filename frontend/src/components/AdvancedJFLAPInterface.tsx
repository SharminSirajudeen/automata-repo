import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import { 
  Cpu, 
  FileText, 
  GitBranch, 
  Zap, 
  Palette, 
  Info,
  Save,
  FolderOpen,
  Share
} from 'lucide-react';

import { MultiTapeTuringMachine } from './MultiTapeTuringMachine';
import { AdvancedGrammarEditor } from './AdvancedGrammarEditor';
import { SLRParserVisualization } from './SLRParserVisualization';
import { UniversalTuringMachine } from './UniversalTuringMachine';
import { LSystemRenderer } from './LSystemRenderer';

import {
  AdvancedAutomaton,
  MultiTapeTMAutomaton,
  UnrestrictedGrammar,
  SLRParser,
  UniversalTM,
  LSystem
} from '../types/advanced-automata';

interface AdvancedJFLAPInterfaceProps {
  onExportProject?: (project: any) => void;
  onImportProject?: (project: any) => void;
  onRequestAIGuidance?: (context: string) => void;
}

type AdvancedFeatureType = 'multi-tape-tm' | 'unrestricted-grammar' | 'slr-parser' | 'universal-tm' | 'l-system';

const defaultAutomata: { [K in AdvancedFeatureType]: AdvancedAutomaton } = {
  'multi-tape-tm': {
    type: 'multi-tape-tm',
    states: [
      { id: 'q0', x: 100, y: 100, is_start: true, is_accept: false, label: 'q0' },
      { id: 'q1', x: 200, y: 100, is_start: false, is_accept: true, label: 'q1' }
    ],
    transitions: [],
    tape_alphabet: ['0', '1', 'B'],
    blank_symbol: 'B',
    num_tapes: 2,
    input_tape_index: 0
  } as MultiTapeTMAutomaton,
  
  'unrestricted-grammar': {
    type: 'unrestricted-grammar',
    terminals: ['a', 'b'],
    non_terminals: ['S', 'A', 'B'],
    productions: [
      {
        id: 'prod1',
        left_side: ['S'],
        right_side: ['A', 'B'],
        context_sensitive: false
      }
    ],
    start_symbol: 'S'
  } as UnrestrictedGrammar,
  
  'slr-parser': {
    type: 'slr-parser',
    grammar: {
      type: 'unrestricted-grammar',
      terminals: ['id', '+', '*', '(', ')'],
      non_terminals: ['E', 'T', 'F'],
      productions: [
        { id: 'p1', left_side: ['E'], right_side: ['E', '+', 'T'] },
        { id: 'p2', left_side: ['E'], right_side: ['T'] },
        { id: 'p3', left_side: ['T'], right_side: ['T', '*', 'F'] },
        { id: 'p4', left_side: ['T'], right_side: ['F'] },
        { id: 'p5', left_side: ['F'], right_side: ['(', 'E', ')'] },
        { id: 'p6', left_side: ['F'], right_side: ['id'] }
      ],
      start_symbol: 'E'
    },
    states: [],
    action_table: {},
    goto_table: {}
  } as SLRParser,
  
  'universal-tm': {
    type: 'universal-tm',
    encoded_tm: '',
    encoding_scheme: 'standard',
    control_states: ['q0', 'q1', 'q2']
  } as UniversalTM,
  
  'l-system': {
    type: 'l-system',
    axiom: 'F',
    productions: {
      'F': 'F+F-F-F+F'
    },
    iterations: 3,
    angle: 90,
    turtle_commands: [
      { symbol: 'F', action: 'forward', value: 10 },
      { symbol: '+', action: 'turn_left' },
      { symbol: '-', action: 'turn_right' }
    ],
    render_3d: false
  } as LSystem
};

export const AdvancedJFLAPInterface: React.FC<AdvancedJFLAPInterfaceProps> = ({
  onExportProject,
  onImportProject,
  onRequestAIGuidance
}) => {
  const [selectedFeature, setSelectedFeature] = useState<AdvancedFeatureType>('multi-tape-tm');
  const [automata, setAutomata] = useState<{ [K in AdvancedFeatureType]: AdvancedAutomaton }>(defaultAutomata);
  const [projectName, setProjectName] = useState('Advanced JFLAP Project');
  const [lastSaved, setLastSaved] = useState<Date | null>(null);

  const handleAutomatonChange = useCallback((type: AdvancedFeatureType, newAutomaton: AdvancedAutomaton) => {
    setAutomata(prev => ({
      ...prev,
      [type]: newAutomaton
    }));
  }, []);

  const handleSaveProject = () => {
    const project = {
      name: projectName,
      automata,
      version: '2.0.0',
      created: new Date().toISOString(),
      type: 'advanced-jflap'
    };
    
    onExportProject?.(project);
    setLastSaved(new Date());
  };

  const handleLoadProject = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const project = JSON.parse(e.target?.result as string);
        if (project.type === 'advanced-jflap') {
          setAutomata(project.automata);
          setProjectName(project.name);
          onImportProject?.(project);
        }
      } catch (error) {
        console.error('Failed to load project:', error);
      }
    };
    reader.readAsText(file);
  };

  const getFeatureIcon = (feature: AdvancedFeatureType) => {
    switch (feature) {
      case 'multi-tape-tm': return <Cpu className="w-4 h-4" />;
      case 'unrestricted-grammar': return <FileText className="w-4 h-4" />;
      case 'slr-parser': return <GitBranch className="w-4 h-4" />;
      case 'universal-tm': return <Zap className="w-4 h-4" />;
      case 'l-system': return <Palette className="w-4 h-4" />;
    }
  };

  const getFeatureName = (feature: AdvancedFeatureType) => {
    switch (feature) {
      case 'multi-tape-tm': return 'Multi-Tape Turing Machine';
      case 'unrestricted-grammar': return 'Unrestricted Grammar';
      case 'slr-parser': return 'SLR(1) Parser';
      case 'universal-tm': return 'Universal Turing Machine';
      case 'l-system': return 'L-System Graphics';
    }
  };

  const getFeatureDescription = (feature: AdvancedFeatureType) => {
    switch (feature) {
      case 'multi-tape-tm': 
        return 'Design and simulate Turing machines with multiple tapes for complex computations';
      case 'unrestricted-grammar': 
        return 'Create and validate unrestricted and context-sensitive grammars with advanced parsing';
      case 'slr-parser': 
        return 'Build SLR(1) parsers with DFA visualization and step-by-step parsing traces';
      case 'universal-tm': 
        return 'Encode and simulate Turing machines using a Universal Turing Machine';
      case 'l-system': 
        return 'Generate fractal patterns and organic structures using L-system rules';
    }
  };

  const renderCurrentFeature = () => {
    const currentAutomaton = automata[selectedFeature];

    switch (selectedFeature) {
      case 'multi-tape-tm':
        return (
          <MultiTapeTuringMachine
            automaton={currentAutomaton as MultiTapeTMAutomaton}
            onAutomatonChange={(newAutomaton) => handleAutomatonChange(selectedFeature, newAutomaton)}
          />
        );
      
      case 'unrestricted-grammar':
        return (
          <AdvancedGrammarEditor
            grammar={currentAutomaton as UnrestrictedGrammar}
            onGrammarChange={(newGrammar) => handleAutomatonChange(selectedFeature, newGrammar)}
          />
        );
      
      case 'slr-parser':
        return (
          <SLRParserVisualization
            parser={currentAutomaton as SLRParser}
            onParserChange={(newParser) => handleAutomatonChange(selectedFeature, newParser)}
          />
        );
      
      case 'universal-tm':
        return (
          <UniversalTuringMachine
            universalTM={currentAutomaton as UniversalTM}
            onUniversalTMChange={(newUTM) => handleAutomatonChange(selectedFeature, newUTM)}
          />
        );
      
      case 'l-system':
        return (
          <LSystemRenderer
            lsystem={currentAutomaton as LSystem}
            onLSystemChange={(newLSystem) => handleAutomatonChange(selectedFeature, newLSystem)}
          />
        );
      
      default:
        return <div>Feature not implemented</div>;
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                Advanced JFLAP Features
                <Badge variant="secondary">v2.0</Badge>
              </CardTitle>
              <p className="text-sm text-gray-600 mt-1">
                Professional automata theory tools with advanced visualizations
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline" onClick={handleSaveProject}>
                <Save className="w-4 h-4 mr-1" />
                Save Project
              </Button>
              <label className="cursor-pointer">
                <Button size="sm" variant="outline" asChild>
                  <span>
                    <FolderOpen className="w-4 h-4 mr-1" />
                    Load Project
                  </span>
                </Button>
                <input
                  type="file"
                  accept=".json"
                  className="hidden"
                  onChange={handleLoadProject}
                />
              </label>
              {onRequestAIGuidance && (
                <Button
                  size="sm"
                  onClick={() => onRequestAIGuidance(`Advanced JFLAP: ${getFeatureName(selectedFeature)}`)}
                  className="bg-gradient-to-r from-purple-500 to-blue-500 text-white"
                >
                  AI Guidance
                </Button>
              )}
            </div>
          </div>
          {lastSaved && (
            <p className="text-xs text-gray-500">
              Last saved: {lastSaved.toLocaleTimeString()}
            </p>
          )}
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Select
              value={selectedFeature}
              onValueChange={(value: AdvancedFeatureType) => setSelectedFeature(value)}
            >
              <SelectTrigger className="w-80">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(defaultAutomata).map((feature) => (
                  <SelectItem key={feature} value={feature}>
                    <div className="flex items-center gap-2">
                      {getFeatureIcon(feature as AdvancedFeatureType)}
                      {getFeatureName(feature as AdvancedFeatureType)}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Alert className="flex-1">
              <Info className="h-4 w-4" />
              <AlertDescription>
                {getFeatureDescription(selectedFeature)}
              </AlertDescription>
            </Alert>
          </div>
        </CardContent>
      </Card>

      {/* Feature Interface */}
      {renderCurrentFeature()}

      {/* Footer */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div className="flex items-center gap-4">
              <span>Current Feature: {getFeatureName(selectedFeature)}</span>
              <Badge variant="outline">
                {selectedFeature === 'multi-tape-tm' && `${(automata[selectedFeature] as MultiTapeTMAutomaton).num_tapes} Tapes`}
                {selectedFeature === 'unrestricted-grammar' && `${(automata[selectedFeature] as UnrestrictedGrammar).productions.length} Productions`}
                {selectedFeature === 'slr-parser' && `${(automata[selectedFeature] as SLRParser).states.length} States`}
                {selectedFeature === 'universal-tm' && (automata[selectedFeature] as UniversalTM).encoding_scheme}
                {selectedFeature === 'l-system' && `Gen ${(automata[selectedFeature] as LSystem).iterations}`}
              </Badge>
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="ghost">
                <Share className="w-4 h-4 mr-1" />
                Share
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};