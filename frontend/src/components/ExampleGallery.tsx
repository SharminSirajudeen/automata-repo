import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Search, Copy, Eye } from 'lucide-react';
import { ExtendedAutomaton, AutomataType } from '../types/automata';

interface ExampleAutomaton {
  id: string;
  name: string;
  type: AutomataType;
  description: string;
  automaton: ExtendedAutomaton;
  tags: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

interface ExampleGalleryProps {
  onLoadExample: (automaton: ExtendedAutomaton) => void;
}

export const ExampleGallery: React.FC<ExampleGalleryProps> = ({ onLoadExample }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedType, setSelectedType] = useState<AutomataType | 'all'>('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState<'all' | 'beginner' | 'intermediate' | 'advanced'>('all');

  const exampleAutomata: ExampleAutomaton[] = [
    {
      id: 'dfa-binary-even',
      name: 'Even Binary Numbers',
      type: 'dfa',
      description: 'DFA that accepts binary strings representing even numbers',
      automaton: {
        type: 'dfa',
        states: [
          { id: 'q0', x: 100, y: 100, is_start: true, is_accept: true, label: 'q0' },
          { id: 'q1', x: 300, y: 100, is_start: false, is_accept: false, label: 'q1' }
        ],
        transitions: [
          { from_state: 'q0', to_state: 'q0', symbol: '0' },
          { from_state: 'q0', to_state: 'q1', symbol: '1' },
          { from_state: 'q1', to_state: 'q0', symbol: '0' },
          { from_state: 'q1', to_state: 'q1', symbol: '1' }
        ],
        alphabet: ['0', '1']
      },
      tags: ['binary', 'even', 'basic'],
      difficulty: 'beginner'
    },
    {
      id: 'nfa-substring',
      name: 'Contains "101"',
      type: 'nfa',
      description: 'NFA that accepts strings containing the substring "101"',
      automaton: {
        type: 'nfa',
        states: [
          { id: 'q0', x: 50, y: 100, is_start: true, is_accept: false, label: 'q0' },
          { id: 'q1', x: 150, y: 100, is_start: false, is_accept: false, label: 'q1' },
          { id: 'q2', x: 250, y: 100, is_start: false, is_accept: false, label: 'q2' },
          { id: 'q3', x: 350, y: 100, is_start: false, is_accept: true, label: 'q3' }
        ],
        transitions: [
          { from_state: 'q0', to_state: 'q0', symbol: '0' },
          { from_state: 'q0', to_state: 'q0', symbol: '1' },
          { from_state: 'q0', to_state: 'q1', symbol: '1' },
          { from_state: 'q1', to_state: 'q2', symbol: '0' },
          { from_state: 'q2', to_state: 'q3', symbol: '1' },
          { from_state: 'q3', to_state: 'q3', symbol: '0' },
          { from_state: 'q3', to_state: 'q3', symbol: '1' }
        ],
        alphabet: ['0', '1']
      },
      tags: ['substring', 'nfa', 'pattern'],
      difficulty: 'intermediate'
    },
    {
      id: 'pda-balanced',
      name: 'Balanced Parentheses',
      type: 'pda',
      description: 'PDA that accepts strings with balanced parentheses',
      automaton: {
        type: 'pda',
        states: [
          { id: 'q0', x: 100, y: 100, is_start: true, is_accept: false, label: 'q0' },
          { id: 'q1', x: 300, y: 100, is_start: false, is_accept: true, label: 'q1' }
        ],
        transitions: [
          { from_state: 'q0', to_state: 'q0', symbol: '(', stack_pop: 'Z', stack_push: 'XZ' },
          { from_state: 'q0', to_state: 'q0', symbol: '(', stack_pop: 'X', stack_push: 'XX' },
          { from_state: 'q0', to_state: 'q0', symbol: ')', stack_pop: 'X', stack_push: '' },
          { from_state: 'q0', to_state: 'q1', symbol: '', stack_pop: 'Z', stack_push: 'Z' }
        ],
        alphabet: ['(', ')'],
        stack_alphabet: ['Z', 'X']
      } as any,
      tags: ['parentheses', 'balanced', 'context-free'],
      difficulty: 'advanced'
    }
  ];

  const filteredExamples = exampleAutomata.filter(example => {
    const matchesSearch = example.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         example.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         example.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesType = selectedType === 'all' || example.type === selectedType;
    const matchesDifficulty = selectedDifficulty === 'all' || example.difficulty === selectedDifficulty;
    
    return matchesSearch && matchesType && matchesDifficulty;
  });

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

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Eye className="w-5 h-5 text-purple-600" />
          Example Gallery
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              placeholder="Search examples..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          
          <div className="flex gap-2">
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value as AutomataType | 'all')}
              className="px-3 py-1 border rounded text-sm"
            >
              <option value="all">All Types</option>
              <option value="dfa">DFA</option>
              <option value="nfa">NFA</option>
              <option value="enfa">ε-NFA</option>
              <option value="pda">PDA</option>
              <option value="cfg">CFG</option>
              <option value="tm">Turing Machine</option>
            </select>
            
            <select
              value={selectedDifficulty}
              onChange={(e) => setSelectedDifficulty(e.target.value as any)}
              className="px-3 py-1 border rounded text-sm"
            >
              <option value="all">All Levels</option>
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
          </div>
        </div>

        <div className="space-y-3 max-h-96 overflow-y-auto">
          {filteredExamples.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Eye className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No examples found</p>
              <p className="text-xs">Try adjusting your search criteria</p>
            </div>
          ) : (
            filteredExamples.map(example => (
              <div key={example.id} className="p-3 border rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <h3 className="font-medium text-sm">{example.name}</h3>
                    <p className="text-xs text-gray-600 mt-1">{example.description}</p>
                  </div>
                  
                  <div className="flex gap-1 ml-2">
                    <Button
                      onClick={() => onLoadExample(example.automaton)}
                      size="sm"
                      variant="outline"
                      className="h-7 px-2"
                    >
                      <Copy className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
                
                <div className="flex items-center gap-2 flex-wrap">
                  <Badge variant="secondary" className="text-xs">
                    {getTypeDisplayName(example.type)}
                  </Badge>
                  <Badge className={`text-xs ${getDifficultyColor(example.difficulty)}`}>
                    {example.difficulty}
                  </Badge>
                  {example.tags.slice(0, 2).map(tag => (
                    <Badge key={tag} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                  {example.tags.length > 2 && (
                    <span className="text-xs text-gray-500">+{example.tags.length - 2} more</span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
};
