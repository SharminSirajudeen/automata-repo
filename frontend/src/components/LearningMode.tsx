import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { HelpCircle, Lightbulb, AlertCircle, BookOpen, Target } from 'lucide-react';
import { AutomataType } from '../types/automata';

interface LearningModeProps {
  automatonType: AutomataType;
  isEnabled: boolean;
  onToggle: (enabled: boolean) => void;
  onHintRequest: () => void;
  onStepGuidance: () => void;
}

export const LearningMode: React.FC<LearningModeProps> = ({
  automatonType,
  isEnabled,
  onToggle,
  onHintRequest,
  onStepGuidance
}) => {
  const [currentHint, setCurrentHint] = useState<string | null>(null);

  const getAutomataExplanations = () => {
    const explanations: { [key in AutomataType]: { [key: string]: string } } = {
      'dfa': {
        'state': 'A state represents a condition or situation in the automaton. States can be initial (start) or final (accept).',
        'transition': 'A transition shows how the automaton moves from one state to another when reading a specific symbol.',
        'alphabet': 'The alphabet is the set of symbols that the automaton can read from the input string.',
        'final_state': 'Final states (accept states) determine if the automaton accepts the input string.',
        'start_state': 'The start state is where the automaton begins processing the input string.'
      },
      'nfa': {
        'state': 'In NFAs, states can have multiple transitions for the same symbol, allowing non-deterministic choices.',
        'transition': 'NFA transitions can be non-deterministic - multiple paths possible for the same input symbol.',
        'epsilon': 'ε-transitions allow the automaton to move between states without consuming input symbols.',
        'acceptance': 'An NFA accepts if ANY possible path leads to an accept state.'
      },
      'enfa': {
        'epsilon_transition': 'ε-transitions (empty transitions) allow movement without reading input symbols.',
        'closure': 'ε-closure includes all states reachable via ε-transitions from a given state.',
        'conversion': 'ε-NFAs can be converted to regular NFAs by eliminating ε-transitions.'
      },
      'pda': {
        'stack': 'The stack provides infinite memory for the PDA to recognize context-free languages.',
        'push': 'Push operations add symbols to the top of the stack.',
        'pop': 'Pop operations remove symbols from the top of the stack.',
        'stack_symbol': 'Stack symbols can be different from input alphabet symbols.'
      },
      'cfg': {
        'production': 'Production rules define how non-terminals can be replaced with terminals and non-terminals.',
        'derivation': 'A derivation shows step-by-step replacement of non-terminals using production rules.',
        'parse_tree': 'Parse trees show the hierarchical structure of how a string is generated.',
        'ambiguity': 'A grammar is ambiguous if a string has multiple parse trees.'
      },
      'tm': {
        'tape': 'The tape provides infinite memory and can be read from and written to.',
        'head': 'The tape head can move left, right, or stay in the same position.',
        'halt': 'A Turing machine halts when it reaches a state with no valid transitions.',
        'computation': 'TM computation involves reading, writing, and moving the head based on current state and symbol.'
      },
      'regex': {
        'concatenation': 'Concatenation combines patterns in sequence (ab means a followed by b).',
        'union': 'Union (|) matches either pattern (a|b means a or b).',
        'kleene_star': 'Kleene star (*) matches zero or more repetitions of a pattern.',
        'precedence': 'Operator precedence: * (highest), concatenation, | (lowest).'
      },
      'pumping': {
        'pumping_length': 'The pumping length is the minimum length where pumping property applies.',
        'decomposition': 'Every long string can be decomposed into xyz where y can be pumped.',
        'contradiction': 'To prove non-regularity, show that pumping leads to strings outside the language.',
        'witness': 'Choose a witness string that cannot be pumped while staying in the language.'
      }
    };

    return explanations[automatonType] || {};
  };

  const getStepByStepGuidance = () => {
    const guidance: { [key in AutomataType]: string[] } = {
      'dfa': [
        '1. Identify the language pattern from the problem description',
        '2. Determine the minimum number of states needed',
        '3. Create the start state and mark it clearly',
        '4. Add transitions for each alphabet symbol from each state',
        '5. Mark accept states based on the language definition',
        '6. Verify with test cases'
      ],
      'nfa': [
        '1. Understand the non-deterministic nature of the problem',
        '2. Create states for different possible paths',
        '3. Add multiple transitions for the same symbol if needed',
        '4. Use ε-transitions for convenience (if ε-NFA)',
        '5. Mark all possible accept states',
        '6. Test with various input strings'
      ],
      'enfa': [
        '1. Design the basic NFA structure',
        '2. Add ε-transitions to simplify the design',
        '3. Ensure ε-closure includes all reachable states',
        '4. Verify acceptance conditions with ε-transitions',
        '5. Consider conversion to regular NFA if needed'
      ],
      'pda': [
        '1. Identify the context-free pattern (often nested or balanced)',
        '2. Design states for different phases of recognition',
        '3. Use stack to remember important information',
        '4. Define push/pop operations for each transition',
        '5. Ensure proper stack management for acceptance',
        '6. Test with nested/balanced examples'
      ],
      'cfg': [
        '1. Identify terminals and non-terminals',
        '2. Choose a start symbol',
        '3. Write production rules for the language pattern',
        '4. Ensure all strings in the language can be generated',
        '5. Check for ambiguity with example derivations',
        '6. Simplify grammar if possible'
      ],
      'tm': [
        '1. Understand the computation required',
        '2. Design states for different phases of computation',
        '3. Define tape alphabet including work symbols',
        '4. Specify read/write/move operations for each transition',
        '5. Ensure proper halting conditions',
        '6. Trace through example computations'
      ],
      'regex': [
        '1. Identify the pattern components',
        '2. Use basic operators: concatenation, union, Kleene star',
        '3. Apply operator precedence rules',
        '4. Use parentheses for grouping when needed',
        '5. Test with positive and negative examples',
        '6. Simplify the expression if possible'
      ],
      'pumping': [
        '1. Assume the language is regular for contradiction',
        '2. Let p be the pumping length',
        '3. Choose a witness string w with |w| ≥ p',
        '4. Consider all possible decompositions w = xyz',
        '5. Show that pumping xy^i z leads to contradiction',
        '6. Conclude the language is not regular'
      ]
    };

    return guidance[automatonType] || [];
  };

  const getPuzzleMode = () => {
    return {
      title: 'Puzzle Mode',
      description: 'Complete the missing parts of this automaton',
      missingElements: [
        'Add the missing final state',
        'Complete the transition from state q1',
        'Determine the correct alphabet symbol'
      ]
    };
  };

  const explanations = getAutomataExplanations();
  const stepGuidance = getStepByStepGuidance();
  const puzzleMode = getPuzzleMode();

  return (
    <TooltipProvider>
      <Card className="h-full">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-blue-600" />
            Learning Mode
            <div className="ml-auto flex items-center gap-2">
              <Badge variant={isEnabled ? "default" : "outline"}>
                {isEnabled ? 'ON' : 'OFF'}
              </Badge>
              <Button
                size="sm"
                variant={isEnabled ? "secondary" : "default"}
                onClick={() => onToggle(!isEnabled)}
              >
                {isEnabled ? 'Disable' : 'Enable'}
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {isEnabled && (
            <>
              <div className="space-y-3">
                <h3 className="font-medium text-gray-900 flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Step-by-Step Guidance
                </h3>
                <div className="space-y-2">
                  {stepGuidance.map((step, index) => (
                    <div key={index} className="flex items-start gap-2 p-2 bg-blue-50 rounded">
                      <div className="w-5 h-5 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold mt-0.5">
                        {index + 1}
                      </div>
                      <p className="text-sm text-blue-800">{step}</p>
                    </div>
                  ))}
                </div>
                <Button onClick={onStepGuidance} className="w-full" size="sm">
                  Get Next Step Guidance
                </Button>
              </div>

              <div className="space-y-3">
                <h3 className="font-medium text-gray-900 flex items-center gap-2">
                  <HelpCircle className="w-4 h-4" />
                  Interactive Explanations
                </h3>
                <div className="grid grid-cols-1 gap-2">
                  {Object.entries(explanations).map(([concept, explanation]) => (
                    <Tooltip key={concept}>
                      <TooltipTrigger asChild>
                        <div className="p-2 bg-gray-50 rounded cursor-help hover:bg-gray-100 transition-colors">
                          <div className="flex items-center gap-2">
                            <HelpCircle className="w-3 h-3 text-gray-500" />
                            <span className="text-sm font-medium capitalize">
                              {concept.replace('_', ' ')}
                            </span>
                          </div>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-sm">{explanation}</p>
                      </TooltipContent>
                    </Tooltip>
                  ))}
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="font-medium text-gray-900 flex items-center gap-2">
                  <Lightbulb className="w-4 h-4" />
                  Smart Hints
                </h3>
                <Button 
                  onClick={onHintRequest} 
                  variant="outline" 
                  className="w-full"
                  size="sm"
                >
                  Request Hint
                </Button>
                {currentHint && (
                  <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
                    <div className="flex items-start gap-2">
                      <Lightbulb className="w-4 h-4 text-yellow-600 mt-0.5" />
                      <p className="text-sm text-yellow-800">{currentHint}</p>
                    </div>
                  </div>
                )}
              </div>

              <div className="space-y-3">
                <h3 className="font-medium text-gray-900 flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  {puzzleMode.title}
                </h3>
                <p className="text-sm text-gray-600">{puzzleMode.description}</p>
                <div className="space-y-2">
                  {puzzleMode.missingElements.map((element, index) => (
                    <div key={index} className="flex items-center gap-2 p-2 bg-orange-50 rounded">
                      <AlertCircle className="w-4 h-4 text-orange-600" />
                      <span className="text-sm text-orange-800">{element}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="font-medium text-gray-900">What Went Wrong?</h3>
                <Button 
                  variant="outline" 
                  className="w-full" 
                  size="sm"
                  onClick={() => {
                    setCurrentHint("Check if all states are reachable from the start state.");
                  }}
                >
                  <AlertCircle className="w-4 h-4 mr-2" />
                  Diagnose Issues
                </Button>
              </div>
            </>
          )}

          {!isEnabled && (
            <div className="text-center py-8 text-gray-500">
              <BookOpen className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>Enable Learning Mode for step-by-step guidance, hints, and interactive explanations.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};
