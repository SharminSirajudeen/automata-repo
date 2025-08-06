import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { CheckCircle, Lock, Clock, BookOpen } from 'lucide-react';
import { useProgressStorage } from '@/hooks/useProgressStorage';

interface Module {
  id: string;
  title: string;
  description: string;
  concepts: string[];
  prerequisites: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  estimatedHours: number;
  learningObjectives: string[];
}

const modules: Module[] = [
  {
    id: 'mod-1',
    title: 'Mathematical Foundations',
    description: 'Essential mathematical concepts for automata theory',
    concepts: [],
    prerequisites: [],
    difficulty: 'beginner',
    estimatedHours: 4,
    learningObjectives: [
      'Understand sets, relations, and functions',
      'Master proof techniques (induction, contradiction)',
      'Work with formal languages and alphabets'
    ]
  },
  {
    id: 'mod-2',
    title: 'Finite Automata and Regular Languages',
    description: 'Introduction to DFA, NFA, and regular expressions',
    concepts: ['DFA', 'NFA', 'Regular Expressions'],
    prerequisites: ['mod-1'],
    difficulty: 'beginner',
    estimatedHours: 8,
    learningObjectives: [
      'Design and analyze DFAs for pattern recognition',
      'Convert between DFA, NFA, and regular expressions',
      'Apply closure properties of regular languages'
    ]
  },
  {
    id: 'mod-3',
    title: 'Context-Free Languages',
    description: 'CFGs, PDAs, and parsing techniques',
    concepts: ['CFG', 'PDA', 'Parsing'],
    prerequisites: ['mod-2'],
    difficulty: 'intermediate',
    estimatedHours: 10,
    learningObjectives: [
      'Design context-free grammars',
      'Construct pushdown automata',
      'Apply CYK parsing algorithm'
    ]
  },
  {
    id: 'mod-4',
    title: 'Turing Machines',
    description: 'Universal computation and decidability',
    concepts: ['Turing Machine', 'Decidability'],
    prerequisites: ['mod-3'],
    difficulty: 'advanced',
    estimatedHours: 12,
    learningObjectives: [
      'Design Turing machines',
      'Understand Church-Turing thesis',
      'Prove undecidability'
    ]
  }
];

export const CourseStructure: React.FC = () => {
  const { progress } = useProgressStorage();
  
  const isModuleCompleted = (moduleId: string) => {
    // Simple check - module is complete if certain problems are done
    const moduleProblems = {
      'mod-1': ['problem-1', 'problem-2', 'problem-3'],
      'mod-2': ['problem-4', 'problem-5', 'problem-6'],
      'mod-3': ['problem-7', 'problem-8', 'problem-9'],
      'mod-4': ['problem-10', 'problem-11', 'problem-12']
    };
    
    const required = moduleProblems[moduleId] || [];
    return required.every(p => progress.completedProblems.includes(p));
  };
  
  const arePrerequisitesMet = (prerequisites: string[]) => {
    return prerequisites.every(prereq => isModuleCompleted(prereq));
  };
  
  const getDifficultyColor = (difficulty: string) => {
    const colors = {
      beginner: 'bg-green-100 text-green-800',
      intermediate: 'bg-yellow-100 text-yellow-800',
      advanced: 'bg-orange-100 text-orange-800',
      expert: 'bg-red-100 text-red-800'
    };
    return colors[difficulty] || colors.beginner;
  };
  
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold mb-2">Course Structure</h2>
        <p className="text-gray-600">
          MIT/Oxford-level curriculum for Theory of Computation
        </p>
      </div>
      
      <div className="grid gap-4">
        {modules.map((module) => {
          const completed = isModuleCompleted(module.id);
          const unlocked = arePrerequisitesMet(module.prerequisites);
          
          return (
            <Card 
              key={module.id}
              className={`transition-all ${
                !unlocked ? 'opacity-60' : ''
              } ${completed ? 'border-green-500' : ''}`}
            >
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="flex items-center gap-2">
                      {completed ? (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      ) : !unlocked ? (
                        <Lock className="w-5 h-5 text-gray-400" />
                      ) : (
                        <BookOpen className="w-5 h-5 text-blue-500" />
                      )}
                      {module.title}
                    </CardTitle>
                    <CardDescription className="mt-1">
                      {module.description}
                    </CardDescription>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <Badge className={getDifficultyColor(module.difficulty)}>
                      {module.difficulty}
                    </Badge>
                    <div className="flex items-center gap-1 text-sm text-gray-500">
                      <Clock className="w-4 h-4" />
                      {module.estimatedHours}h
                    </div>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent>
                {module.concepts.length > 0 && (
                  <div className="mb-4">
                    <div className="flex gap-2 flex-wrap">
                      {module.concepts.map((concept) => (
                        <Badge key={concept} variant="secondary">
                          {concept}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
                
                <div className="space-y-2">
                  <h4 className="font-semibold text-sm">Learning Objectives:</h4>
                  <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                    {module.learningObjectives.map((objective, i) => (
                      <li key={i}>{objective}</li>
                    ))}
                  </ul>
                </div>
                
                {module.prerequisites.length > 0 && (
                  <div className="mt-4 text-sm text-gray-500">
                    Prerequisites: {module.prerequisites.join(', ')}
                  </div>
                )}
                
                <div className="mt-4">
                  <Button 
                    className="w-full" 
                    disabled={!unlocked}
                    variant={completed ? "secondary" : "default"}
                  >
                    {completed ? 'Review Module' : !unlocked ? 'Locked' : 'Start Module'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
};