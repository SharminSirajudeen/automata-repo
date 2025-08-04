import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Search, BookOpen, Brain, Zap } from 'lucide-react';
import { Problem } from '../types/automata';
import { apiService } from '../services/api';

interface ProblemSelectorProps {
  onSelectProblem: (problem: Problem) => void;
}

export const ProblemSelector: React.FC<ProblemSelectorProps> = ({ onSelectProblem }) => {
  const [problems, setProblems] = useState<Problem[]>([]);
  const [filteredProblems, setFilteredProblems] = useState<Problem[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadProblems();
  }, []);

  useEffect(() => {
    filterProblems();
  }, [problems, searchTerm, selectedType]);

  const loadProblems = async () => {
    try {
      const response = await apiService.getProblems();
      setProblems(response.problems);
    } catch (error) {
      console.error('Failed to load problems:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const filterProblems = () => {
    let filtered = problems;

    if (selectedType !== 'all') {
      filtered = filtered.filter(problem => problem.type === selectedType);
    }

    if (searchTerm) {
      filtered = filtered.filter(problem =>
        problem.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        problem.description.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    setFilteredProblems(filtered);
  };

  const problemTypes = ['all', ...Array.from(new Set(problems.map(p => p.type)))];

  const getDifficultyLevel = (problem: Problem) => {
    const testCaseCount = problem.test_strings.length;
    const alphabetSize = problem.alphabet.length;
    
    if (testCaseCount <= 5 && alphabetSize <= 2) return 'Easy';
    if (testCaseCount <= 10 && alphabetSize <= 3) return 'Medium';
    return 'Hard';
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Easy': return 'bg-green-100 text-green-800';
      case 'Medium': return 'bg-yellow-100 text-yellow-800';
      case 'Hard': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (isLoading) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading problems...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-gray-900">
          Theory of Computation Tutor
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Master automata theory with interactive practice problems and AI-powered guidance.
          Build DFAs, NFAs, and more with step-by-step feedback from our intelligent tutor.
        </p>
      </div>

      <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <Input
            placeholder="Search problems..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        
        <div className="flex gap-2">
          {problemTypes.map(type => (
            <Button
              key={type}
              onClick={() => setSelectedType(type)}
              variant={selectedType === type ? "default" : "outline"}
              size="sm"
              className="capitalize"
            >
              {type === 'all' ? 'All Types' : type.toUpperCase()}
            </Button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredProblems.map(problem => {
          const difficulty = getDifficultyLevel(problem);
          return (
            <Card key={problem.id} className="hover:shadow-lg transition-shadow cursor-pointer group">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="space-y-2">
                    <CardTitle className="text-lg group-hover:text-blue-600 transition-colors">
                      {problem.title}
                    </CardTitle>
                    <div className="flex gap-2">
                      <Badge variant="secondary" className="uppercase">
                        {problem.type}
                      </Badge>
                      <Badge className={getDifficultyColor(difficulty)}>
                        {difficulty}
                      </Badge>
                    </div>
                  </div>
                  <div className="text-right text-sm text-gray-500">
                    <div className="flex items-center gap-1">
                      <BookOpen className="w-4 h-4" />
                      {problem.test_strings.length} tests
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-gray-600 text-sm line-clamp-3">
                  {problem.description}
                </p>
                
                <div className="space-y-2">
                  <div className="text-sm">
                    <span className="font-medium">Language: </span>
                    <span className="text-gray-600">{problem.language_description}</span>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">Alphabet:</span>
                    <div className="flex gap-1">
                      {problem.alphabet.map(symbol => (
                        <Badge key={symbol} variant="outline" className="text-xs">
                          {symbol}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-2">
                  <div className="flex items-center gap-4 text-xs text-gray-500">
                    <div className="flex items-center gap-1">
                      <Brain className="w-3 h-3" />
                      AI Guided
                    </div>
                    <div className="flex items-center gap-1">
                      <Zap className="w-3 h-3" />
                      Interactive
                    </div>
                  </div>
                  
                  <Button
                    onClick={() => onSelectProblem(problem)}
                    size="sm"
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    Start Problem
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {filteredProblems.length === 0 && (
        <div className="text-center py-12">
          <div className="text-gray-400 mb-4">
            <Search className="w-12 h-12 mx-auto" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No problems found</h3>
          <p className="text-gray-600">
            Try adjusting your search terms or filters to find problems.
          </p>
        </div>
      )}
    </div>
  );
};
