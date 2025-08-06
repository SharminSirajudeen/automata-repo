import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Separator } from './ui/separator';
import { ScrollArea } from './ui/scroll-area';
import { Alert, AlertDescription } from './ui/alert';
import { 
  TrendingUp,
  Target,
  Clock,
  Award,
  BookOpen,
  Brain,
  ArrowRight,
  ArrowUp,
  ArrowDown,
  Eye,
  Calendar,
  BarChart3,
  PieChart,
  Activity,
  Lightbulb,
  CheckCircle,
  AlertTriangle,
  Star,
  Zap
} from 'lucide-react';
import { Problem, AutomataType } from '../types/automata';
import { API_BASE_URL } from '../config/api';

interface LearningMetrics {
  total_problems_attempted: number;
  problems_solved: number;
  success_rate: number;
  average_attempts_per_problem: number;
  time_spent_learning: number; // in minutes
  topics_mastered: string[];
  current_streak: number;
  longest_streak: number;
  difficulty_progression: number; // 1-10 scale
}

interface PerformanceData {
  topic: string;
  problems_attempted: number;
  success_rate: number;
  average_time: number;
  difficulty_level: number;
  last_attempt: string;
  trend: 'improving' | 'stable' | 'declining';
}

interface Recommendation {
  type: 'problem' | 'topic' | 'review' | 'challenge';
  title: string;
  description: string;
  reasoning: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimated_time: number;
  priority: number;
  problem_id?: string;
  topic?: string;
}

interface LearningPath {
  name: string;
  description: string;
  total_problems: number;
  completed_problems: number;
  estimated_completion_time: number;
  topics: string[];
  next_milestone: string;
  progress_percentage: number;
}

interface AdaptiveLearningProps {
  currentProblem?: Problem;
  onRecommendationSelect?: (recommendation: Recommendation) => void;
  onPathSelect?: (path: LearningPath) => void;
}

export const AdaptiveLearning: React.FC<AdaptiveLearningProps> = ({
  currentProblem,
  onRecommendationSelect,
  onPathSelect
}) => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'recommendations' | 'paths' | 'analytics'>('dashboard');
  const [metrics, setMetrics] = useState<LearningMetrics | null>(null);
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [learningPaths, setLearningPaths] = useState<LearningPath[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'week' | 'month' | 'all'>('week');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadLearningData();
  }, [selectedTimeRange]);

  const loadLearningData = async () => {
    setIsLoading(true);
    
    try {
      const [metricsRes, performanceRes, recommendationsRes, pathsRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/learning-metrics?timeRange=${selectedTimeRange}`),
        fetch(`${API_BASE_URL}/api/performance-data?timeRange=${selectedTimeRange}`),
        fetch(`${API_BASE_URL}/api/learning-recommendations`),
        fetch(`${API_BASE_URL}/api/learning-paths`)
      ]);

      const [metricsData, performanceDataRes, recommendationsData, pathsData] = await Promise.all([
        metricsRes.json(),
        performanceRes.json(),
        recommendationsRes.json(),
        pathsRes.json()
      ]);

      setMetrics(metricsData);
      setPerformanceData(performanceDataRes);
      setRecommendations(recommendationsData);
      setLearningPaths(pathsData);
    } catch (error) {
      console.error('Failed to load learning data:', error);
      // Set fallback data
      setMetrics({
        total_problems_attempted: 15,
        problems_solved: 12,
        success_rate: 80,
        average_attempts_per_problem: 2.3,
        time_spent_learning: 180,
        topics_mastered: ['DFA', 'NFA', 'Regular Languages'],
        current_streak: 5,
        longest_streak: 8,
        difficulty_progression: 6.5
      });
      setPerformanceData([
        {
          topic: 'DFA Construction',
          problems_attempted: 8,
          success_rate: 87.5,
          average_time: 12,
          difficulty_level: 3,
          last_attempt: '2024-01-15',
          trend: 'improving'
        },
        {
          topic: 'NFA to DFA Conversion',
          problems_attempted: 5,
          success_rate: 60,
          average_time: 18,
          difficulty_level: 5,
          last_attempt: '2024-01-14',
          trend: 'stable'
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const getDifficultyColor = (difficulty: string | number) => {
    if (typeof difficulty === 'string') {
      switch (difficulty) {
        case 'beginner': return 'bg-green-100 text-green-800 border-green-200';
        case 'intermediate': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
        case 'advanced': return 'bg-red-100 text-red-800 border-red-200';
        default: return 'bg-gray-100 text-gray-800 border-gray-200';
      }
    } else {
      if (difficulty <= 3) return 'bg-green-100 text-green-800 border-green-200';
      if (difficulty <= 6) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      return 'bg-red-100 text-red-800 border-red-200';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return <ArrowUp className="w-4 h-4 text-green-600" />;
      case 'declining': return <ArrowDown className="w-4 h-4 text-red-600" />;
      default: return <ArrowRight className="w-4 h-4 text-gray-600" />;
    }
  };

  const renderDashboard = () => (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Success Rate</p>
                <p className="text-2xl font-bold text-green-600">
                  {metrics?.success_rate}%
                </p>
              </div>
              <Target className="w-8 h-8 text-green-600 opacity-60" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Problems Solved</p>
                <p className="text-2xl font-bold text-blue-600">
                  {metrics?.problems_solved}
                </p>
              </div>
              <CheckCircle className="w-8 h-8 text-blue-600 opacity-60" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Current Streak</p>
                <p className="text-2xl font-bold text-orange-600">
                  {metrics?.current_streak}
                </p>
              </div>
              <Zap className="w-8 h-8 text-orange-600 opacity-60" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Time Spent</p>
                <p className="text-2xl font-bold text-purple-600">
                  {Math.round((metrics?.time_spent_learning || 0) / 60)}h
                </p>
              </div>
              <Clock className="w-8 h-8 text-purple-600 opacity-60" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Progress Overview */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Learning Progress
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span>Overall Difficulty Progression</span>
              <span className="font-mono">{metrics?.difficulty_progression}/10</span>
            </div>
            <Progress value={(metrics?.difficulty_progression || 0) * 10} className="h-2" />
          </div>

          <Separator />

          <div className="space-y-2">
            <h4 className="text-sm font-medium">Topics Mastered</h4>
            <div className="flex flex-wrap gap-2">
              {metrics?.topics_mastered.map((topic, index) => (
                <Badge key={index} className="bg-green-100 text-green-800 border-green-200">
                  <Award className="w-3 h-3 mr-1" />
                  {topic}
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance by Topic */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Performance by Topic
            </div>
            <Select value={selectedTimeRange} onValueChange={(value: any) => setSelectedTimeRange(value)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="week">This Week</SelectItem>
                <SelectItem value="month">This Month</SelectItem>
                <SelectItem value="all">All Time</SelectItem>
              </SelectContent>
            </Select>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-64">
            <div className="space-y-3">
              {performanceData.map((data, index) => (
                <div key={index} className="p-3 border rounded-lg hover:bg-gray-50 transition-colors">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-medium">{data.topic}</h4>
                      {getTrendIcon(data.trend)}
                    </div>
                    <Badge className={`text-xs border ${getDifficultyColor(data.difficulty_level)}`}>
                      Level {data.difficulty_level}
                    </Badge>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 text-xs text-gray-600">
                    <div>
                      <span className="block">Success Rate</span>
                      <span className="font-medium text-sm">{data.success_rate}%</span>
                    </div>
                    <div>
                      <span className="block">Avg Time</span>
                      <span className="font-medium text-sm">{data.average_time}min</span>
                    </div>
                    <div>
                      <span className="block">Problems</span>
                      <span className="font-medium text-sm">{data.problems_attempted}</span>
                    </div>
                  </div>

                  <div className="mt-2">
                    <Progress value={data.success_rate} className="h-1" />
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );

  const renderRecommendations = () => (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-lg font-medium mb-2">Personalized Recommendations</h3>
        <p className="text-sm text-gray-600">
          AI-curated suggestions based on your learning progress and performance
        </p>
      </div>

      {recommendations.length === 0 ? (
        <Card>
          <CardContent className="p-8 text-center">
            <Brain className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p className="text-sm text-gray-500">
              Complete a few more problems to get personalized recommendations
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {recommendations.map((rec, index) => (
            <Card 
              key={index}
              className="hover:shadow-md transition-shadow cursor-pointer"
              onClick={() => onRecommendationSelect?.(rec)}
            >
              <CardContent className="p-4">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    {rec.type === 'problem' && <Target className="w-4 h-4 text-blue-600" />}
                    {rec.type === 'topic' && <BookOpen className="w-4 h-4 text-green-600" />}
                    {rec.type === 'review' && <Eye className="w-4 h-4 text-orange-600" />}
                    {rec.type === 'challenge' && <Star className="w-4 h-4 text-purple-600" />}
                    
                    <h4 className="font-medium">{rec.title}</h4>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Badge className={`text-xs border ${getDifficultyColor(rec.difficulty)}`}>
                      {rec.difficulty}
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      {rec.estimated_time}min
                    </Badge>
                  </div>
                </div>

                <p className="text-sm text-gray-700 mb-2">{rec.description}</p>
                
                <div className="text-xs text-gray-600 bg-blue-50 p-2 rounded border-l-2 border-blue-200">
                  <Lightbulb className="w-3 h-3 inline mr-1" />
                  <strong>Why this helps:</strong> {rec.reasoning}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );

  const renderLearningPaths = () => (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-lg font-medium mb-2">Learning Paths</h3>
        <p className="text-sm text-gray-600">
          Structured learning journeys tailored to your goals
        </p>
      </div>

      <div className="grid gap-4">
        {learningPaths.map((path, index) => (
          <Card 
            key={index}
            className="hover:shadow-md transition-shadow cursor-pointer"
            onClick={() => onPathSelect?.(path)}
          >
            <CardContent className="p-4">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">{path.name}</h4>
                  <Badge variant="outline" className="text-xs">
                    {path.completed_problems}/{path.total_problems} problems
                  </Badge>
                </div>

                <p className="text-sm text-gray-700">{path.description}</p>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span>Progress</span>
                    <span>{Math.round(path.progress_percentage)}%</span>
                  </div>
                  <Progress value={path.progress_percentage} className="h-2" />
                </div>

                <div className="flex items-center justify-between text-xs text-gray-600">
                  <div className="flex items-center gap-4">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {Math.round(path.estimated_completion_time / 60)}h remaining
                    </span>
                    <span className="flex items-center gap-1">
                      <BookOpen className="w-3 h-3" />
                      {path.topics.length} topics
                    </span>
                  </div>
                </div>

                <div className="flex flex-wrap gap-1">
                  {path.topics.slice(0, 3).map((topic, topicIndex) => (
                    <Badge key={topicIndex} variant="outline" className="text-xs">
                      {topic}
                    </Badge>
                  ))}
                  {path.topics.length > 3 && (
                    <Badge variant="outline" className="text-xs">
                      +{path.topics.length - 3} more
                    </Badge>
                  )}
                </div>

                {path.next_milestone && (
                  <Alert>
                    <Target className="h-4 w-4" />
                    <AlertDescription className="text-xs">
                      <strong>Next milestone:</strong> {path.next_milestone}
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  const renderAnalytics = () => (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-lg font-medium mb-2">Learning Analytics</h3>
        <p className="text-sm text-gray-600">
          Deep insights into your learning patterns and progress
        </p>
      </div>

      {/* Learning Patterns */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Activity className="w-4 h-4" />
            Learning Patterns
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="text-sm font-medium">Peak Performance Hours</h4>
              <div className="text-xs text-gray-600 space-y-1">
                <div className="flex justify-between">
                  <span>Morning (8-12 PM)</span>
                  <span className="font-medium">85% success rate</span>
                </div>
                <div className="flex justify-between">
                  <span>Afternoon (12-6 PM)</span>
                  <span className="font-medium">78% success rate</span>
                </div>
                <div className="flex justify-between">
                  <span>Evening (6-10 PM)</span>
                  <span className="font-medium">72% success rate</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-medium">Learning Velocity</h4>
              <div className="text-xs text-gray-600 space-y-1">
                <div className="flex justify-between">
                  <span>Problems/Hour</span>
                  <span className="font-medium">3.2</span>
                </div>
                <div className="flex justify-between">
                  <span>Avg Time/Problem</span>
                  <span className="font-medium">18.5 min</span>
                </div>
                <div className="flex justify-between">
                  <span>Retention Rate</span>
                  <span className="font-medium">89%</span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Strengths and Weaknesses */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2 text-green-600">
              <CheckCircle className="w-4 h-4" />
              Strengths
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span>DFA Construction</span>
                <Badge className="bg-green-100 text-green-800 text-xs">92%</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span>Regular Expressions</span>
                <Badge className="bg-green-100 text-green-800 text-xs">88%</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span>State Minimization</span>
                <Badge className="bg-green-100 text-green-800 text-xs">85%</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2 text-orange-600">
              <AlertTriangle className="w-4 h-4" />
              Areas for Improvement
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span>Context-Free Grammars</span>
                <Badge className="bg-orange-100 text-orange-800 text-xs">45%</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span>Pumping Lemma</span>
                <Badge className="bg-orange-100 text-orange-800 text-xs">52%</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span>NFA to DFA Conversion</span>
                <Badge className="bg-orange-100 text-orange-800 text-xs">58%</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Goal Tracking */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Target className="w-4 h-4" />
            Goal Tracking
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div>
              <div className="flex items-center justify-between text-sm mb-1">
                <span>Weekly Problem Goal</span>
                <span>8/10 problems</span>
              </div>
              <Progress value={80} className="h-2" />
            </div>

            <div>
              <div className="flex items-center justify-between text-sm mb-1">
                <span>Monthly Mastery Goal</span>
                <span>3/5 topics</span>
              </div>
              <Progress value={60} className="h-2" />
            </div>

            <div>
              <div className="flex items-center justify-between text-sm mb-1">
                <span>Streak Goal</span>
                <span>5/7 days</span>
              </div>
              <Progress value={71} className="h-2" />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  if (isLoading) {
    return (
      <Card className="h-full">
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-sm text-gray-500">Loading learning data...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-blue-600" />
          Adaptive Learning
        </CardTitle>
      </CardHeader>
      
      <CardContent>
        <Tabs value={activeTab} onValueChange={(value: any) => setActiveTab(value)}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
            <TabsTrigger value="paths">Learning Paths</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
          </TabsList>

          <TabsContent value="dashboard" className="mt-6">
            {renderDashboard()}
          </TabsContent>

          <TabsContent value="recommendations" className="mt-6">
            {renderRecommendations()}
          </TabsContent>

          <TabsContent value="paths" className="mt-6">
            {renderLearningPaths()}
          </TabsContent>

          <TabsContent value="analytics" className="mt-6">
            {renderAnalytics()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};