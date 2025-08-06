import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { ScrollArea } from './ui/scroll-area';
import { Separator } from './ui/separator';
import { Alert, AlertDescription } from './ui/alert';
import { 
  BookOpen,
  Search,
  Filter,
  Star,
  ExternalLink,
  Download,
  Calendar,
  Users,
  Tag,
  Bookmark,
  Eye,
  Quote,
  TrendingUp,
  Award,
  Clock,
  Heart,
  Share,
  ChevronDown,
  ChevronUp,
  FileText,
  Globe
} from 'lucide-react';
import { API_BASE_URL } from '../config/api';

interface ResearchPaper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  venue: string;
  year: number;
  doi?: string;
  url?: string;
  pdf_url?: string;
  keywords: string[];
  citation_count: number;
  relevance_score: number;
  topics: string[];
  difficulty_level: 'beginner' | 'intermediate' | 'advanced';
  reading_time: number; // estimated minutes
  is_bookmarked: boolean;
  is_read: boolean;
  user_rating?: number;
  notes?: string;
}

interface CitationEntry {
  paper_id: string;
  citation_text: string;
  context: string;
  date_added: string;
}

interface ReadingList {
  id: string;
  name: string;
  description: string;
  papers: string[];
  created_at: string;
  is_public: boolean;
  tags: string[];
}

interface FilterOptions {
  query: string;
  topics: string[];
  venues: string[];
  yearRange: [number, number];
  difficultyLevels: string[];
  onlyBookmarked: boolean;
  onlyUnread: boolean;
  sortBy: 'relevance' | 'citations' | 'year' | 'title';
  sortOrder: 'asc' | 'desc';
}

interface ResearchPapersProps {
  currentTopic?: string;
  onPaperSelect?: (paper: ResearchPaper) => void;
  onCitationAdd?: (citation: CitationEntry) => void;
}

export const ResearchPapers: React.FC<ResearchPapersProps> = ({
  currentTopic,
  onPaperSelect,
  onCitationAdd
}) => {
  const [activeTab, setActiveTab] = useState<'browse' | 'reading-lists' | 'citations' | 'recommendations'>('browse');
  const [papers, setPapers] = useState<ResearchPaper[]>([]);
  const [filteredPapers, setFilteredPapers] = useState<ResearchPaper[]>([]);
  const [readingLists, setReadingLists] = useState<ReadingList[]>([]);
  const [citations, setCitations] = useState<CitationEntry[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<ResearchPaper | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showFilters, setShowFilters] = useState(false);

  const [filters, setFilters] = useState<FilterOptions>({
    query: currentTopic || '',
    topics: [],
    venues: [],
    yearRange: [2010, new Date().getFullYear()],
    difficultyLevels: [],
    onlyBookmarked: false,
    onlyUnread: false,
    sortBy: 'relevance',
    sortOrder: 'desc'
  });

  const [availableTopics, setAvailableTopics] = useState<string[]>([]);
  const [availableVenues, setAvailableVenues] = useState<string[]>([]);

  useEffect(() => {
    loadPapers();
    loadReadingLists();
    loadCitations();
  }, []);

  useEffect(() => {
    if (currentTopic && currentTopic !== filters.query) {
      setFilters(prev => ({ ...prev, query: currentTopic }));
    }
  }, [currentTopic]);

  useEffect(() => {
    applyFilters();
  }, [papers, filters]);

  const loadPapers = async () => {
    setIsLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/research-papers`);
      const data = await response.json();
      
      setPapers(data.papers || []);
      setAvailableTopics(data.topics || []);
      setAvailableVenues(data.venues || []);
    } catch (error) {
      console.error('Failed to load papers:', error);
      // Set fallback data
      setPapers([
        {
          id: '1',
          title: 'A Tutorial on Regular Languages and Finite Automata',
          authors: ['John E. Hopcroft', 'Rajeev Motwani', 'Jeffrey D. Ullman'],
          abstract: 'This paper provides a comprehensive introduction to the theory of regular languages and finite automata, covering fundamental concepts and applications in computer science.',
          venue: 'ACM Computing Surveys',
          year: 2020,
          doi: '10.1145/3123456',
          keywords: ['finite automata', 'regular languages', 'theoretical computer science'],
          citation_count: 245,
          relevance_score: 95,
          topics: ['Finite Automata', 'Regular Languages'],
          difficulty_level: 'beginner',
          reading_time: 45,
          is_bookmarked: false,
          is_read: false
        },
        {
          id: '2',
          title: 'Advanced Techniques in Context-Free Grammar Parsing',
          authors: ['Donald E. Knuth', 'Robert W. Floyd'],
          abstract: 'An exploration of efficient parsing algorithms for context-free grammars, including LR parsing and error recovery techniques.',
          venue: 'Journal of the ACM',
          year: 2019,
          citation_count: 156,
          relevance_score: 88,
          topics: ['Context-Free Grammars', 'Parsing'],
          difficulty_level: 'advanced',
          reading_time: 65,
          is_bookmarked: true,
          is_read: false
        }
      ]);
      setAvailableTopics(['Finite Automata', 'Regular Languages', 'Context-Free Grammars', 'Parsing', 'Complexity Theory']);
      setAvailableVenues(['ACM Computing Surveys', 'Journal of the ACM', 'IEEE Computer', 'Communications of the ACM']);
    } finally {
      setIsLoading(false);
    }
  };

  const loadReadingLists = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/reading-lists`);
      const data = await response.json();
      setReadingLists(data || []);
    } catch (error) {
      console.error('Failed to load reading lists:', error);
      setReadingLists([
        {
          id: '1',
          name: 'Automata Theory Fundamentals',
          description: 'Essential papers for understanding automata theory',
          papers: ['1'],
          created_at: '2024-01-01',
          is_public: false,
          tags: ['fundamentals', 'theory']
        }
      ]);
    }
  };

  const loadCitations = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/citations`);
      const data = await response.json();
      setCitations(data || []);
    } catch (error) {
      console.error('Failed to load citations:', error);
      setCitations([]);
    }
  };

  const applyFilters = () => {
    let filtered = [...papers];

    // Text search
    if (filters.query) {
      const query = filters.query.toLowerCase();
      filtered = filtered.filter(paper =>
        paper.title.toLowerCase().includes(query) ||
        paper.abstract.toLowerCase().includes(query) ||
        paper.authors.some(author => author.toLowerCase().includes(query)) ||
        paper.keywords.some(keyword => keyword.toLowerCase().includes(query))
      );
    }

    // Topic filter
    if (filters.topics.length > 0) {
      filtered = filtered.filter(paper =>
        paper.topics.some(topic => filters.topics.includes(topic))
      );
    }

    // Venue filter
    if (filters.venues.length > 0) {
      filtered = filtered.filter(paper =>
        filters.venues.includes(paper.venue)
      );
    }

    // Year range filter
    filtered = filtered.filter(paper =>
      paper.year >= filters.yearRange[0] && paper.year <= filters.yearRange[1]
    );

    // Difficulty level filter
    if (filters.difficultyLevels.length > 0) {
      filtered = filtered.filter(paper =>
        filters.difficultyLevels.includes(paper.difficulty_level)
      );
    }

    // Bookmarked filter
    if (filters.onlyBookmarked) {
      filtered = filtered.filter(paper => paper.is_bookmarked);
    }

    // Unread filter
    if (filters.onlyUnread) {
      filtered = filtered.filter(paper => !paper.is_read);
    }

    // Sort
    filtered.sort((a, b) => {
      let comparison = 0;
      
      switch (filters.sortBy) {
        case 'relevance':
          comparison = b.relevance_score - a.relevance_score;
          break;
        case 'citations':
          comparison = b.citation_count - a.citation_count;
          break;
        case 'year':
          comparison = b.year - a.year;
          break;
        case 'title':
          comparison = a.title.localeCompare(b.title);
          break;
      }

      return filters.sortOrder === 'desc' ? comparison : -comparison;
    });

    setFilteredPapers(filtered);
  };

  const toggleBookmark = (paperId: string) => {
    setPapers(prev => prev.map(paper =>
      paper.id === paperId
        ? { ...paper, is_bookmarked: !paper.is_bookmarked }
        : paper
    ));
  };

  const markAsRead = (paperId: string) => {
    setPapers(prev => prev.map(paper =>
      paper.id === paperId
        ? { ...paper, is_read: true }
        : paper
    ));
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800 border-green-200';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'advanced': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const renderPaperCard = (paper: ResearchPaper, isCompact: boolean = false) => (
    <Card 
      key={paper.id}
      className={`hover:shadow-md transition-all cursor-pointer ${paper.is_read ? 'opacity-75' : ''}`}
      onClick={() => {
        setSelectedPaper(paper);
        onPaperSelect?.(paper);
      }}
    >
      <CardContent className="p-4">
        <div className="space-y-3">
          <div className="flex items-start justify-between">
            <div className="flex-1 pr-4">
              <h3 className={`font-medium leading-tight ${isCompact ? 'text-sm' : 'text-base'}`}>
                {paper.title}
              </h3>
              <p className="text-sm text-gray-600 mt-1">
                {paper.authors.slice(0, 3).join(', ')}
                {paper.authors.length > 3 && ` et al.`}
              </p>
            </div>
            
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="ghost"
                onClick={(e) => {
                  e.stopPropagation();
                  toggleBookmark(paper.id);
                }}
              >
                <Bookmark 
                  className={`w-4 h-4 ${paper.is_bookmarked ? 'fill-current text-blue-600' : ''}`} 
                />
              </Button>
              {paper.is_read && (
                <Eye className="w-4 h-4 text-green-600" />
              )}
            </div>
          </div>

          {!isCompact && (
            <p className="text-sm text-gray-700 line-clamp-2">
              {paper.abstract}
            </p>
          )}

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 text-xs text-gray-600">
              <span className="flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                {paper.year}
              </span>
              <span className="flex items-center gap-1">
                <Quote className="w-3 h-3" />
                {paper.citation_count}
              </span>
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {paper.reading_time}min
              </span>
            </div>
            
            <div className="flex items-center gap-2">
              <Badge className={`text-xs border ${getDifficultyColor(paper.difficulty_level)}`}>
                {paper.difficulty_level}
              </Badge>
              <Badge variant="outline" className="text-xs">
                {Math.round(paper.relevance_score)}%
              </Badge>
            </div>
          </div>

          <div className="flex flex-wrap gap-1">
            {paper.topics.slice(0, 3).map((topic, index) => (
              <Badge key={index} variant="outline" className="text-xs">
                {topic}
              </Badge>
            ))}
            {paper.topics.length > 3 && (
              <Badge variant="outline" className="text-xs">
                +{paper.topics.length - 3}
              </Badge>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const renderFilterPanel = () => (
    <Card className="mb-4">
      <CardContent className="p-4">
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Topics</label>
              <Select 
                value={filters.topics[0] || ''} 
                onValueChange={(value) => 
                  setFilters(prev => ({ 
                    ...prev, 
                    topics: value ? [value] : [] 
                  }))
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="All topics" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">All topics</SelectItem>
                  {availableTopics.map((topic) => (
                    <SelectItem key={topic} value={topic}>{topic}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Venue</label>
              <Select 
                value={filters.venues[0] || ''} 
                onValueChange={(value) => 
                  setFilters(prev => ({ 
                    ...prev, 
                    venues: value ? [value] : [] 
                  }))
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="All venues" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">All venues</SelectItem>
                  {availableVenues.map((venue) => (
                    <SelectItem key={venue} value={venue}>{venue}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Difficulty</label>
              <Select 
                value={filters.difficultyLevels[0] || ''} 
                onValueChange={(value) => 
                  setFilters(prev => ({ 
                    ...prev, 
                    difficultyLevels: value ? [value] : [] 
                  }))
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="All levels" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">All levels</SelectItem>
                  <SelectItem value="beginner">Beginner</SelectItem>
                  <SelectItem value="intermediate">Intermediate</SelectItem>
                  <SelectItem value="advanced">Advanced</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Sort by</label>
              <Select 
                value={filters.sortBy} 
                onValueChange={(value: any) => 
                  setFilters(prev => ({ ...prev, sortBy: value }))
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="relevance">Relevance</SelectItem>
                  <SelectItem value="citations">Citations</SelectItem>
                  <SelectItem value="year">Year</SelectItem>
                  <SelectItem value="title">Title</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex flex-wrap gap-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={filters.onlyBookmarked}
                onChange={(e) => 
                  setFilters(prev => ({ ...prev, onlyBookmarked: e.target.checked }))
                }
              />
              Only bookmarked
            </label>
            
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={filters.onlyUnread}
                onChange={(e) => 
                  setFilters(prev => ({ ...prev, onlyUnread: e.target.checked }))
                }
              />
              Only unread
            </label>
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const renderBrowseTab = () => (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <Input
            placeholder="Search papers, authors, or keywords..."
            value={filters.query}
            onChange={(e) => setFilters(prev => ({ ...prev, query: e.target.value }))}
            className="pl-10"
          />
        </div>
        
        <Button
          variant="outline"
          onClick={() => setShowFilters(!showFilters)}
        >
          <Filter className="w-4 h-4 mr-2" />
          Filters
          {showFilters ? <ChevronUp className="w-4 h-4 ml-2" /> : <ChevronDown className="w-4 h-4 ml-2" />}
        </Button>
      </div>

      {showFilters && renderFilterPanel()}

      <div className="flex items-center justify-between text-sm text-gray-600">
        <span>{filteredPapers.length} papers found</span>
        <div className="flex items-center gap-2">
          <span>Showing</span>
          <Select value="20" onValueChange={() => {}}>
            <SelectTrigger className="w-16">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="10">10</SelectItem>
              <SelectItem value="20">20</SelectItem>
              <SelectItem value="50">50</SelectItem>
            </SelectContent>
          </Select>
          <span>per page</span>
        </div>
      </div>

      <ScrollArea className="h-96">
        <div className="space-y-4">
          {filteredPapers.map(paper => renderPaperCard(paper))}
        </div>
      </ScrollArea>
    </div>
  );

  const renderReadingListsTab = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">My Reading Lists</h3>
        <Button size="sm">
          <BookOpen className="w-4 h-4 mr-2" />
          New List
        </Button>
      </div>

      <div className="grid gap-4">
        {readingLists.map((list) => (
          <Card key={list.id} className="hover:shadow-md transition-shadow cursor-pointer">
            <CardContent className="p-4">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h4 className="font-medium">{list.name}</h4>
                  <p className="text-sm text-gray-600 mt-1">{list.description}</p>
                </div>
                <Badge variant="outline" className="text-xs">
                  {list.papers.length} papers
                </Badge>
              </div>
              
              <div className="flex items-center justify-between mt-3">
                <div className="flex flex-wrap gap-1">
                  {list.tags.map((tag, index) => (
                    <Badge key={index} variant="outline" className="text-xs">
                      <Tag className="w-3 h-3 mr-1" />
                      {tag}
                    </Badge>
                  ))}
                </div>
                
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Calendar className="w-3 h-3" />
                  {new Date(list.created_at).toLocaleDateString()}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  const renderCitationsTab = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">Citation Manager</h3>
        <div className="flex gap-2">
          <Button size="sm" variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Export BibTeX
          </Button>
          <Button size="sm">
            <Quote className="w-4 h-4 mr-2" />
            Add Citation
          </Button>
        </div>
      </div>

      <ScrollArea className="h-96">
        <div className="space-y-3">
          {citations.map((citation, index) => (
            <Card key={index}>
              <CardContent className="p-4">
                <div className="space-y-2">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="text-sm font-mono bg-gray-50 p-2 rounded">
                        {citation.citation_text}
                      </p>
                    </div>
                    <Button size="sm" variant="ghost">
                      <Share className="w-4 h-4" />
                    </Button>
                  </div>
                  
                  <div className="text-xs text-gray-600">
                    <p><strong>Context:</strong> {citation.context}</p>
                    <p><strong>Added:</strong> {new Date(citation.date_added).toLocaleDateString()}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>
    </div>
  );

  const renderRecommendationsTab = () => (
    <div className="space-y-4">
      <div className="text-center">
        <h3 className="text-lg font-medium mb-2">Personalized Recommendations</h3>
        <p className="text-sm text-gray-600">
          Papers curated based on your interests and reading history
        </p>
      </div>

      <div className="grid gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Trending in Your Field
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {filteredPapers.slice(0, 3).map(paper => renderPaperCard(paper, true))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Award className="w-4 h-4" />
              Highly Cited Recent Papers
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {filteredPapers
                .filter(paper => paper.year >= 2020)
                .sort((a, b) => b.citation_count - a.citation_count)
                .slice(0, 3)
                .map(paper => renderPaperCard(paper, true))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Heart className="w-4 h-4" />
              Similar to Your Bookmarks
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {filteredPapers
                .filter(paper => !paper.is_bookmarked)
                .slice(0, 3)
                .map(paper => renderPaperCard(paper, true))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );

  if (isLoading) {
    return (
      <Card className="h-full">
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-sm text-gray-500">Loading research papers...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-blue-600" />
          Research Papers
        </CardTitle>
      </CardHeader>
      
      <CardContent>
        <Tabs value={activeTab} onValueChange={(value: any) => setActiveTab(value)}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="browse">Browse</TabsTrigger>
            <TabsTrigger value="reading-lists">Reading Lists</TabsTrigger>
            <TabsTrigger value="citations">Citations</TabsTrigger>
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          </TabsList>

          <TabsContent value="browse" className="mt-6">
            {renderBrowseTab()}
          </TabsContent>

          <TabsContent value="reading-lists" className="mt-6">
            {renderReadingListsTab()}
          </TabsContent>

          <TabsContent value="citations" className="mt-6">
            {renderCitationsTab()}
          </TabsContent>

          <TabsContent value="recommendations" className="mt-6">
            {renderRecommendationsTab()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};