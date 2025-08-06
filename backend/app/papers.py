"""
Research Papers Integration Module for Automata Theory
Provides paper metadata management, context-aware recommendations, and citation formatting.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import json
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class PaperCategory(str, Enum):
    """Categories of research papers"""
    TEXTBOOK = "textbook"
    SURVEY = "survey"
    RESEARCH_PAPER = "research_paper"
    CONFERENCE_PAPER = "conference_paper"
    JOURNAL_ARTICLE = "journal_article"
    TUTORIAL = "tutorial"
    THESIS = "thesis"

class PaperDifficulty(str, Enum):
    """Difficulty levels for papers"""
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    ADVANCED_GRADUATE = "advanced_graduate"
    RESEARCH = "research"

class TopicArea(str, Enum):
    """Topic areas in automata theory"""
    FINITE_AUTOMATA = "finite_automata"
    REGULAR_LANGUAGES = "regular_languages"
    CONTEXT_FREE_LANGUAGES = "context_free_languages"
    PUSHDOWN_AUTOMATA = "pushdown_automata"
    TURING_MACHINES = "turing_machines"
    COMPUTABILITY = "computability"
    COMPLEXITY_THEORY = "complexity_theory"
    P_VS_NP = "p_vs_np"
    SPACE_COMPLEXITY = "space_complexity"
    RANDOMIZED_COMPUTATION = "randomized_computation"
    APPROXIMATION_ALGORITHMS = "approximation_algorithms"
    DESCRIPTIVE_COMPLEXITY = "descriptive_complexity"
    QUANTUM_COMPUTATION = "quantum_computation"
    FORMAL_VERIFICATION = "formal_verification"
    MODEL_CHECKING = "model_checking"

class CitationStyle(str, Enum):
    """Citation formatting styles"""
    APA = "apa"
    MLA = "mla"
    IEEE = "ieee"
    ACM = "acm"
    BIBTEX = "bibtex"

class Paper(BaseModel):
    """Research paper representation"""
    id: str
    title: str
    authors: List[str]
    year: int
    category: PaperCategory
    difficulty: PaperDifficulty
    topics: List[TopicArea]
    abstract: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    venue: Optional[str] = None  # Journal/Conference name
    pages: Optional[str] = None
    volume: Optional[str] = None
    number: Optional[str] = None
    publisher: Optional[str] = None
    isbn: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    citations_count: int = 0
    impact_score: float = Field(default=0.0, ge=0.0, le=10.0)
    is_seminal: bool = False  # Foundational/seminal papers
    prerequisites: List[str] = Field(default_factory=list)  # Required background
    
class PaperRecommendationRequest(BaseModel):
    """Request for paper recommendations"""
    student_level: PaperDifficulty
    topics_of_interest: List[TopicArea]
    current_paper_id: Optional[str] = None  # For related papers
    learning_objective: str = "general"  # general, research, implementation, theory
    max_recommendations: int = Field(default=5, ge=1, le=20)

class PaperSearchRequest(BaseModel):
    """Request for paper search"""
    query: str
    topics: Optional[List[TopicArea]] = None
    difficulty: Optional[PaperDifficulty] = None
    category: Optional[PaperCategory] = None
    year_range: Optional[Tuple[int, int]] = None
    max_results: int = Field(default=10, ge=1, le=50)

class CitationRequest(BaseModel):
    """Request for citation formatting"""
    paper_id: str
    style: CitationStyle
    include_url: bool = True

class PaperDatabase:
    """Database of research papers in automata theory"""
    
    def __init__(self):
        self.papers: Dict[str, Paper] = {}
        self.topic_index: Dict[TopicArea, Set[str]] = {}
        self.author_index: Dict[str, Set[str]] = {}
        self.keyword_index: Dict[str, Set[str]] = {}
        self._load_papers_database()
    
    def _load_papers_database(self):
        """Load papers from database file or create default database"""
        try:
            # Try to load from file first
            db_path = Path(__file__).parent / "papers_database.json"
            if db_path.exists():
                with open(db_path, 'r') as f:
                    data = json.load(f)
                    for paper_data in data.get('papers', []):
                        paper = Paper(**paper_data)
                        self.add_paper(paper)
            else:
                # Create default database
                self._create_default_database()
                self._save_database()
        except Exception as e:
            logger.error(f"Error loading papers database: {e}")
            self._create_default_database()
    
    def _create_default_database(self):
        """Create default database with seminal papers"""
        seminal_papers = [
            Paper(
                id="sipser_2012",
                title="Introduction to the Theory of Computation",
                authors=["Michael Sipser"],
                year=2012,
                category=PaperCategory.TEXTBOOK,
                difficulty=PaperDifficulty.UNDERGRADUATE,
                topics=[TopicArea.FINITE_AUTOMATA, TopicArea.REGULAR_LANGUAGES, 
                       TopicArea.CONTEXT_FREE_LANGUAGES, TopicArea.TURING_MACHINES,
                       TopicArea.COMPUTABILITY, TopicArea.COMPLEXITY_THEORY],
                abstract="Comprehensive introduction to theoretical computer science covering automata theory, computability, and complexity theory.",
                publisher="Cengage Learning",
                isbn="978-1133187790",
                keywords=["automata", "computability", "complexity", "theory of computation"],
                impact_score=9.5,
                is_seminal=True,
                prerequisites=["discrete_mathematics", "basic_algorithms"]
            ),
            Paper(
                id="hopcroft_ullman_1979",
                title="Introduction to Automata Theory, Languages, and Computation",
                authors=["John E. Hopcroft", "Jeffrey D. Ullman"],
                year=1979,
                category=PaperCategory.TEXTBOOK,
                difficulty=PaperDifficulty.UNDERGRADUATE,
                topics=[TopicArea.FINITE_AUTOMATA, TopicArea.REGULAR_LANGUAGES,
                       TopicArea.CONTEXT_FREE_LANGUAGES, TopicArea.PUSHDOWN_AUTOMATA],
                abstract="Classic textbook on automata theory and formal languages.",
                publisher="Addison-Wesley",
                keywords=["automata", "formal languages", "parsing"],
                impact_score=9.8,
                is_seminal=True,
                prerequisites=["discrete_mathematics"]
            ),
            Paper(
                id="cook_1971",
                title="The complexity of theorem-proving procedures",
                authors=["Stephen A. Cook"],
                year=1971,
                category=PaperCategory.CONFERENCE_PAPER,
                difficulty=PaperDifficulty.RESEARCH,
                topics=[TopicArea.COMPLEXITY_THEORY, TopicArea.P_VS_NP],
                abstract="Introduces NP-completeness and proves that satisfiability is NP-complete.",
                venue="STOC '71",
                pages="151-158",
                doi="10.1145/800157.805047",
                keywords=["NP-completeness", "satisfiability", "computational complexity"],
                citations_count=8000,
                impact_score=10.0,
                is_seminal=True,
                prerequisites=["complexity_theory", "logic"]
            ),
            Paper(
                id="karp_1972",
                title="Reducibility Among Combinatorial Problems",
                authors=["Richard M. Karp"],
                year=1972,
                category=PaperCategory.RESEARCH_PAPER,
                difficulty=PaperDifficulty.RESEARCH,
                topics=[TopicArea.COMPLEXITY_THEORY, TopicArea.P_VS_NP],
                abstract="Identifies 21 NP-complete problems and establishes the foundation of NP-completeness theory.",
                venue="Complexity of Computer Computations",
                pages="85-103",
                keywords=["NP-complete", "polynomial reductions", "combinatorial optimization"],
                citations_count=6000,
                impact_score=9.9,
                is_seminal=True,
                prerequisites=["cook_1971", "graph_theory"]
            ),
            Paper(
                id="turing_1936",
                title="On Computable Numbers, with an Application to the Entscheidungsproblem",
                authors=["Alan M. Turing"],
                year=1936,
                category=PaperCategory.JOURNAL_ARTICLE,
                difficulty=PaperDifficulty.RESEARCH,
                topics=[TopicArea.TURING_MACHINES, TopicArea.COMPUTABILITY],
                abstract="Introduces Turing machines and proves the undecidability of the halting problem.",
                venue="Proceedings of the London Mathematical Society",
                volume="42",
                number="2",
                pages="230-265",
                keywords=["Turing machines", "computability", "halting problem", "undecidability"],
                citations_count=15000,
                impact_score=10.0,
                is_seminal=True,
                prerequisites=["mathematical_logic"]
            ),
            Paper(
                id="rabin_scott_1959",
                title="Finite Automata and Their Decision Problems",
                authors=["Michael O. Rabin", "Dana Scott"],
                year=1959,
                category=PaperCategory.JOURNAL_ARTICLE,
                difficulty=PaperDifficulty.GRADUATE,
                topics=[TopicArea.FINITE_AUTOMATA, TopicArea.REGULAR_LANGUAGES],
                abstract="Establishes the theory of finite automata and proves fundamental decidability results.",
                venue="IBM Journal of Research and Development",
                volume="3",
                number="2",
                pages="114-125",
                keywords=["finite automata", "decidability", "regular languages"],
                citations_count=3000,
                impact_score=9.5,
                is_seminal=True,
                prerequisites=["formal_languages"]
            ),
            Paper(
                id="myhill_1957",
                title="Finite Automata and the Representation of Events",
                authors=["John Myhill"],
                year=1957,
                category=PaperCategory.RESEARCH_PAPER,
                difficulty=PaperDifficulty.GRADUATE,
                topics=[TopicArea.FINITE_AUTOMATA, TopicArea.REGULAR_LANGUAGES],
                abstract="Introduces the Myhill-Nerode theorem for characterizing regular languages.",
                venue="WADD Technical Report",
                keywords=["Myhill-Nerode theorem", "regular languages", "equivalence relations"],
                citations_count=1500,
                impact_score=8.5,
                is_seminal=True,
                prerequisites=["finite_automata", "equivalence_relations"]
            ),
            Paper(
                id="chomsky_1956",
                title="Three models for the description of language",
                authors=["Noam Chomsky"],
                year=1956,
                category=PaperCategory.JOURNAL_ARTICLE,
                difficulty=PaperDifficulty.GRADUATE,
                topics=[TopicArea.CONTEXT_FREE_LANGUAGES, TopicArea.FORMAL_VERIFICATION],
                abstract="Introduces the Chomsky hierarchy of formal grammars.",
                venue="IRE Transactions on Information Theory",
                volume="2",
                number="3",
                pages="113-124",
                keywords=["Chomsky hierarchy", "formal grammars", "context-free languages"],
                citations_count=4000,
                impact_score=9.7,
                is_seminal=True,
                prerequisites=["formal_languages"]
            ),
            Paper(
                id="savitch_1970",
                title="Relationships between nondeterministic and deterministic tape complexities",
                authors=["Walter J. Savitch"],
                year=1970,
                category=PaperCategory.JOURNAL_ARTICLE,
                difficulty=PaperDifficulty.RESEARCH,
                topics=[TopicArea.SPACE_COMPLEXITY, TopicArea.COMPLEXITY_THEORY],
                abstract="Proves Savitch's theorem: NSPACE(s(n)) ⊆ DSPACE(s²(n)).",
                venue="Journal of Computer and System Sciences",
                volume="4",
                number="2",
                pages="177-192",
                keywords=["space complexity", "nondeterminism", "Savitch's theorem"],
                citations_count=2000,
                impact_score=8.8,
                is_seminal=True,
                prerequisites=["complexity_theory", "turing_machines"]
            ),
            Paper(
                id="immerman_1988",
                title="Nondeterministic space is closed under complementation",
                authors=["Neil Immerman"],
                year=1988,
                category=PaperCategory.JOURNAL_ARTICLE,
                difficulty=PaperDifficulty.RESEARCH,
                topics=[TopicArea.SPACE_COMPLEXITY, TopicArea.DESCRIPTIVE_COMPLEXITY],
                abstract="Proves that NSPACE is closed under complement (Immerman-Szelepcsényi theorem).",
                venue="SIAM Journal on Computing",
                volume="17",
                number="4",
                pages="935-938",
                keywords=["space complexity", "complement", "descriptive complexity"],
                citations_count=800,
                impact_score=8.5,
                is_seminal=True,
                prerequisites=["savitch_1970", "descriptive_complexity"]
            ),
            Paper(
                id="ladner_1975",
                title="On the Structure of Polynomial Time Reducibility",
                authors=["Richard E. Ladner"],
                year=1975,
                category=PaperCategory.JOURNAL_ARTICLE,
                difficulty=PaperDifficulty.RESEARCH,
                topics=[TopicArea.COMPLEXITY_THEORY, TopicArea.P_VS_NP],
                abstract="Proves that if P ≠ NP, then there exist NP-intermediate problems.",
                venue="Journal of the ACM",
                volume="22",
                number="1",
                pages="155-171",
                keywords=["polynomial reductions", "NP-intermediate", "complexity hierarchy"],
                citations_count=1200,
                impact_score=8.0,
                is_seminal=True,
                prerequisites=["cook_1971", "karp_1972"]
            ),
            Paper(
                id="garey_johnson_1979",
                title="Computers and Intractability: A Guide to the Theory of NP-Completeness",
                authors=["Michael R. Garey", "David S. Johnson"],
                year=1979,
                category=PaperCategory.TEXTBOOK,
                difficulty=PaperDifficulty.GRADUATE,
                topics=[TopicArea.COMPLEXITY_THEORY, TopicArea.P_VS_NP],
                abstract="Comprehensive guide to NP-completeness theory with extensive problem catalog.",
                publisher="W. H. Freeman",
                isbn="978-0716710455",
                keywords=["NP-completeness", "intractability", "complexity theory"],
                impact_score=9.6,
                is_seminal=True,
                prerequisites=["cook_1971", "discrete_mathematics"]
            )
        ]
        
        for paper in seminal_papers:
            self.add_paper(paper)
    
    def add_paper(self, paper: Paper):
        """Add a paper to the database"""
        self.papers[paper.id] = paper
        
        # Update indices
        for topic in paper.topics:
            if topic not in self.topic_index:
                self.topic_index[topic] = set()
            self.topic_index[topic].add(paper.id)
        
        for author in paper.authors:
            author_key = author.lower()
            if author_key not in self.author_index:
                self.author_index[author_key] = set()
            self.author_index[author_key].add(paper.id)
        
        for keyword in paper.keywords:
            keyword_key = keyword.lower()
            if keyword_key not in self.keyword_index:
                self.keyword_index[keyword_key] = set()
            self.keyword_index[keyword_key].add(paper.id)
    
    def _save_database(self):
        """Save database to file"""
        try:
            db_path = Path(__file__).parent / "papers_database.json"
            data = {
                "papers": [paper.dict() for paper in self.papers.values()],
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "1.0",
                    "total_papers": len(self.papers)
                }
            }
            with open(db_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving papers database: {e}")
    
    def search_papers(self, request: PaperSearchRequest) -> List[Paper]:
        """Search papers based on criteria"""
        results = set(self.papers.keys())
        
        # Filter by query (title, abstract, keywords)
        if request.query:
            query_lower = request.query.lower()
            query_results = set()
            
            for paper_id, paper in self.papers.items():
                # Search in title
                if query_lower in paper.title.lower():
                    query_results.add(paper_id)
                # Search in abstract
                elif paper.abstract and query_lower in paper.abstract.lower():
                    query_results.add(paper_id)
                # Search in keywords
                elif any(query_lower in keyword.lower() for keyword in paper.keywords):
                    query_results.add(paper_id)
                # Search in authors
                elif any(query_lower in author.lower() for author in paper.authors):
                    query_results.add(paper_id)
            
            results = results.intersection(query_results)
        
        # Filter by topics
        if request.topics:
            topic_results = set()
            for topic in request.topics:
                if topic in self.topic_index:
                    topic_results.update(self.topic_index[topic])
            results = results.intersection(topic_results)
        
        # Filter by difficulty
        if request.difficulty:
            difficulty_results = {
                paper_id for paper_id, paper in self.papers.items()
                if paper.difficulty == request.difficulty
            }
            results = results.intersection(difficulty_results)
        
        # Filter by category
        if request.category:
            category_results = {
                paper_id for paper_id, paper in self.papers.items()
                if paper.category == request.category
            }
            results = results.intersection(category_results)
        
        # Filter by year range
        if request.year_range:
            start_year, end_year = request.year_range
            year_results = {
                paper_id for paper_id, paper in self.papers.items()
                if start_year <= paper.year <= end_year
            }
            results = results.intersection(year_results)
        
        # Convert to Paper objects and sort by relevance
        papers = [self.papers[paper_id] for paper_id in results]
        papers.sort(key=lambda p: (p.is_seminal, p.impact_score, p.citations_count), reverse=True)
        
        return papers[:request.max_results]
    
    def recommend_papers(self, request: PaperRecommendationRequest) -> List[Paper]:
        """Recommend papers based on student profile"""
        recommendations = []
        
        # Get papers matching topics and difficulty
        topic_papers = set()
        for topic in request.topics_of_interest:
            if topic in self.topic_index:
                topic_papers.update(self.topic_index[topic])
        
        # Filter by appropriate difficulty
        suitable_papers = []
        for paper_id in topic_papers:
            paper = self.papers[paper_id]
            if self._is_appropriate_difficulty(paper.difficulty, request.student_level):
                suitable_papers.append(paper)
        
        # If looking for related papers to current one
        if request.current_paper_id and request.current_paper_id in self.papers:
            current_paper = self.papers[request.current_paper_id]
            related_papers = self._find_related_papers(current_paper)
            suitable_papers.extend(related_papers)
        
        # Remove duplicates and sort by relevance
        seen = set()
        unique_papers = []
        for paper in suitable_papers:
            if paper.id not in seen:
                seen.add(paper.id)
                unique_papers.append(paper)
        
        # Score and sort papers
        scored_papers = []
        for paper in unique_papers:
            score = self._calculate_recommendation_score(paper, request)
            scored_papers.append((paper, score))
        
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        return [paper for paper, score in scored_papers[:request.max_recommendations]]
    
    def _is_appropriate_difficulty(self, paper_difficulty: PaperDifficulty, 
                                  student_level: PaperDifficulty) -> bool:
        """Check if paper difficulty is appropriate for student level"""
        difficulty_order = {
            PaperDifficulty.UNDERGRADUATE: 0,
            PaperDifficulty.GRADUATE: 1,
            PaperDifficulty.ADVANCED_GRADUATE: 2,
            PaperDifficulty.RESEARCH: 3
        }
        
        paper_level = difficulty_order[paper_difficulty]
        student_level_num = difficulty_order[student_level]
        
        # Allow papers at same level or one level above
        return paper_level <= student_level_num + 1
    
    def _find_related_papers(self, current_paper: Paper) -> List[Paper]:
        """Find papers related to the current paper"""
        related = []
        
        # Find papers with overlapping topics
        for paper_id, paper in self.papers.items():
            if paper_id == current_paper.id:
                continue
            
            # Check topic overlap
            topic_overlap = len(set(paper.topics) & set(current_paper.topics))
            if topic_overlap > 0:
                related.append(paper)
        
        # Find papers that cite or are cited by current paper
        # (In a real system, this would use citation data)
        
        return related[:10]  # Limit results
    
    def _calculate_recommendation_score(self, paper: Paper, 
                                       request: PaperRecommendationRequest) -> float:
        """Calculate recommendation score for a paper"""
        score = 0.0
        
        # Base score from impact
        score += paper.impact_score
        
        # Bonus for seminal papers
        if paper.is_seminal:
            score += 2.0
        
        # Topic relevance
        topic_overlap = len(set(paper.topics) & set(request.topics_of_interest))
        score += topic_overlap * 1.5
        
        # Learning objective bonus
        if request.learning_objective == "research" and paper.category in [
            PaperCategory.RESEARCH_PAPER, PaperCategory.JOURNAL_ARTICLE
        ]:
            score += 1.0
        elif request.learning_objective == "theory" and paper.category == PaperCategory.TEXTBOOK:
            score += 1.0
        
        # Recent papers bonus (for non-historical context)
        if paper.year >= 2000:
            score += 0.5
        
        return score
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """Get paper by ID"""
        return self.papers.get(paper_id)
    
    def get_papers_by_topic(self, topic: TopicArea) -> List[Paper]:
        """Get all papers for a topic"""
        if topic not in self.topic_index:
            return []
        
        paper_ids = self.topic_index[topic]
        papers = [self.papers[pid] for pid in paper_ids]
        papers.sort(key=lambda p: (p.is_seminal, p.impact_score), reverse=True)
        return papers
    
    def get_papers_by_author(self, author_name: str) -> List[Paper]:
        """Get papers by author"""
        author_key = author_name.lower()
        if author_key not in self.author_index:
            return []
        
        paper_ids = self.author_index[author_key]
        papers = [self.papers[pid] for pid in paper_ids]
        papers.sort(key=lambda p: p.year, reverse=True)
        return papers

class CitationFormatter:
    """Formats citations in various academic styles"""
    
    def format_citation(self, paper: Paper, style: CitationStyle, 
                       include_url: bool = True) -> str:
        """Format paper citation in specified style"""
        if style == CitationStyle.APA:
            return self._format_apa(paper, include_url)
        elif style == CitationStyle.MLA:
            return self._format_mla(paper, include_url)
        elif style == CitationStyle.IEEE:
            return self._format_ieee(paper, include_url)
        elif style == CitationStyle.ACM:
            return self._format_acm(paper, include_url)
        elif style == CitationStyle.BIBTEX:
            return self._format_bibtex(paper)
        else:
            return self._format_apa(paper, include_url)  # Default to APA
    
    def _format_apa(self, paper: Paper, include_url: bool) -> str:
        """Format in APA style"""
        authors = self._format_authors_apa(paper.authors)
        title = paper.title
        year = str(paper.year)
        
        if paper.category == PaperCategory.TEXTBOOK:
            citation = f"{authors} ({year}). {title}. {paper.publisher}."
        elif paper.category in [PaperCategory.JOURNAL_ARTICLE, PaperCategory.RESEARCH_PAPER]:
            venue = paper.venue or "Unknown Journal"
            volume_info = ""
            if paper.volume:
                volume_info = f", {paper.volume}"
                if paper.number:
                    volume_info += f"({paper.number})"
            pages = f", {paper.pages}" if paper.pages else ""
            citation = f"{authors} ({year}). {title}. {venue}{volume_info}{pages}."
        else:
            citation = f"{authors} ({year}). {title}."
        
        if include_url and paper.url:
            citation += f" Retrieved from {paper.url}"
        elif include_url and paper.doi:
            citation += f" https://doi.org/{paper.doi}"
        
        return citation
    
    def _format_mla(self, paper: Paper, include_url: bool) -> str:
        """Format in MLA style"""
        authors = self._format_authors_mla(paper.authors)
        title = f'"{paper.title}"'
        
        if paper.category == PaperCategory.TEXTBOOK:
            citation = f"{authors}. {title}. {paper.publisher}, {paper.year}."
        elif paper.category in [PaperCategory.JOURNAL_ARTICLE, PaperCategory.RESEARCH_PAPER]:
            venue = paper.venue or "Unknown Journal"
            volume_info = ""
            if paper.volume:
                volume_info = f", vol. {paper.volume}"
                if paper.number:
                    volume_info += f", no. {paper.number}"
            pages = f", pp. {paper.pages}" if paper.pages else ""
            citation = f"{authors}. {title} {venue}{volume_info}, {paper.year}{pages}."
        else:
            citation = f"{authors}. {title} {paper.year}."
        
        if include_url and paper.url:
            citation += f" Web. {paper.url}"
        
        return citation
    
    def _format_ieee(self, paper: Paper, include_url: bool) -> str:
        """Format in IEEE style"""
        authors = self._format_authors_ieee(paper.authors)
        title = f'"{paper.title}"'
        
        if paper.category == PaperCategory.TEXTBOOK:
            citation = f"{authors}, {title}. {paper.publisher}, {paper.year}."
        elif paper.category in [PaperCategory.JOURNAL_ARTICLE, PaperCategory.RESEARCH_PAPER]:
            venue = paper.venue or "Unknown Journal"
            volume_info = ""
            if paper.volume:
                volume_info = f", vol. {paper.volume}"
                if paper.number:
                    volume_info += f", no. {paper.number}"
            pages = f", pp. {paper.pages}" if paper.pages else ""
            citation = f"{authors}, {title}, {venue}{volume_info}{pages}, {paper.year}."
        else:
            citation = f"{authors}, {title}, {paper.year}."
        
        if include_url and paper.doi:
            citation += f" doi: {paper.doi}"
        
        return citation
    
    def _format_acm(self, paper: Paper, include_url: bool) -> str:
        """Format in ACM style"""
        authors = self._format_authors_acm(paper.authors)
        year = str(paper.year)
        title = paper.title
        
        if paper.category == PaperCategory.TEXTBOOK:
            citation = f"{authors}. {year}. {title}. {paper.publisher}."
        elif paper.category in [PaperCategory.CONFERENCE_PAPER]:
            venue = paper.venue or "Unknown Conference"
            pages = f", {paper.pages}" if paper.pages else ""
            citation = f"{authors}. {year}. {title}. In {venue}{pages}."
        elif paper.category in [PaperCategory.JOURNAL_ARTICLE, PaperCategory.RESEARCH_PAPER]:
            venue = paper.venue or "Unknown Journal"
            volume_info = ""
            if paper.volume:
                volume_info = f" {paper.volume}"
                if paper.number:
                    volume_info += f", {paper.number}"
            pages = f" ({paper.year}), {paper.pages}" if paper.pages else f" ({paper.year})"
            citation = f"{authors}. {year}. {title}. {venue}{volume_info}{pages}."
        else:
            citation = f"{authors}. {year}. {title}."
        
        if include_url and paper.doi:
            citation += f" DOI: https://doi.org/{paper.doi}"
        
        return citation
    
    def _format_bibtex(self, paper: Paper) -> str:
        """Format as BibTeX entry"""
        entry_type = self._get_bibtex_type(paper.category)
        authors = " and ".join(paper.authors)
        
        bibtex = f"@{entry_type}{{{paper.id},\n"
        bibtex += f"  title={{{paper.title}}},\n"
        bibtex += f"  author={{{authors}}},\n"
        bibtex += f"  year={{{paper.year}}},\n"
        
        if paper.venue:
            if paper.category == PaperCategory.TEXTBOOK:
                bibtex += f"  publisher={{{paper.venue}}},\n"
            else:
                field_name = "journal" if paper.category == PaperCategory.JOURNAL_ARTICLE else "booktitle"
                bibtex += f"  {field_name}={{{paper.venue}}},\n"
        
        if paper.volume:
            bibtex += f"  volume={{{paper.volume}}},\n"
        if paper.number:
            bibtex += f"  number={{{paper.number}}},\n"
        if paper.pages:
            bibtex += f"  pages={{{paper.pages}}},\n"
        if paper.publisher and paper.category != PaperCategory.TEXTBOOK:
            bibtex += f"  publisher={{{paper.publisher}}},\n"
        if paper.doi:
            bibtex += f"  doi={{{paper.doi}}},\n"
        if paper.url:
            bibtex += f"  url={{{paper.url}}},\n"
        
        bibtex = bibtex.rstrip(",\n") + "\n}"
        return bibtex
    
    def _get_bibtex_type(self, category: PaperCategory) -> str:
        """Get BibTeX entry type for paper category"""
        mapping = {
            PaperCategory.TEXTBOOK: "book",
            PaperCategory.JOURNAL_ARTICLE: "article",
            PaperCategory.CONFERENCE_PAPER: "inproceedings",
            PaperCategory.RESEARCH_PAPER: "article",
            PaperCategory.THESIS: "phdthesis",
            PaperCategory.SURVEY: "article",
            PaperCategory.TUTORIAL: "misc"
        }
        return mapping.get(category, "misc")
    
    def _format_authors_apa(self, authors: List[str]) -> str:
        """Format authors for APA style"""
        if len(authors) == 1:
            return self._lastname_first(authors[0])
        elif len(authors) == 2:
            return f"{self._lastname_first(authors[0])}, & {authors[1]}"
        else:
            formatted = [self._lastname_first(authors[0])]
            formatted.extend(authors[1:-1])
            formatted.append(f"& {authors[-1]}")
            return ", ".join(formatted)
    
    def _format_authors_mla(self, authors: List[str]) -> str:
        """Format authors for MLA style"""
        if len(authors) == 1:
            return self._lastname_first(authors[0])
        elif len(authors) == 2:
            return f"{self._lastname_first(authors[0])}, and {authors[1]}"
        else:
            return f"{self._lastname_first(authors[0])}, et al"
    
    def _format_authors_ieee(self, authors: List[str]) -> str:
        """Format authors for IEEE style"""
        if len(authors) <= 3:
            return ", ".join(authors)
        else:
            return f"{authors[0]}, et al"
    
    def _format_authors_acm(self, authors: List[str]) -> str:
        """Format authors for ACM style"""
        return ", ".join(authors)
    
    def _lastname_first(self, name: str) -> str:
        """Convert name to Lastname, F. format"""
        parts = name.split()
        if len(parts) < 2:
            return name
        
        lastname = parts[-1]
        firstnames = parts[:-1]
        initials = ". ".join([f[0] for f in firstnames]) + "."
        return f"{lastname}, {initials}"

# Global instances
paper_database = PaperDatabase()
citation_formatter = CitationFormatter()

def search_papers(request: PaperSearchRequest) -> List[Paper]:
    """Search papers in database"""
    return paper_database.search_papers(request)

def recommend_papers(request: PaperRecommendationRequest) -> List[Paper]:
    """Get paper recommendations"""
    return paper_database.recommend_papers(request)

def get_paper_citation(request: CitationRequest) -> Optional[str]:
    """Get formatted citation for paper"""
    paper = paper_database.get_paper_by_id(request.paper_id)
    if not paper:
        return None
    
    return citation_formatter.format_citation(paper, request.style, request.include_url)

def get_papers_by_topic(topic: TopicArea) -> List[Paper]:
    """Get papers for specific topic"""
    return paper_database.get_papers_by_topic(topic)

def get_paper_details(paper_id: str) -> Optional[Paper]:
    """Get detailed information about a paper"""
    return paper_database.get_paper_by_id(paper_id)

def get_database_stats() -> Dict[str, Any]:
    """Get statistics about the papers database"""
    return {
        "total_papers": len(paper_database.papers),
        "seminal_papers": sum(1 for p in paper_database.papers.values() if p.is_seminal),
        "topics_covered": len(paper_database.topic_index),
        "authors_count": len(paper_database.author_index),
        "categories": {
            category.value: sum(1 for p in paper_database.papers.values() if p.category == category)
            for category in PaperCategory
        },
        "difficulty_distribution": {
            difficulty.value: sum(1 for p in paper_database.papers.values() if p.difficulty == difficulty)
            for difficulty in PaperDifficulty
        }
    }