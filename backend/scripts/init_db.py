#!/usr/bin/env python3
"""
Initialize the database with tables and sample data.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_db, SessionLocal, Problem, engine
from app.config import settings
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_problems():
    """Load sample problems into the database."""
    db = SessionLocal()
    
    try:
        # Check if problems already exist
        existing_count = db.query(Problem).count()
        if existing_count > 0:
            logger.info(f"Database already contains {existing_count} problems")
            return
        
        # Sample problems
        sample_problems = [
            {
                "id": "dfa_even_length",
                "type": "dfa",
                "title": "Even Length Binary Strings",
                "description": "Construct a DFA that accepts all binary strings of even length.",
                "difficulty": "beginner",
                "category": "regular_languages",
                "language_description": "L = {w ∈ {0,1}* | |w| is even}",
                "alphabet": ["0", "1"],
                "test_strings": [
                    {"string": "", "should_accept": True},
                    {"string": "0", "should_accept": False},
                    {"string": "00", "should_accept": True},
                    {"string": "101", "should_accept": False},
                    {"string": "1010", "should_accept": True}
                ],
                "hints": [
                    "Think about tracking whether you've seen an even or odd number of symbols",
                    "You'll need exactly 2 states",
                    "The start state should also be an accept state (empty string has even length)"
                ],
                "concepts": ["dfa", "regular_languages", "state_machines"]
            },
            {
                "id": "nfa_contains_substring",
                "type": "nfa",
                "title": "Contains '101' Substring",
                "description": "Construct an NFA that accepts all binary strings containing '101' as a substring.",
                "difficulty": "intermediate",
                "category": "regular_languages",
                "language_description": "L = {w ∈ {0,1}* | w contains '101'}",
                "alphabet": ["0", "1"],
                "test_strings": [
                    {"string": "101", "should_accept": True},
                    {"string": "0101", "should_accept": True},
                    {"string": "1010", "should_accept": True},
                    {"string": "111", "should_accept": False},
                    {"string": "000", "should_accept": False}
                ],
                "hints": [
                    "NFAs are great for pattern matching",
                    "You can stay in the start state until you see the beginning of the pattern",
                    "Think about what happens after you've seen '1', then '10', then '101'"
                ],
                "concepts": ["nfa", "pattern_matching", "nondeterminism"]
            },
            {
                "id": "pda_balanced_parentheses",
                "type": "pda",
                "title": "Balanced Parentheses",
                "description": "Construct a PDA that accepts all strings of balanced parentheses.",
                "difficulty": "intermediate",
                "category": "context_free",
                "language_description": "L = {w ∈ {(,)}* | w has balanced parentheses}",
                "alphabet": ["(", ")"],
                "test_strings": [
                    {"string": "()", "should_accept": True},
                    {"string": "(())", "should_accept": True},
                    {"string": "()()", "should_accept": True},
                    {"string": "(()", "should_accept": False},
                    {"string": "())", "should_accept": False}
                ],
                "hints": [
                    "Use the stack to keep track of open parentheses",
                    "Push '(' onto the stack when you see one",
                    "Pop from the stack when you see ')'"
                ],
                "concepts": ["pda", "context_free", "stack", "balanced_parentheses"]
            }
        ]
        
        # Insert problems
        for problem_data in sample_problems:
            problem = Problem(**problem_data)
            db.add(problem)
        
        db.commit()
        logger.info(f"Successfully loaded {len(sample_problems)} sample problems")
        
    except Exception as e:
        logger.error(f"Error loading sample problems: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def main():
    """Main initialization function."""
    logger.info("Initializing database...")
    
    # Create tables
    init_db()
    logger.info("Database tables created successfully")
    
    # Load sample data
    load_sample_problems()
    
    logger.info("Database initialization complete!")


if __name__ == "__main__":
    main()