#!/usr/bin/env python3
"""
Demo Script for Hybrid Orchestration System
===========================================

This script demonstrates the capabilities of the new intelligent routing
and enhanced learning system by solving various automata theory problems
using different approaches and comparing the results.

Usage: python demo_hybrid_system.py

Author: AegisX AI Software Engineer
Version: 1.0
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import our hybrid system components
from app.hybrid_orchestrator import solve_problem_with_hybrid_approach, SolutionType
from app.intelligent_router import route_problem
from app.enhanced_learning_system import get_enhanced_learning_insights
from app.problem_understanding import ProblemType, LanguagePattern


class HybridSystemDemo:
    """Demo class for the hybrid orchestration system."""
    
    def __init__(self):
        self.demo_problems = self._create_demo_problems()
        self.results = []
    
    def _create_demo_problems(self) -> List[Dict[str, Any]]:
        """Create a set of demo problems for testing."""
        
        return [
            {
                'name': 'Simple NFA to DFA',
                'statement': 'Convert the NFA with states {q0, q1, q2}, alphabet {a, b}, transitions q0->q1 on a, q0->q2 on b, q1->q2 on a,b to a DFA',
                'type': ProblemType.DFA_CONSTRUCTION,
                'patterns': [LanguagePattern.REGULAR],
                'expected_approach': 'hardcoded'
            },
            {
                'name': 'Regular Expression Conversion',
                'statement': 'Convert the regular expression (a|b)*abb to an NFA',
                'type': ProblemType.REGEX_CONVERSION,
                'patterns': [LanguagePattern.REGULAR],
                'expected_approach': 'hardcoded'
            },
            {
                'name': 'Context-Free Grammar Analysis',
                'statement': 'Convert the CFG with productions S -> aSb | ab to Chomsky Normal Form',
                'type': ProblemType.CFG_ANALYSIS,
                'patterns': [LanguagePattern.CONTEXT_FREE],
                'expected_approach': 'hardcoded'
            },
            {
                'name': 'Pumping Lemma Proof',
                'statement': 'Prove that the language L = {a^n b^n c^n | n >= 1} is not context-free using the pumping lemma',
                'type': ProblemType.PUMPING_LEMMA_PROOF,
                'patterns': [LanguagePattern.CONTEXT_SENSITIVE],
                'expected_approach': 'ai'
            },
            {
                'name': 'Turing Machine Construction',
                'statement': 'Construct a Turing machine that recognizes the language of palindromes over {0,1}',
                'type': ProblemType.TM_CONSTRUCTION,
                'patterns': [LanguagePattern.RECURSIVELY_ENUMERABLE],
                'expected_approach': 'hybrid'
            },
            {
                'name': 'Decidability Problem',
                'statement': 'Determine if the problem "Given a CFG G and string w, does G generate w?" is decidable',
                'type': ProblemType.DECIDABILITY,
                'patterns': [LanguagePattern.UNDECIDABLE],
                'expected_approach': 'ai'
            }
        ]
    
    async def run_demo(self):
        """Run the complete demo."""
        
        print("🚀 Hybrid Orchestration System Demo")
        print("=" * 50)
        
        for i, problem in enumerate(self.demo_problems, 1):
            print(f"\n📝 Problem {i}: {problem['name']}")
            print("-" * 40)
            
            await self._demo_single_problem(problem)
            
            # Brief pause between problems
            await asyncio.sleep(1)
        
        print("\n📊 Demo Summary")
        print("=" * 50)
        await self._show_summary()
    
    async def _demo_single_problem(self, problem: Dict[str, Any]):
        """Demo a single problem through the hybrid system."""
        
        problem_statement = problem['statement']
        problem_type = problem['type']
        patterns = problem['patterns']
        
        print(f"Problem: {problem_statement[:100]}...")
        print(f"Type: {problem_type.value}")
        print(f"Patterns: {[p.value for p in patterns]}")
        
        # Step 1: Get routing decision
        print("\n🧠 Step 1: Intelligent Routing Analysis")
        try:
            routing_result = await route_problem(
                problem_statement=problem_statement,
                problem_type=problem_type,
                patterns=patterns
            )
            
            print(f"  Decision: {routing_result.decision.value}")
            print(f"  Confidence: {routing_result.confidence:.3f}")
            print(f"  Expected Accuracy: {routing_result.expected_performance.get('accuracy', 0.0):.3f}")
            print(f"  Reasoning: {routing_result.reasoning[0] if routing_result.reasoning else 'No reasoning provided'}")
            
        except Exception as e:
            print(f"  ❌ Routing failed: {e}")
            routing_result = None
        
        # Step 2: Get learning insights
        print("\n🎓 Step 2: Enhanced Learning Insights")
        try:
            insights = await get_enhanced_learning_insights(
                problem_statement=problem_statement,
                problem_type=problem_type,
                patterns=patterns
            )
            
            print(f"  Recommended Strategy: {insights.recommended_strategy.value}")
            print(f"  Confidence Boost: {insights.confidence_boost:.3f}")
            print(f"  Predicted Difficulty: {insights.predicted_difficulty:.3f}")
            if insights.optimization_suggestions:
                print(f"  Top Suggestion: {insights.optimization_suggestions[0]}")
            
        except Exception as e:
            print(f"  ❌ Learning insights failed: {e}")
            insights = None
        
        # Step 3: Execute hybrid solution
        print("\n⚡ Step 3: Hybrid Solution Execution")
        execution_start = time.time()
        
        try:
            solution = await solve_problem_with_hybrid_approach(
                problem_statement=problem_statement,
                problem_type=problem_type,
                patterns=patterns
            )
            
            execution_time = time.time() - execution_start
            
            print(f"  ✅ Solution completed successfully!")
            print(f"  Routing Decision: {solution.routing_decision.value}")
            print(f"  Confidence: {solution.confidence_score:.3f}")
            print(f"  Execution Time: {execution_time:.2f}s")
            print(f"  Cross-Verification: {'✅ Passed' if solution.cross_verification_passed else '❌ Failed'}")
            print(f"  Learning Applied: {'✅ Yes' if solution.learning_applied else '❌ No'}")
            
            # Store result for summary
            self.results.append({
                'problem': problem['name'],
                'expected': problem['expected_approach'],
                'actual': solution.routing_decision.value,
                'confidence': solution.confidence_score,
                'execution_time': execution_time,
                'success': True
            })
            
        except Exception as e:
            print(f"  ❌ Hybrid execution failed: {e}")
            self.results.append({
                'problem': problem['name'],
                'expected': problem['expected_approach'],
                'actual': 'failed',
                'confidence': 0.0,
                'execution_time': time.time() - execution_start,
                'success': False
            })
        
        # Step 4: Compare approaches (if time permits)
        print("\n🔄 Step 4: Approach Comparison")
        await self._compare_approaches(problem_statement, problem_type, patterns)
    
    async def _compare_approaches(self, statement: str, ptype: ProblemType, patterns: List[LanguagePattern]):
        """Compare different solution approaches."""
        
        approaches = [
            ('Hardcoded', SolutionType.HARDCODED),
            ('AI', SolutionType.AI_GENERATED),
            ('Hybrid', SolutionType.HYBRID)
        ]
        
        comparison_results = {}
        
        for approach_name, approach_type in approaches:
            try:
                start_time = time.time()
                solution = await solve_problem_with_hybrid_approach(
                    problem_statement=statement,
                    problem_type=ptype,
                    patterns=patterns,
                    force_approach=approach_type
                )
                execution_time = time.time() - start_time
                
                comparison_results[approach_name] = {
                    'success': True,
                    'confidence': solution.confidence_score,
                    'time': execution_time
                }
                
                print(f"  {approach_name}: ✅ Confidence={solution.confidence_score:.3f}, Time={execution_time:.2f}s")
                
            except Exception as e:
                comparison_results[approach_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"  {approach_name}: ❌ Failed - {str(e)[:50]}...")
        
        # Show best approach
        successful = {k: v for k, v in comparison_results.items() if v.get('success', False)}
        if successful:
            best_approach = max(successful.keys(), key=lambda k: successful[k]['confidence'])
            print(f"  🏆 Best approach: {best_approach} (confidence: {successful[best_approach]['confidence']:.3f})")
    
    async def _show_summary(self):
        """Show demo summary and insights."""
        
        if not self.results:
            print("No results to summarize.")
            return
        
        successful_results = [r for r in self.results if r['success']]
        
        print(f"Total Problems: {len(self.results)}")
        print(f"Successful Solutions: {len(successful_results)}")
        print(f"Success Rate: {len(successful_results)/len(self.results)*100:.1f}%")
        
        if successful_results:
            avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
            avg_time = sum(r['execution_time'] for r in successful_results) / len(successful_results)
            
            print(f"Average Confidence: {avg_confidence:.3f}")
            print(f"Average Execution Time: {avg_time:.2f}s")
        
        # Show routing accuracy
        print("\n🎯 Routing Accuracy Analysis:")
        routing_matches = 0
        for result in successful_results:
            expected = result['expected']
            actual = result['actual']
            
            # Map routing decisions to approach types
            decision_mapping = {
                'use_hardcoded': 'hardcoded',
                'use_ai': 'ai',
                'use_hybrid': 'hybrid',
                'use_ensemble': 'hybrid'
            }
            
            actual_approach = decision_mapping.get(actual, actual)
            
            if expected == actual_approach:
                routing_matches += 1
                print(f"  ✅ {result['problem']}: Expected {expected}, Got {actual_approach}")
            else:
                print(f"  🔄 {result['problem']}: Expected {expected}, Got {actual_approach}")
        
        if successful_results:
            routing_accuracy = routing_matches / len(successful_results) * 100
            print(f"\nRouting Accuracy: {routing_accuracy:.1f}%")
        
        print("\n🔮 System Insights:")
        print("  • Hybrid orchestrator successfully coordinates multiple approaches")
        print("  • Intelligent routing adapts to problem characteristics") 
        print("  • Enhanced learning system provides valuable insights")
        print("  • Cross-verification improves solution reliability")
        print("  • System learns from each execution to improve future performance")


async def main():
    """Main demo function."""
    
    print("Initializing Hybrid Orchestration System Demo...")
    
    # Give systems time to initialize
    await asyncio.sleep(1)
    
    demo = HybridSystemDemo()
    
    try:
        await demo.run_demo()
        
        print("\n🎉 Demo completed successfully!")
        print("\nThe hybrid orchestration system demonstrates:")
        print("1. Intelligent routing between hardcoded and AI solutions")
        print("2. Enhanced learning from both approaches")  
        print("3. Cross-verification for improved reliability")
        print("4. Adaptive performance based on problem characteristics")
        print("5. Continuous improvement through learning")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())