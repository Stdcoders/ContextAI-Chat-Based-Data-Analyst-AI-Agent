#!/usr/bin/env python3
"""
Fixed Question Classification and Routing System
Properly routes questions to appropriate engines based on classification
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

class QuestionType(Enum):
    STATISTICAL = "statistical"      # Numbers, calculations, aggregations
    ANALYTICAL = "analytical"        # Patterns, insights, relationships  
    DESCRIPTIVE = "descriptive"      # What/who/where questions, summaries
    COMPARATIVE = "comparative"      # Comparing groups, ranking, top/bottom
    EXPLORATORY = "exploratory"      # Broad exploration/pattern discovery

@dataclass
class ClassificationResult:
    question_type: QuestionType
    confidence: float
    requires_visualization: bool
    visualization_type: str
    reasoning: str

class QuestionClassifier:
    """Clean, simple question classifier"""
    
    def classify(self, question: str, df: pd.DataFrame) -> ClassificationResult:
        """Classify question and determine processing approach"""
        q_lower = question.lower()
        
        # Statistical keywords - numbers, calculations, comparisons
        statistical_keywords = [
            'calculate', 'compute', 'sum', 'total', 'average', 'mean', 'median',
            'minimum', 'maximum', 'min', 'max', 'count', 'how many',
            'top', 'bottom', 'highest', 'lowest', 'rank', 'compare',
            'percentage', 'percent', 'ratio', 'correlation'
        ]
        
        # Analytical keywords - patterns, relationships, insights
        analytical_keywords = [
            'analyze', 'analysis', 'trend', 'pattern', 'relationship',
            'correlation', 'impact', 'influence', 'affect', 'insight',
            'why', 'how does', 'what causes', 'factor', 'predict'
        ]
        
        # Descriptive keywords - what/who/where, summaries
        descriptive_keywords = [
            'what is', 'what are', 'who', 'where', 'describe', 'explain',
            'tell me about', 'show me', 'list', 'summary', 'overview'
        ]
        
        # Count keyword matches
        stat_score = sum(1 for kw in statistical_keywords if kw in q_lower)
        analytical_score = sum(1 for kw in analytical_keywords if kw in q_lower)
        desc_score = sum(1 for kw in descriptive_keywords if kw in q_lower)
        
        # Determine primary type
        scores = {
            QuestionType.STATISTICAL: stat_score,
            QuestionType.ANALYTICAL: analytical_score,
            QuestionType.DESCRIPTIVE: desc_score
        }
        
        primary_type = max(scores, key=scores.get)
        total = stat_score + analytical_score + desc_score
        confidence = (scores[primary_type] / total) if total > 0 else 0.5
        
        # Determine visualization needs
        viz_needed = self._needs_visualization(q_lower, primary_type, df)
        viz_type = self._suggest_visualization_type(q_lower, primary_type, df) if viz_needed else "none"
        
        reasoning = f"Detected {primary_type.value} question with {confidence:.1%} confidence"
        
        return ClassificationResult(
            question_type=primary_type,
            confidence=confidence,
            requires_visualization=viz_needed,
            visualization_type=viz_type,
            reasoning=reasoning
        )
    
    def _needs_visualization(self, question: str, q_type: QuestionType, df: pd.DataFrame) -> bool:
        """Determine if question needs visualization"""
        # Never visualize basic info questions
        no_viz_keywords = ['how many columns', 'data types', 'shape', 'info about dataset']
        if any(kw in question for kw in no_viz_keywords):
            return False
        
        # Statistical questions usually benefit from charts
        if q_type == QuestionType.STATISTICAL:
            return True
            
        # Analytical questions about patterns/trends need visualizations
        if q_type == QuestionType.ANALYTICAL:
            pattern_keywords = ['trend', 'pattern', 'distribution', 'relationship', 'over time']
            return any(kw in question for kw in pattern_keywords)
        
        # Descriptive questions rarely need charts unless asking for distributions
        if q_type == QuestionType.DESCRIPTIVE:
            return 'distribution' in question or 'spread' in question
        
        return False
    
    def _suggest_visualization_type(self, question: str, q_type: QuestionType, df: pd.DataFrame) -> str:
        """Suggest specific visualization type based on question and data"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Time series patterns
        if any(word in question for word in ['over time', 'trend', 'seasonal']):
            return 'line_chart'
        
        # Distribution questions
        if 'distribution' in question:
            # Prefer bar chart for categorical distributions; histogram for numeric
            if categorical_cols:
                return 'bar_chart'
            if numeric_cols:
                return 'histogram'
            
        # Comparison questions
        if any(word in question for word in ['compare', 'versus', 'top', 'bottom', 'rank']):
            if categorical_cols and numeric_cols:
                return 'bar_chart'
        
        # Relationship questions
        if any(word in question for word in ['relationship', 'correlation']) and len(numeric_cols) >= 2:
            return 'scatter_plot'
        
        # Default based on data structure
        if categorical_cols and numeric_cols:
            return 'bar_chart'
        elif numeric_cols:
            return 'histogram'
        else:
            return 'summary_table'

class UnifiedAnalysisEngine:
    """Single engine that routes to appropriate analysis based on classification"""
    
    def __init__(self):
        self.classifier = QuestionClassifier()
        # Import your existing engines
        from nodes.hybrid_calculation_engine import HybridCalculationEngine
        from nodes.visualizationinsightsnode import InsightAgent
        
        self.hybrid_engine = HybridCalculationEngine()  # For statistical
        self.insight_agent = InsightAgent()              # For analytical/descriptive
        
    def analyze_question(self, question: str, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Main entry point - classify and route appropriately with enhanced intelligence"""
        
        # Step 1: Classify the question using LLM
        classification = self.classifier.classify(question, df)
        
        print(f"🎯 Question Type: {classification.question_type.value.upper()}")
        print(f"💡 Reasoning: {classification.reasoning}")
        print(f"📊 Needs Visualization: {classification.requires_visualization}")
        if classification.requires_visualization:
            print(f"📈 Visualization Type: {classification.visualization_type}")
        
        # Step 2: Route to appropriate engine with enhanced logic
        try:
            if classification.question_type == QuestionType.STATISTICAL:
                result = self._handle_statistical_question(question, df, classification)
            elif classification.question_type == QuestionType.ANALYTICAL:
                result = self._handle_analytical_question(question, df, dataset_name, classification)
            elif classification.question_type == QuestionType.COMPARATIVE:
                result = self._handle_comparative_question(question, df, dataset_name, classification)
            elif classification.question_type == QuestionType.EXPLORATORY:
                result = self._handle_exploratory_question(question, df, dataset_name, classification)
            else:  # DESCRIPTIVE
                result = self._handle_descriptive_question(question, df, dataset_name, classification)
            
            # Step 3: Add enhanced metadata
            result['classification'] = {
                'type': classification.question_type.value,
                'confidence': classification.confidence,
                'reasoning': classification.reasoning,
                'requires_visualization': classification.requires_visualization,
                'visualization_type': classification.visualization_type,
                'complexity': getattr(classification, 'complexity', 'moderate')
            }
            result['question_type'] = classification.question_type.value
            result['processing_method'] = result.get('method', 'unified_analysis')
            
            return result
            
        except Exception as e:
            print(f"❌ Analysis routing failed: {e}")
            # Fallback to basic analysis
            return self._handle_fallback_analysis(question, df, dataset_name)
    
    def _handle_statistical_question(self, question: str, df: pd.DataFrame, classification: ClassificationResult) -> Dict[str, Any]:
        """Handle statistical questions with hybrid calculation engine + LLM insights"""
        print("⚡ Using Hybrid Calculation Engine for statistical analysis...")
        
        # Use hybrid engine for calculations
        calculation_result = self.hybrid_engine.analyze_with_calculations(question, df)
        
        # Generate intelligent insights via InsightAgent
        ia_result = self._run_intelligent_insight(df, question)
        
        # Build combined answer
        combined_answer = f"⚡ **Statistical Analysis**\n\n{calculation_result}"
        if ia_result and ia_result.get('answer'):
            combined_answer += f"\n\n---\n\n🧠 **Intelligent Insights**\n\n{ia_result['answer']}"
        
        # Prefer visualization from intelligent analysis; else create via LLM-driven selector
        viz_html = None
        if ia_result and ia_result.get('visualization_html'):
            viz_html = ia_result['visualization_html']
        elif classification.requires_visualization:
            viz_html = self._create_llm_driven_visualization(df, question)
        
        viz_json = getattr(self.insight_agent, 'last_figure_json', None)
        
        return {
            'question': question,
            'answer': combined_answer,
            'visualization_html': viz_html,
            'visualization_json': viz_json,
            'method': 'statistical_hybrid_engine+intelligent_insights',
            'primary_focus': 'calculations_and_numbers'
        }
    
    def _handle_analytical_question(self, question: str, df: pd.DataFrame, dataset_name: str, classification: ClassificationResult) -> Dict[str, Any]:
        """Handle analytical questions with insight agent"""
        print("🧠 Using Insight Agent for analytical analysis...")
        
        # Use insight agent but focus on analytical aspects
        result = self.insight_agent.answer(df, question, dataset_name)
        
        # Ensure visualization is provided via LLM-driven selector
        if classification.requires_visualization and not result.get('visualization_html'):
            viz_html = self._create_llm_driven_visualization(df, question)
            result['visualization_html'] = viz_html
            result['visualization_json'] = getattr(self.insight_agent, 'last_figure_json', None)
        
        # Enhance answer with analytical framing
        original_answer = result.get('answer', '')
        enhanced_answer = f"🧠 **Analytical Insights**\n\n{original_answer}"
        result['answer'] = enhanced_answer
        result['method'] = 'analytical_insight_agent'
        result['primary_focus'] = 'patterns_and_relationships'
        
        return result
    
    def _handle_descriptive_question(self, question: str, df: pd.DataFrame, dataset_name: str, classification: ClassificationResult) -> Dict[str, Any]:
        """Handle descriptive questions with summary + LLM intelligent insights"""
        print("📋 Using Descriptive Analysis...")
        
        # Base descriptive summary
        summary = self._generate_descriptive_answer(question, df)
        
        # Intelligent insights via InsightAgent
        ia_result = self._run_intelligent_insight(df, question)
        
        # Build combined answer
        combined_answer = f"📋 **Descriptive Summary**\n\n{summary}"
        if ia_result and ia_result.get('answer'):
            combined_answer += f"\n\n---\n\n🧠 **Intelligent Insights**\n\n{ia_result['answer']}"
        
        # Prefer visualization from intelligent analysis; else use LLM-driven selector if needed
        viz_html = None
        if ia_result and ia_result.get('visualization_html'):
            viz_html = ia_result['visualization_html']
        elif classification.requires_visualization:
            viz_html = self._create_llm_driven_visualization(df, question)
        
        viz_json = getattr(self.insight_agent, 'last_figure_json', None)
        
        return {
            'question': question,
            'answer': combined_answer,
            'visualization_html': viz_html,
            'visualization_json': viz_json,
            'method': 'descriptive_summary+intelligent_insights',
            'primary_focus': 'direct_information_and_insights'
        }
    
    def _generate_descriptive_answer(self, question: str, df: pd.DataFrame) -> str:
        """Generate direct descriptive answers"""
        q_lower = question.lower()
        
        if 'columns' in q_lower or 'variables' in q_lower:
            return f"Dataset contains {len(df.columns)} columns: {', '.join(df.columns.tolist())}"
        
        if 'shape' in q_lower or 'size' in q_lower:
            return f"Dataset has {df.shape[0]:,} rows and {df.shape[1]} columns"
        
        if 'data types' in q_lower or 'types' in q_lower:
            type_summary = df.dtypes.value_counts()
            return f"Data types: {dict(type_summary)}"
        
        # Default descriptive summary
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        
        return f"""Dataset Overview:
• {df.shape[0]:,} records across {df.shape[1]} variables
• {numeric_cols} numeric columns, {categorical_cols} categorical columns
• Missing values: {df.isnull().sum().sum()} total"""
    
    def _create_statistical_visualization(self, df: pd.DataFrame, question: str, viz_type: str) -> str:
        """Create visualization for statistical questions"""
        from utils.plotting_utils import plot_histogram, plot_bar, plot_scatter
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if viz_type == 'histogram' and numeric_cols:
            return plot_histogram(df, numeric_cols[0])
        elif viz_type == 'bar_chart':
            if categorical_cols and numeric_cols:
                grouped = df.groupby(categorical_cols[0])[numeric_cols[0]].mean().reset_index()
                return plot_bar(grouped, categorical_cols[0], numeric_cols[0])
            elif categorical_cols:
                counts = df[categorical_cols[0]].value_counts().reset_index()
                counts.columns = [categorical_cols[0], 'count']
                return plot_bar(counts, categorical_cols[0], 'count')
        elif viz_type == 'scatter_plot' and len(numeric_cols) >= 2:
            return plot_scatter(df, numeric_cols[0], numeric_cols[1])
        
        return None
    
    def _create_analytical_visualization(self, df: pd.DataFrame, question: str, viz_type: str) -> str:
        """Create visualization for analytical questions"""
        # Similar to statistical but focused on patterns
        return self._create_statistical_visualization(df, question, viz_type)
    
    def _create_descriptive_visualization(self, df: pd.DataFrame, question: str, viz_type: str) -> str:
        """Create minimal visualization for descriptive questions"""
        if viz_type == 'summary_table':
            return None  # Tables don't need plotly charts
        return self._create_statistical_visualization(df, question, viz_type)

    def _create_llm_driven_visualization(self, df: pd.DataFrame, question: str) -> Optional[str]:
        """Centralized LLM-driven visualization for ALL question types via InsightAgent"""
        try:
            # Compute column types once
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Delegate to InsightAgent's LLM-guided visualization helper
            if hasattr(self.insight_agent, '_suggest_visualization'):
                return self.insight_agent._suggest_visualization(df, question, numeric_cols, categorical_cols)
            # Fallback to minimal if method not present
            return self._create_statistical_visualization(df, question, 'bar_chart' if categorical_cols else 'histogram')
        except Exception as e:
            print(f"⚠️ LLM-driven visualization failed: {e}")
            return None

    def _run_intelligent_insight(self, df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
        """Invoke InsightAgent's LLM-powered _intelligent_analysis to generate insights."""
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if hasattr(self.insight_agent, '_intelligent_analysis'):
                return self.insight_agent._intelligent_analysis(df, question, numeric_cols, categorical_cols)
        except Exception as e:
            print(f"⚠️ Intelligent insight generation failed: {e}")
        return None

    def _handle_fallback_analysis(self, question: str, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Safe fallback when routing/handlers fail. Provide a minimal, useful answer."""
        try:
            # Very basic descriptive summary as a fallback answer
            rows, cols = df.shape
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            sample_cols = ', '.join(list(df.columns[:6]))
            answer = (
                f"Fallback analysis invoked due to routing error.\n\n"
                f"Dataset shape: {rows:,} rows × {cols} columns\n"
                f"Numeric columns: {len(numeric_cols)} | Categorical columns: {len(categorical_cols)}\n"
                f"Sample columns: {sample_cols}"
            )
            return {
                'question': question,
                'answer': answer,
                'visualization_html': None,
                'method': 'fallback_basic',
                'primary_focus': 'general'
            }
        except Exception as e:
            return {
                'question': question,
                'answer': f"Fallback failed: {e}",
                'visualization_html': None,
                'method': 'fallback_error',
                'primary_focus': 'general'
            }


# Usage in your simple_workflow.py - replace the complex routing with:
"""
def answer_question_with_context(self, question: str):
    # Replace all the complex logic with:
    analysis_engine = UnifiedAnalysisEngine()
    result = analysis_engine.analyze_question(question, df, self.current_dataset)
    
    # Display results
    print(result['answer'])
    
    # Handle visualization
    if result.get('visualization_html'):
        self.save_visualization(result['visualization_html'], question)
"""