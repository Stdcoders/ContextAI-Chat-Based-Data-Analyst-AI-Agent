#!/usr/bin/env python3
"""
Universal Calculation Engine v2
LLM-first approach that prioritizes intelligent analysis with numerical backing for ANY dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from gemma_llm import GemmaLLM
import json

class UniversalCalculationEngine:
    """
    Universal calculation engine that uses LLM intelligence first, with mathematical precision.
    Works with any dataset type - economics, healthcare, business, sports, etc.
    """
    
    def __init__(self):
        self.llm = GemmaLLM(
            model_name="google/gemma-2-9b-it",
            temperature=0.15,  # Low temperature for precision
            max_tokens=2000   # Higher limit for detailed analysis
        )
    
    def analyze_with_calculations(self, question: str, data: pd.DataFrame) -> str:
        """
        Primary method: Performs comprehensive LLM-driven analysis with numerical computations.
        This is the main entry point that replaces all the specific calculation methods.
        """
        try:
            # Create rich data context for the LLM
            data_context = self._create_comprehensive_data_context(data)
            
            # Generate LLM analysis with calculation focus
            analysis_result = self._perform_llm_calculation_analysis(question, data_context, data)
            
            if analysis_result and self._is_valid_analysis(analysis_result):
                print(f"[UniversalCalculationEngine] Generated comprehensive analysis: {len(analysis_result)} chars")
                return analysis_result
            else:
                print(f"[UniversalCalculationEngine] LLM analysis insufficient, using computational backup")
                return self._computational_backup_analysis(question, data)
                
        except Exception as e:
            print(f"[UniversalCalculationEngine] Error in analysis: {e}")
            return self._basic_computational_fallback(question, data)
    
    def _perform_llm_calculation_analysis(self, question: str, data_context: str, data: pd.DataFrame) -> str:
        """
        Uses LLM to perform intelligent analysis with calculation focus.
        """
        prompt = f"""You are an expert quantitative analyst capable of working with ANY type of dataset. 
Your specialty is performing precise numerical analysis while providing clear, insightful explanations.

QUESTION: {question}

DATASET CONTEXT:
{data_context}

INSTRUCTIONS FOR ANALYSIS:
1. **IDENTIFY CALCULATIONS NEEDED**: What specific numerical operations does this question require?
2. **LOCATE RELEVANT DATA**: Which columns and data points are most important?
3. **PERFORM CALCULATIONS**: Execute the necessary mathematical operations step-by-step
4. **PROVIDE NUMERICAL RESULTS**: Give precise numbers, percentages, rates, etc.
5. **EXPLAIN METHODOLOGY**: Clearly describe how you arrived at the results
6. **CONTEXTUALIZE FINDINGS**: Interpret what the numbers mean in practical terms

CALCULATION CAPABILITIES:
- Growth rates, percentage changes, CAGR
- Statistical measures (mean, median, standard deviation, variance)
- Correlations and relationships between variables  
- Time series analysis and trend detection
- Comparative analysis between groups/periods
- Ratios, proportions, and normalized values
- Sum, totals, averages across categories
- Min/max values and ranges
- Volatility and risk measures

RESPONSE FORMAT:
Provide a comprehensive analysis that includes:
- **Direct Answer**: Clear numerical answer to the specific question
- **Calculations**: Step-by-step mathematical work shown
- **Key Metrics**: Important numbers and statistics
- **Methodology**: How the calculations were performed
- **Interpretation**: What these results mean and their significance
- **Context**: How results relate to the broader dataset

Focus on being both mathematically precise and practically insightful. Show your work clearly.
"""

        try:
            response = self.llm.generate(prompt)
            return response if response else ""
        except Exception as e:
            print(f"[UniversalCalculationEngine] LLM generation error: {e}")
            return ""
    
    def _create_comprehensive_data_context(self, data: pd.DataFrame) -> str:
        """
        Creates rich context about the dataset for the LLM to understand.
        """
        try:
            context_parts = []
            
            # Basic dataset info
            context_parts.append(f"Dataset Shape: {data.shape[0]} rows × {data.shape[1]} columns")
            
            # Column analysis with type detection
            context_parts.append("\nCOLUMN ANALYSIS:")
            for col in data.columns[:12]:  # Limit to avoid token overflow
                col_info = self._analyze_column(data[col], col)
                context_parts.append(f"• {col}: {col_info}")
            
            # Sample data
            if len(data) > 0:
                context_parts.append(f"\nSAMPLE DATA (first 3 rows):")
                context_parts.append(data.head(3).to_string(max_cols=8))
                
                if len(data) > 3:
                    context_parts.append(f"\nSAMPLE DATA (last 2 rows):")
                    context_parts.append(data.tail(2).to_string(max_cols=8))
            
            # Identify potential time/date columns
            time_cols = self._identify_time_columns(data)
            if time_cols:
                context_parts.append(f"\nTIME COLUMNS DETECTED: {time_cols}")
            
            # Identify key numeric columns
            numeric_cols = self._identify_key_numeric_columns(data)
            if numeric_cols:
                context_parts.append(f"\nKEY NUMERIC COLUMNS: {numeric_cols}")
                
            return "\n".join(context_parts)
            
        except Exception as e:
            return f"Dataset: {data.shape[0]} rows × {data.shape[1]} columns (context generation error: {e})"
    
    def _analyze_column(self, series: pd.Series, col_name: str) -> str:
        """Analyze individual column to understand its nature."""
        try:
            # Try numeric conversion
            numeric_series = pd.to_numeric(series, errors='coerce')
            numeric_ratio = numeric_series.count() / len(series)
            
            if numeric_ratio > 0.8:  # Mostly numeric
                stats = {
                    'min': numeric_series.min(),
                    'max': numeric_series.max(), 
                    'mean': numeric_series.mean()
                }
                return f"Numeric ({numeric_ratio*100:.0f}% valid), range: {stats['min']:.2f} to {stats['max']:.2f}, avg: {stats['mean']:.2f}"
            
            elif numeric_ratio > 0.3:  # Partially numeric
                sample_numeric = numeric_series.dropna().head(2).tolist()
                sample_text = series[series != numeric_series].head(2).tolist()
                return f"Mixed ({numeric_ratio*100:.0f}% numeric), numeric samples: {sample_numeric}, text samples: {sample_text}"
            
            else:  # Mostly categorical/text
                unique_count = series.nunique()
                top_values = series.value_counts().head(3).index.tolist()
                return f"Categorical ({unique_count} unique values), top: {top_values}"
                
        except Exception as e:
            return f"Analysis error: {e}"
    
    def _identify_time_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify columns that might contain time/date information."""
        time_columns = []
        time_keywords = ['year', 'date', 'time', 'period', 'month', 'quarter']
        
        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in time_keywords):
                time_columns.append(col)
            # Also check if values look like years
            elif col_lower in ['year'] or self._looks_like_year_column(data[col]):
                time_columns.append(col)
                
        return time_columns
    
    def _looks_like_year_column(self, series: pd.Series) -> bool:
        """Check if a column contains year-like values."""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) > 0:
                # Check if values are in reasonable year range
                min_val, max_val = numeric_series.min(), numeric_series.max()
                return 1900 <= min_val <= 2030 and 1900 <= max_val <= 2030
        except:
            pass
        return False
    
    def _identify_key_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify the most important numeric columns for calculations."""
        numeric_cols = []
        
        for col in data.columns:
            numeric_series = pd.to_numeric(data[col], errors='coerce')
            if numeric_series.count() > len(data) * 0.7:  # At least 70% numeric
                numeric_cols.append(col)
        
        return numeric_cols[:6]  # Return top 6 most relevant
    
    def _computational_backup_analysis(self, question: str, data: pd.DataFrame) -> str:
        """
        Backup method that performs computational analysis when LLM fails.
        """
        try:
            # Identify numeric columns
            numeric_columns = self._identify_key_numeric_columns(data)
            
            if not numeric_columns:
                return "❌ No suitable numeric data found for calculations."
            
            # Perform basic statistical analysis
            results = []
            results.append("🔢 **Computational Analysis Results**\n")
            
            for col in numeric_columns[:4]:  # Limit to prevent overflow
                numeric_series = pd.to_numeric(data[col], errors='coerce').dropna()
                
                if len(numeric_series) > 0:
                    stats = {
                        'mean': numeric_series.mean(),
                        'median': numeric_series.median(),
                        'std': numeric_series.std(),
                        'min': numeric_series.min(),
                        'max': numeric_series.max(),
                        'count': len(numeric_series)
                    }
                    
                    results.append(f"**{col}:**")
                    results.append(f"  • Average: {stats['mean']:,.2f}")
                    results.append(f"  • Range: {stats['min']:,.2f} to {stats['max']:,.2f}")
                    results.append(f"  • Std Dev: {stats['std']:,.2f}")
                    results.append(f"  • Data Points: {stats['count']:,}")
                    
                    # Calculate growth if we can identify time structure
                    if len(numeric_series) >= 2:
                        growth = ((numeric_series.iloc[-1] - numeric_series.iloc[0]) / numeric_series.iloc[0]) * 100
                        results.append(f"  • Overall Change: {growth:+.2f}%")
                    results.append("")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"❌ Computational analysis error: {e}"
    
    def _basic_computational_fallback(self, question: str, data: pd.DataFrame) -> str:
        """
        Basic fallback when everything else fails.
        """
        return f"""🔍 **Basic Dataset Summary**

Dataset contains {data.shape[0]} rows and {data.shape[1]} columns.

Available columns: {', '.join(data.columns[:8].tolist())}

For calculation questions, please ensure your data contains appropriate numeric columns and try rephrasing your question with specific column names if possible.
"""
    
    def _is_valid_analysis(self, analysis: str) -> bool:
        """
        Check if the LLM analysis is substantial and useful.
        """
        if not analysis or len(analysis.strip()) < 100:
            return False
        
        # Check for signs of actual analysis
        analysis_indicators = [
            'calculation', 'analysis', 'result', 'average', 'mean', 'total',
            'growth', 'change', 'rate', '%', 'correlation', 'trend',
            'compare', 'difference', 'ratio', 'statistic'
        ]
        
        analysis_lower = analysis.lower()
        indicator_count = sum(1 for indicator in analysis_indicators if indicator in analysis_lower)
        
        return indicator_count >= 2  # At least 2 analytical indicators