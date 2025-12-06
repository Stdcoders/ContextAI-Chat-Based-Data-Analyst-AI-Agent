#!/usr/bin/env python3
"""
Calculation Engine for Data Analysis
Handles precise numerical computations when users ask calculation-based questions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from gemma_llm import GemmaLLM
import re
import datetime

class CalculationEngine:
    """
    Universal calculation engine that handles numerical computations for ANY dataset type.
    Uses LLM-driven analysis as the primary method with precise mathematical backing.
    """
    
    def __init__(self):
        self.llm = GemmaLLM(
            model_name="google/gemma-2-9b-it",
            temperature=0.2,  # Low temperature for precise analysis
            max_tokens=1500   # Higher limit for detailed explanations
        )
    
    def is_calculation_question(self, question: str, df: pd.DataFrame) -> bool:
        """
        Use LLM to determine if a question requires numerical calculations.
        """
        try:
            # Get column information for context
            columns_info = self._get_columns_summary(df)
            
            prompt = f"""You are a data analysis expert. Determine if this question requires NUMERICAL CALCULATIONS or STATISTICAL COMPUTATIONS.

QUESTION: "{question}"

AVAILABLE DATA COLUMNS:
{columns_info}

**CALCULATION QUESTIONS** require specific numerical operations like:
- Growth rates, percentages, differences
- Mathematical operations (+, -, *, /, averages, sums)
- Statistical calculations (mean, median, correlation, standard deviation)
- Comparisons between specific values or time periods
- Quantitative analysis with precise numbers

**NON-CALCULATION QUESTIONS** are more qualitative like:
- General patterns, trends, insights
- Explanations or interpretations
- Recommendations or strategies
- Open-ended analysis

Examples:
- "What was the GDP growth rate from 2019 to 2023?" → CALCULATION
- "Calculate the correlation between exports and GDP" → CALCULATION
- "What are the main economic trends?" → NON-CALCULATION
- "How has the economy performed overall?" → NON-CALCULATION

Answer with only: CALCULATION or NON-CALCULATION"""

            response = self.llm(prompt).strip().upper()
            
            return "CALCULATION" in response
            
        except Exception as e:
            print(f"⚠️ Calculation detection failed: {e}")
            # Fallback to keyword detection
            return self._fallback_calculation_detection(question)
    
    def _fallback_calculation_detection(self, question: str) -> bool:
        """Fallback method using keyword detection"""
        calculation_keywords = [
            'calculate', 'compute', 'growth rate', 'percentage', 'percent', '%',
            'correlation', 'average', 'mean', 'median', 'sum', 'total',
            'difference', 'change', 'increase', 'decrease', 'ratio',
            'from', 'to', 'between', 'during', 'standard deviation',
            'volatility', 'variance', 'compare', 'vs', 'versus'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in calculation_keywords)
    
    def perform_calculation(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform the actual calculation based on the user's question.
        """
        try:
            # Step 1: Analyze the question to understand what calculation is needed
            calculation_plan = self._analyze_calculation_need(question, df)
            
            # Step 2: Execute the calculation
            if calculation_plan['type'] == 'growth_rate':
                result = self._calculate_growth_rate(df, calculation_plan)
            elif calculation_plan['type'] == 'correlation':
                result = self._calculate_correlation(df, calculation_plan)
            elif calculation_plan['type'] == 'statistical':
                result = self._calculate_statistics(df, calculation_plan)
            elif calculation_plan['type'] == 'comparison':
                result = self._calculate_comparison(df, calculation_plan)
            elif calculation_plan['type'] == 'time_series':
                result = self._calculate_time_series(df, calculation_plan)
            else:
                result = self._generic_calculation(df, calculation_plan)
            
            # Step 3: Format the result with explanation
            formatted_result = self._format_calculation_result(question, result, calculation_plan)
            
            return {
                "question": question,
                "answer": formatted_result,
                "calculation_type": calculation_plan['type'],
                "raw_result": result,
                "method": "precise_calculation"
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"❌ Calculation error: {str(e)}. Please check if the required data is available in the correct format.",
                "calculation_type": "error",
                "method": "calculation_error"
            }
    
    def _analyze_calculation_need(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze what type of calculation is needed and identify relevant columns.
        """
        columns_info = self._get_columns_summary(df)
        
        prompt = f"""Analyze this calculation question and determine the computation needed.

QUESTION: "{question}"

AVAILABLE DATA COLUMNS:
{columns_info}

SAMPLE DATA (first 3 rows):
{df.head(3).to_string()}

Determine:
1. **CALCULATION TYPE**: growth_rate, correlation, statistical, comparison, time_series, or other
2. **RELEVANT COLUMNS**: Which specific columns are needed (use exact column names)
3. **TIME PERIOD**: If specific years/periods mentioned
4. **OPERATION**: What mathematical operation is required

Respond in this exact format:
TYPE: [calculation_type]
COLUMNS: [column1, column2, ...]
TIME_PERIOD: [if applicable]
OPERATION: [specific calculation needed]
"""

        response = self.llm(prompt)
        
        # Parse the LLM response
        plan = self._parse_calculation_plan(response, df)
        return plan
    
    def _parse_calculation_plan(self, response: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse the LLM response into a structured calculation plan."""
        plan = {
            'type': 'statistical',
            'columns': [],
            'time_period': None,
            'operation': 'basic_stats'
        }
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('TYPE:'):
                plan['type'] = line.replace('TYPE:', '').strip().lower()
            elif line.startswith('COLUMNS:'):
                columns_text = line.replace('COLUMNS:', '').strip()
                # Extract column names and validate they exist
                columns = [col.strip(' [],"') for col in columns_text.split(',')]
                plan['columns'] = [col for col in columns if col in df.columns]
            elif line.startswith('TIME_PERIOD:'):
                plan['time_period'] = line.replace('TIME_PERIOD:', '').strip()
            elif line.startswith('OPERATION:'):
                plan['operation'] = line.replace('OPERATION:', '').strip()
        
        # Fallback: if no columns identified, try to find relevant ones
        if not plan['columns']:
            plan['columns'] = self._find_relevant_columns(df)
            
        return plan
    
    def _find_relevant_columns(self, df: pd.DataFrame) -> List[str]:
        """Find the most relevant columns for calculation."""
        # Prioritize columns that can be converted to numeric
        relevant_cols = []
        
        for col in df.columns:
            if col.lower() in ['unnamed: 0']:  # Skip index columns
                continue
            
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if numeric_series.count() > len(df) * 0.3:  # At least 30% numeric
                relevant_cols.append(col)
        
        return relevant_cols[:5]  # Return top 5 most relevant columns
    
    def _calculate_growth_rate(self, df: pd.DataFrame, plan: Dict) -> Dict[str, Any]:
        """Calculate growth rates between time periods."""
        if not plan['columns']:
            return {"error": "No suitable columns found for growth rate calculation"}
        
        results = {}
        
        for col in plan['columns']:
            # Convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if len(numeric_series) < 2:
                continue
                
            # Calculate overall growth rate
            start_value = numeric_series.iloc[0]
            end_value = numeric_series.iloc[-1]
            
            if start_value != 0:
                growth_rate = ((end_value - start_value) / start_value) * 100
                results[col] = {
                    'growth_rate_percent': growth_rate,
                    'start_value': start_value,
                    'end_value': end_value,
                    'absolute_change': end_value - start_value
                }
        
        return results
    
    def _calculate_correlation(self, df: pd.DataFrame, plan: Dict) -> Dict[str, Any]:
        """Calculate correlations between variables."""
        if len(plan['columns']) < 2:
            return {"error": "Need at least 2 columns for correlation"}
        
        # Convert columns to numeric
        numeric_data = {}
        for col in plan['columns']:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if numeric_series.count() > len(df) * 0.3:
                numeric_data[col] = numeric_series
        
        if len(numeric_data) < 2:
            return {"error": "Insufficient numeric data for correlation"}
        
        # Create correlation matrix
        correlation_df = pd.DataFrame(numeric_data).corr()
        
        results = {
            'correlation_matrix': correlation_df.to_dict(),
            'strongest_correlations': []
        }
        
        # Find strongest correlations
        for i, col1 in enumerate(correlation_df.columns):
            for j, col2 in enumerate(correlation_df.columns[i+1:], i+1):
                corr_val = correlation_df.iloc[i, j]
                if not pd.isna(corr_val):
                    results['strongest_correlations'].append({
                        'column1': col1,
                        'column2': col2,
                        'correlation': corr_val
                    })
        
        # Sort by absolute correlation value
        results['strongest_correlations'].sort(
            key=lambda x: abs(x['correlation']), reverse=True
        )
        
        return results
    
    def _calculate_statistics(self, df: pd.DataFrame, plan: Dict) -> Dict[str, Any]:
        """Calculate statistical measures."""
        results = {}
        
        for col in plan['columns']:
            numeric_series = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if len(numeric_series) == 0:
                continue
                
            stats = {
                'mean': numeric_series.mean(),
                'median': numeric_series.median(),
                'std': numeric_series.std(),
                'min': numeric_series.min(),
                'max': numeric_series.max(),
                'count': len(numeric_series),
                'range': numeric_series.max() - numeric_series.min()
            }
            
            results[col] = stats
        
        return results
    
    def _calculate_comparison(self, df: pd.DataFrame, plan: Dict) -> Dict[str, Any]:
        """Calculate comparisons between different groups or time periods."""
        # Implementation for comparison calculations
        return self._calculate_statistics(df, plan)
    
    def _calculate_time_series(self, df: pd.DataFrame, plan: Dict) -> Dict[str, Any]:
        """Calculate time series metrics."""
        # Implementation for time series calculations
        return self._calculate_statistics(df, plan)
    
    def _generic_calculation(self, df: pd.DataFrame, plan: Dict) -> Dict[str, Any]:
        """Generic calculation fallback."""
        return self._calculate_statistics(df, plan)
    
    def _format_calculation_result(self, question: str, result: Dict, plan: Dict) -> str:
        """Format the calculation result into a clear, readable answer."""
        if "error" in result:
            return f"❌ {result['error']}"
        
        if not result:
            return "❌ No calculations could be performed with the available data."
        
        formatted_answer = "🔢 **Calculation Results**\n\n"
        
        if plan['type'] == 'growth_rate':
            for col, data in result.items():
                if isinstance(data, dict) and 'growth_rate_percent' in data:
                    formatted_answer += f"📈 **{col}**:\n"
                    formatted_answer += f"  • Growth Rate: {data['growth_rate_percent']:.2f}%\n"
                    formatted_answer += f"  • Start Value: {data['start_value']:,.2f}\n"
                    formatted_answer += f"  • End Value: {data['end_value']:,.2f}\n"
                    formatted_answer += f"  • Absolute Change: {data['absolute_change']:,.2f}\n\n"
        
        elif plan['type'] == 'correlation':
            formatted_answer += "📊 **Correlation Analysis**:\n\n"
            if 'strongest_correlations' in result:
                for i, corr in enumerate(result['strongest_correlations'][:5], 1):
                    strength = "Strong" if abs(corr['correlation']) > 0.7 else "Moderate" if abs(corr['correlation']) > 0.3 else "Weak"
                    formatted_answer += f"{i}. {corr['column1']} ↔ {corr['column2']}: {corr['correlation']:.3f} ({strength})\n"
        
        elif plan['type'] == 'statistical':
            formatted_answer += "📊 **Statistical Summary**:\n\n"
            for col, stats in result.items():
                if isinstance(stats, dict):
                    formatted_answer += f"**{col}**:\n"
                    formatted_answer += f"  • Mean: {stats.get('mean', 0):,.2f}\n"
                    formatted_answer += f"  • Median: {stats.get('median', 0):,.2f}\n"
                    formatted_answer += f"  • Std Dev: {stats.get('std', 0):,.2f}\n"
                    formatted_answer += f"  • Range: {stats.get('min', 0):,.2f} to {stats.get('max', 0):,.2f}\n"
                    formatted_answer += f"  • Count: {stats.get('count', 0):,} values\n\n"
        
        return formatted_answer
    
    def _get_columns_summary(self, df: pd.DataFrame) -> str:
        """Get a summary of available columns for LLM context."""
        summary_lines = []
        
        for col in df.columns[:10]:  # Limit to first 10 columns
            # Check if column can be converted to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            numeric_pct = (numeric_series.count() / len(df)) * 100
            
            if numeric_pct > 50:
                sample_vals = numeric_series.dropna().head(3).tolist()
                summary_lines.append(f"- {col}: Numeric ({numeric_pct:.0f}% numeric), samples: {sample_vals}")
            else:
                unique_vals = df[col].value_counts().head(3).index.tolist()
                summary_lines.append(f"- {col}: Categorical, top values: {unique_vals}")
        
        return "\n".join(summary_lines)