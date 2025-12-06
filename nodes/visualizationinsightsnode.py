import os
import re
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from dotenv import load_dotenv
from gemma_llm import GemmaLLM
from groq_llm import GroqLLM
from .hybrid_calculation_engine import HybridCalculationEngine
from .llm_question_classifier import LLMQuestionClassifier
import utils.plotting_utils as plotting_utils

# Structured + Text plotting
from utils.plotting_utils import (
    # Structured
    plot_histogram, plot_scatter, plot_bar, plot_line,
    plot_box, plot_violin, plot_heatmap, plot_correlation_matrix, plot_timeseries,
    # Text
    plot_word_frequencies, plot_wordcloud, sentiment_distribution
)

# Import WorkflowState if you're using a global state
from utils.state import STATE

load_dotenv()


class InsightAgent:
    def __init__(self, model: str = "google/gemma-2-9b-it"):
        self.model_name = model
        self.llm = None  # Gemma for analytical insights
        self.groq_llm = None  # Groq for fast calculations
        self.dataset_cache = {}       # ✅ cache per dataset
        self.analysis_history = {}    # ✅ history per dataset
        self.text_cache = {}          # ✅ text datasets (e.g., pdf, txt)
        self.calculation_engine = HybridCalculationEngine()  # 🔥 Hybrid engine with DeepSeek + Gemma
        
        # 🎯 NEW: LLM-powered question intent classifier
        try:
            self.question_classifier = LLMQuestionClassifier()
            print("🎯 LLM Question Classifier initialized")
        except Exception as e:
            print(f"⚠️ Question classifier failed to initialize: {e}")
            self.question_classifier = None
        
        self._init_llms()

    # ================= INIT LLMs =================
    def _init_llms(self):
        # Initialize Gemma-2-9B-IT for analytical insights
        try:
            self.llm = GemmaLLM(
                model_name="google/gemma-2-9b-it",
                temperature=0.2,  # Lower temperature for more precise analysis
                max_tokens=2000   # Higher token limit for detailed analysis
            )
            
            if self.llm.is_available():
                print("Initialization of Insight Node successful!")
            else:
                self.llm = None
                
        except Exception as e:
            print(f"⚠️ Failed to init Gemma LLM ({e}).")
            self.llm = None
        
        # Initialize Groq GPT-OSS-120B for ultra-fast calculations
        try:
            self.groq_llm = GroqLLM(
                model_name="openai/gpt-oss-120b",
                temperature=0.0,  # Zero temperature for precise calculations
                max_tokens=2000
            )
            
            if self.groq_llm.is_available():
                print("Initialization of calculation node successful!")
            else:
                self.groq_llm = None
                
        except Exception as e:
            print(f"Failed to init Groq LLM ({e}).")
            self.groq_llm = None

    # ================= CACHE =================
    def set_dataset(self, dataset_name: str, df: pd.DataFrame):
        """Cache DataFrame-level metrics for reuse."""
        if dataset_name not in self.dataset_cache:
            # Get properly typed columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            
            # Safe correlation matrix calculation
            correlation_matrix = None
            if len(numeric_cols) > 1:
                try:
                    # Only use truly numeric columns for correlation
                    numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 1:
                        correlation_matrix = numeric_df.corr()
                except Exception as e:
                    print(f"Correlation calculation failed: {e}")
                    correlation_matrix = None
            
            self.dataset_cache[dataset_name] = {
                "df": df,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "basic_stats": self._compute_basic_stats(df),
                "correlation_matrix": correlation_matrix,
            }
            self.analysis_history[dataset_name] = []

    def set_text_dataset(self, dataset_name: str, docs: list[str]):
        """Cache text data (e.g., from PDF or txt)."""
        if dataset_name not in self.text_cache:
            self.text_cache[dataset_name] = {
                "docs": docs,
                "stats": {
                    "num_docs": len(docs),
                    "avg_len": np.mean([len(d.split()) for d in docs]) if docs else 0
                }
            }
            self.analysis_history[dataset_name] = []

    def _compute_basic_stats(self, df: pd.DataFrame) -> dict:
        stats_dict = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "q1": df[col].quantile(0.25),
                "q3": df[col].quantile(0.75),
                "count": df[col].count(),
            }
        return stats_dict

    # ================= LLM-DRIVEN INTELLIGENT ANALYSIS =================
    def _generate_visualization_from_llm(self, llm_response: str, df: pd.DataFrame) -> str:
        """
    Parses the LLM response to find a structured visualization suggestion
    and generates the plot using plotting_utils.
    """
        try:
        # Find the JSON block in the LLM's response
            json_match = re.search(r"```json\n(.*)\n```", llm_response, re.DOTALL)
            if not json_match:
                return None

            plot_info = json.loads(json_match.group(1))
            plot_type = plot_info.get("plot_type")

            if not plot_type:
                return None

        # Dynamically call the correct plotting function from plotting_utils
            if hasattr(plotting_utils, f"plot_{plot_type}"):
                plot_function = getattr(plotting_utils, f"plot_{plot_type}")
            # a simple way to get the required arguments for the plot function
                required_args = plot_function.__code__.co_varnames[:plot_function.__code__.co_argcount]
                plot_args = {arg: plot_info.get(arg) for arg in required_args if arg != 'df'}

            # make sure all required arguments are present
                if all(arg in plot_info for arg in plot_args.keys()):
                 return plot_function(df, **plot_args)
            return None
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            print(f"Error generating visualization from LLM response: {e}")
        return None
    
    def _intelligent_analysis(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> dict:
        """LLM-powered intelligent analysis with question classification and routing"""
        
        print(f"Analyzing: {question}")
        
        try:
            # 🎯 STEP 1: Classify question intent using LLM
            if self.question_classifier and self.question_classifier.is_available():
                # Create data context for classification
                dataset_context = self._create_classification_context(df, numeric_cols, categorical_cols)
                
                # Get intent classification
                classification = self.question_classifier.analyze_question_intent(question, dataset_context)
                
                print(f"Intent: {classification.intent.value.upper()} | Engine: {classification.recommended_engine.upper()} | Confidence: {classification.confidence:.1%}")
                print(f"Reasoning: {classification.reasoning}")
                
                # Route to appropriate engine based on classification
                if classification.recommended_engine == "groq" and self.groq_llm and self.groq_llm.is_available():
                    return self._perform_groq_analysis(df, question, classification, numeric_cols, categorical_cols)
                elif classification.recommended_engine == "gemma" and self.llm and self.llm.is_available():
                    return self._perform_gemma_analysis(df, question, classification, numeric_cols, categorical_cols)
                else:
                    print(f"Recommended engine unavailable, using hybrid fallback...")
                    return self._perform_hybrid_fallback(df, question, numeric_cols, categorical_cols)
            else:
                print(f"Question classifier unavailable, using hybrid engine...")
                return self._perform_hybrid_fallback(df, question, numeric_cols, categorical_cols)
            
        except Exception as e:
            print(f"Intelligent analysis failed: {e}")
            return self._fallback_analysis(df, question, numeric_cols, categorical_cols)
    
    def _create_classification_context(self, df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> str:
        """Create concise context for question classification"""
        context = f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns\n"
        
        if numeric_cols:
            context += f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:5]}\n"
        
        if categorical_cols:
            context += f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}\n"
        
        # Add sample data info
        context += f"Sample data structure: {list(df.head(1).to_dict('records'))[0] if len(df) > 0 else 'No data'}"
        
        return context
    
    def _perform_groq_analysis(self, df: pd.DataFrame, question: str, classification, numeric_cols: list, categorical_cols: list) -> dict:
        """Perform ultra-fast statistical/comparative analysis using Groq"""
        try:
            # Create comprehensive data context for Groq
            data_context = self._create_data_context(df, numeric_cols, categorical_cols)
            
            # Use Groq for mathematical calculations and statistical analysis
            groq_prompt = f"""You are a mathematical calculation expert. Perform precise statistical analysis.

QUESTION: {question}

DATA CONTEXT:
{data_context}

PERFORM EXACT CALCULATIONS:
1.  Identify the specific mathematical operations needed
2.  Use the actual data values provided in the context
3.  Calculate precise numerical results
4.  Show your calculation methodology
5.  Provide specific answers with numbers

After your analysis, suggest a single, relevant visualization to help illustrate your findings.
Provide your suggestion in a JSON block like this:
```json
{{
  "plot_type": "the_type_of_plot",
  "title": "A Descriptive Title",
  "x_column": "the_column_for_the_x_axis",
  "y_column": "the_column_for_the_y_axis",
  "group_by": "optional_column_for_grouping"
}}"""
            
            groq_response = self.groq_llm.generate(groq_prompt)
            
            if groq_response and len(groq_response.strip()) > 100:
                # Generate appropriate visualization for statistical analysis
                viz_html = self._suggest_visualization(df, question, numeric_cols, categorical_cols)
                
                # Add visualization reasoning to the answer
                if viz_html:
                    viz_message = self._get_visualization_message(question)
                    enhanced_answer = f"**Ultra-Fast Statistical Analysis**\n\n{groq_response}\n\n---\n\n**Visualization Analysis**\n\n{viz_message}"
                else:
                    enhanced_answer = f"**Ultra-Fast Statistical Analysis**\n\n{groq_response}"
                
                return {
                    "question": question,
                    "answer": enhanced_answer,
                    "visualization_html": viz_html,
                    "method": "groq_statistical_analysis",
                    "classification": classification.intent.value
                }
            else:
                return self._perform_gemma_analysis(df, question, classification, numeric_cols, categorical_cols)
                
        except Exception as e:
            return self._perform_gemma_analysis(df, question, classification, numeric_cols, categorical_cols)
    
    def _perform_gemma_analysis(self, df: pd.DataFrame, question: str, classification, numeric_cols: list, categorical_cols: list) -> dict:
        """Perform rich analytical/descriptive analysis using Gemma and generate a plot from its suggestion."""
        try:

        # 1. Get the detailed analysis from the comprehensive LLM method
            result = self._comprehensive_llm_analysis(df, question, [], numeric_cols, categorical_cols)

            if result and result.get("answer"):
                result["method"] = "gemma_analytical_analysis"
                result["classification"] = classification.intent.value

            # --- ✨ NEW VISUALIZATION LOGIC ---
            # 2. Generate visualization based on the LLM's structured suggestion
            # The LLM response is in result["answer"], which we parse for a JSON block
                viz_html = self._generate_visualization_from_llm(result["answer"], df)
                result["visualization_html"] = viz_html
            # --- END OF NEW LOGIC ---

            # 3. Update the final answer to include insights and the visualization message
                original_answer = result["answer"]
            
            # This part of your code now works as intended because "visualization_html" is populated
                if result.get("visualization_html"):
                    viz_message = self._get_visualization_message(question)
                    enhanced_answer = f"🧠 **Rich Analytical Insights (Gemma-2-9B-IT)**\n\n{original_answer}\n\n---\n\n📊 **Visualization Analysis**\n\n{viz_message}"
                else:
                    enhanced_answer = f"🧠 **Rich Analytical Insights (Gemma-2-9B-IT)**\n\n{original_answer}"

                result["answer"] = enhanced_answer
                return result
            else:
                print("🔄 Gemma analysis was insufficient, switching to hybrid fallback...")
                return self._perform_hybrid_fallback(df, question, numeric_cols, categorical_cols)

        except Exception as e:
            print(f"⚠️ Gemma analysis failed with an error: {e}")
            return self._perform_hybrid_fallback(df, question, numeric_cols, categorical_cols)
    
    def _perform_hybrid_fallback(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> dict:
        """Fallback to hybrid calculation engine when specific engines are unavailable"""
        try:
            print(f"🔥 Using Hybrid Calculation Engine fallback...")
            calculation_result = self.calculation_engine.analyze_with_calculations(question, df)
            
            if calculation_result and len(calculation_result.strip()) > 150:
                # 🔧 FIX: Create visualization for hybrid engine results
                print(f"📊 Creating visualization for hybrid engine result...")
                viz_html = self._suggest_visualization(df, question, numeric_cols, categorical_cols)
                
                return {
                    "question": question,
                    "answer": calculation_result,
                    "method": "hybrid_calculation_engine",
                    "visualization_html": viz_html  # 🔧 FIX: Use visualization_html instead of empty visualizations
                }
            else:
                # Final fallback to analytical analysis
                return self._perform_analytical_analysis(df, question, numeric_cols, categorical_cols)
                
        except Exception as e:
            print(f"⚠️ Hybrid fallback failed: {e}")
            return self._perform_analytical_analysis(df, question, numeric_cols, categorical_cols)
    
    def _perform_analytical_analysis(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> dict:
        """Perform qualitative analytical analysis using LLM intelligence"""
        try:
            # Step 1: Use LLM to determine analysis type
            analysis_type = self._determine_analysis_type_llm(question, numeric_cols, categorical_cols)
            print(f"📊 Analysis type: {analysis_type}")
            
            # Step 2: Use LLM to find relevant columns for the question
            relevant_cols = self._find_relevant_columns_llm(question, numeric_cols, categorical_cols, df)
            print(f"🎯 Using columns: {relevant_cols}")
            print(f"🗺️ Available - Numeric: {numeric_cols[:3]}... Categorical: {categorical_cols[:3]}...")
            
            # Step 3: Try comprehensive LLM analysis first for custom questions
            if self.llm and self.llm.is_available():
                llm_result = self._comprehensive_llm_analysis(df, question, relevant_cols, numeric_cols, categorical_cols)
                if llm_result and llm_result.get("answer") and "error" not in llm_result["answer"].lower():
                    print(f"✅ Using comprehensive LLM analysis")
                    return llm_result
            
            # Step 4: Fallback to structured analysis
            print(f"🔄 Fallback to structured analysis")
            result = self._perform_analysis(df, question, analysis_type, relevant_cols, numeric_cols, categorical_cols)
            
            # Step 5: Generate LLM insights
            if result["answer"]:
                enhanced_insights = self._generate_llm_insights(df, question, result, relevant_cols)
                if enhanced_insights:
                    result["answer"] = enhanced_insights
            
            return result
            
        except Exception as e:
            print(f"⚠️ Analytical analysis failed: {e}")
            return self._fallback_analysis(df, question, numeric_cols, categorical_cols)
    
    def _create_data_context(self, df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> str:
        """Create comprehensive data context for LLM"""
        context = f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        
        # Numeric columns summary
        if numeric_cols:
            context += "NUMERIC COLUMNS:\n"
            for col in numeric_cols[:5]:  # Limit to first 5
                stats = df[col].describe()
                context += f"- {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]\n"
        
        # Categorical columns summary
        if categorical_cols:
            context += "\nCATEGORICAL COLUMNS:\n"
            for col in categorical_cols[:5]:  # Limit to first 5
                unique_count = df[col].nunique()
                top_values = df[col].value_counts().head(3)
                context += f"- {col}: {unique_count} unique values, top: {dict(top_values)}\n"
        
        # Sample data
        context += f"\nSAMPLE DATA (first 3 rows):\n{df.head(3).to_string()}\n"
        
        return context
    
    def _determine_analysis_type_llm(self, question: str, numeric_cols: list, categorical_cols: list) -> str:
        """Use LLM to determine what type of analysis to perform"""
        if not self.llm or not self.llm.is_available():
            # Simple fallback
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                return "comparison"
            elif len(numeric_cols) >= 2:
                return "correlation"
            elif len(numeric_cols) >= 1:
                return "distribution"
            else:
                return "aggregation"
        
        try:
            analysis_types = [
                "correlation - relationships between numeric variables",
                "distribution - patterns and spread of single variable", 
                "comparison - comparing groups or categories",
                "aggregation - summary statistics and totals",
                "trend - changes over time"
            ]
            
            prompt = f"""Question: "{question}"

Available data:
- Numeric columns: {len(numeric_cols)} ({numeric_cols[:3]}...)
- Categorical columns: {len(categorical_cols)} ({categorical_cols[:3]}...)

What type of analysis is most appropriate?

Options:
{chr(10).join(analysis_types)}

Respond with only one word: correlation, distribution, comparison, aggregation, or trend"""
            
            response = self.llm(prompt).strip().lower()
            
            # Validate response
            valid_types = ["correlation", "distribution", "comparison", "aggregation", "trend"]
            if response in valid_types:
                return response
            
            # Try to extract valid type from response
            for valid_type in valid_types:
                if valid_type in response:
                    return valid_type
            
            # Fallback
            return "comparison" if categorical_cols and numeric_cols else "distribution"
            
        except Exception as e:
            print(f"⚠️ LLM analysis type detection failed: {e}")
            return "comparison" if categorical_cols and numeric_cols else "distribution"
    
    def _comprehensive_llm_analysis(self, df: pd.DataFrame, question: str, relevant_cols: list, numeric_cols: list, categorical_cols: list) -> dict:
        """Intelligent data analysis using Gemma-2-9B-IT with smart column interpretation and value analysis"""
        try:
            # Step 1: Create intelligent data context with column interpretation
            data_context = self._create_intelligent_data_context(df, question, relevant_cols, numeric_cols, categorical_cols)
            
            # Step 2: Create expert analysis prompt
            prompt = f"""You are a senior data scientist with expertise in econometrics, statistics, and business intelligence.

QUESTION: {question}

DATASET ANALYSIS:
{data_context}

**YOUR TASK:**
Analyze the actual dataset to provide a comprehensive, data-driven answer. Use the specific column names, values, and patterns shown in the data.

**REQUIREMENTS:**
1. **Column Intelligence**: Interpret what each column represents based on its name and values
2. **Data-Driven Insights**: Use actual numbers, percentages, and statistical measures from the data
3. **Pattern Recognition**: Identify trends, relationships, and anomalies in the actual data
4. **Economic/Domain Context**: Apply relevant domain knowledge (if economics data, use economic principles)
5. **Actionable Conclusions**: Provide specific, measurable insights that answer the question

**ANALYSIS APPROACH:**
- Look at column names to understand what they measure (e.g., "GDP.2" likely means quarterly growth rates)
- Examine actual values to identify patterns, outliers, and trends
- Use statistical measures (means, correlations, volatility) where relevant
- Consider temporal aspects if time-series data is present
- Provide quantified insights with specific numbers from the data

**FORMAT:** Provide clear, bullet-pointed insights with supporting data evidence."""
            
            response = self.llm(prompt)
            
            # Debug: Log the LLM response
            print(f"🔍 LLM Response Length: {len(response.strip()) if response else 0}")
            print(f"🔍 LLM Response Preview: {response[:200] if response else 'None'}...")
            
            # Enhanced validation for intelligent responses
            is_intelligent = self._is_intelligent_response(response) if response else False
            print(f"🔍 Is Intelligent Response: {is_intelligent}")
            
            if response and len(response.strip()) > 50 and is_intelligent:
                # Generate appropriate visualization
                viz_html = self._suggest_intelligent_visualization(df, question, relevant_cols, numeric_cols, categorical_cols)
                
                # Store the actual figure for PNG export
                if viz_html:
                    self._store_figure_for_export(df, question, numeric_cols, categorical_cols)
                
                # Format the response for better readability
                formatted_response = self._format_intelligent_response(response.strip())
                
                return {
                    "question": question,
                    "answer": formatted_response,
                    "visualization_html": viz_html,
                    "method": "intelligent_gemma_analysis"
                }
            
            return None  # Signal to use fallback
            
        except Exception as e:
            print(f"⚠️ Intelligent analysis failed: {e}")
            return None
    
    def _extract_relevant_data_context(self, df: pd.DataFrame, question: str, relevant_cols: list, numeric_cols: list, categorical_cols: list) -> str:
        """Extract deep, relevant data context based on the question type and content"""
        context_parts = [f"Dataset: {df.shape[0]:,} records, {df.shape[1]} features"]
        
        # Detect question type and extract appropriate data
        question_lower = question.lower()
        
        # For comparison questions (between categories, groups, diseases, etc.)
        if any(word in question_lower for word in ['compare', 'difference', 'between', 'vs', 'versus', 'similar']):
            context_parts.extend(self._extract_comparison_context(df, question, relevant_cols, categorical_cols))
        
        # For pattern/trend questions
        elif any(word in question_lower for word in ['pattern', 'trend', 'over time', 'seasonal', 'temporal']):
            context_parts.extend(self._extract_temporal_context(df, question, relevant_cols, numeric_cols))
        
        # For correlation/relationship questions
        elif any(word in question_lower for word in ['correlation', 'relationship', 'impact', 'affect', 'influence']):
            context_parts.extend(self._extract_correlation_context(df, question, relevant_cols, numeric_cols))
        
        # For distribution/frequency questions
        elif any(word in question_lower for word in ['distribution', 'frequency', 'most common', 'least common']):
            context_parts.extend(self._extract_distribution_context(df, question, relevant_cols, categorical_cols))
        
        # For specific value/entity questions
        elif any(word in question_lower for word in ['which', 'what', 'highest', 'lowest', 'best', 'worst', 'top', 'bottom']):
            context_parts.extend(self._extract_specific_value_context(df, question, relevant_cols, numeric_cols, categorical_cols))
        
        # Default: comprehensive context
        else:
            context_parts.extend(self._extract_comprehensive_context(df, relevant_cols, numeric_cols, categorical_cols))
        
        return "\n".join(context_parts)
    
    def _extract_comparison_context(self, df: pd.DataFrame, question: str, relevant_cols: list, categorical_cols: list) -> list:
        """Extract context for comparison questions"""
        context = []
        
        # Find categorical columns for grouping
        for cat_col in categorical_cols[:2]:
            if cat_col in df.columns:
                unique_values = df[cat_col].value_counts()
                context.append(f"\n{cat_col} categories: {dict(unique_values)}")
                
                # For text data, extract samples from each category
                if 'text' in df.columns:
                    context.append(f"\nSample data by {cat_col}:")
                    for category in unique_values.head(5).index:
                        samples = df[df[cat_col] == category]['text'].head(2)
                        context.append(f"- {category}: {list(samples)}")
                
                # For numeric data, show statistics by category
                numeric_in_relevant = [col for col in relevant_cols if col in df.select_dtypes(include=[np.number]).columns]
                if numeric_in_relevant:
                    context.append(f"\nStatistics by {cat_col}:")
                    for category in unique_values.head(3).index:
                        for num_col in numeric_in_relevant[:2]:
                            subset = df[df[cat_col] == category][num_col]
                            if not subset.empty:
                                context.append(f"- {category} {num_col}: mean={subset.mean():.2f}, count={len(subset)}")
        
        return context
    
    def _extract_temporal_context(self, df: pd.DataFrame, question: str, relevant_cols: list, numeric_cols: list) -> list:
        """Extract context for temporal/trend questions"""
        context = []
        
        # Look for date columns
        date_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
        
        if date_cols:
            date_col = date_cols[0]
            context.append(f"\nTemporal data: {date_col}")
            context.append(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
            
            # Add temporal statistics if numeric columns exist
            for num_col in numeric_cols[:2]:
                if num_col in df.columns:
                    df_sorted = df.sort_values(date_col)
                    context.append(f"- {num_col} trend: start={df_sorted[num_col].iloc[0]:.2f}, end={df_sorted[num_col].iloc[-1]:.2f}")
        else:
            context.append("\nNo explicit date columns found. Using record order for temporal analysis.")
            
        return context
    
    def _extract_correlation_context(self, df: pd.DataFrame, question: str, relevant_cols: list, numeric_cols: list) -> list:
        """Extract context for correlation/relationship questions"""
        context = []
        
        if len(numeric_cols) >= 2:
            # Calculate correlation matrix for relevant numeric columns
            numeric_relevant = [col for col in relevant_cols if col in numeric_cols][:3]
            if len(numeric_relevant) >= 2:
                corr_matrix = df[numeric_relevant].corr()
                context.append("\nCorrelation Analysis:")
                for i, col1 in enumerate(numeric_relevant):
                    for col2 in numeric_relevant[i+1:]:
                        corr_val = corr_matrix.loc[col1, col2]
                        context.append(f"- {col1} ↔ {col2}: correlation = {corr_val:.3f}")
        
        return context
    
    def _extract_distribution_context(self, df: pd.DataFrame, question: str, relevant_cols: list, categorical_cols: list) -> list:
        """Extract context for distribution questions"""
        context = []
        
        for cat_col in categorical_cols[:2]:
            if cat_col in df.columns:
                value_counts = df[cat_col].value_counts()
                total = len(df)
                context.append(f"\n{cat_col} distribution:")
                for value, count in value_counts.head(5).items():
                    percentage = (count / total) * 100
                    context.append(f"- {value}: {count} ({percentage:.1f}%)")
        
        return context
    
    def _extract_specific_value_context(self, df: pd.DataFrame, question: str, relevant_cols: list, numeric_cols: list, categorical_cols: list) -> list:
        """Extract context for specific value/ranking questions"""
        context = []
        
        # For numeric columns, show top/bottom values
        for num_col in numeric_cols[:2]:
            if num_col in df.columns:
                context.append(f"\n{num_col} values:")
                context.append(f"- Highest: {df[num_col].max():.2f}")
                context.append(f"- Lowest: {df[num_col].min():.2f}")
                context.append(f"- Average: {df[num_col].mean():.2f}")
                
                # Show top records
                top_records = df.nlargest(3, num_col)
                if not top_records.empty:
                    context.append(f"- Top records: {top_records[[col for col in relevant_cols if col in df.columns][:3]].to_dict('records')}")
        
        # For categorical columns, show most/least common
        for cat_col in categorical_cols[:2]:
            if cat_col in df.columns:
                value_counts = df[cat_col].value_counts()
                context.append(f"\n{cat_col} rankings:")
                context.append(f"- Most common: {value_counts.head(3).to_dict()}")
                context.append(f"- Least common: {value_counts.tail(2).to_dict()}")
        
        return context
    
    def _extract_comprehensive_context(self, df: pd.DataFrame, relevant_cols: list, numeric_cols: list, categorical_cols: list) -> list:
        """Extract comprehensive context for general questions"""
        context = []
        
        # Add detailed column information
        for col in relevant_cols[:4]:
            if col in numeric_cols:
                stats = df[col].describe()
                context.append(f"\n{col} (numeric): mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
                context.append(f"  - Distribution: Q1={stats['25%']:.2f}, Median={stats['50%']:.2f}, Q3={stats['75%']:.2f}")
            elif col in categorical_cols:
                value_counts = df[col].value_counts()
                context.append(f"\n{col} (categorical): {len(value_counts)} unique values")
                context.append(f"  - Top values: {dict(value_counts.head(3))}")
            elif col in df.columns:
                # Handle text or other columns
                sample_values = df[col].dropna().head(3).tolist()
                context.append(f"\n{col}: sample values = {sample_values}")
        
        # Add sample records
        if relevant_cols:
            sample_data = df[relevant_cols].head(3)
            context.append(f"\nSample records:\n{sample_data.to_string()}")
        
        return context
    
    def _is_data_driven_response(self, response: str) -> bool:
        """Check if the response is actually data-driven and not generic"""
        # Check for data-driven indicators
        data_indicators = [
            'shows', 'data reveals', 'records indicate', 'analysis shows',
            'found', 'observed', 'identified', 'discovered',
            '%', 'percent', 'average', 'mean', 'total', 'count',
            'highest', 'lowest', 'most', 'least', 'top', 'bottom'
        ]
        
        # Check for generic/evasive phrases that indicate poor analysis
        generic_phrases = [
            'no data provided', 'not sufficient', 'cannot determine',
            'would need', 'not available', 'not enough information',
            'does not contain', 'lacks information', 'insufficient data'
        ]
        
        response_lower = response.lower()
        
        # Reject if response contains generic evasive phrases
        if any(phrase in response_lower for phrase in generic_phrases):
            return False
        
        # Accept if response contains data-driven indicators
        return any(indicator in response_lower for indicator in data_indicators)
    
    def _format_analysis_response(self, response: str) -> str:
        """Format the analysis response into clean, readable bullet points"""
        try:
            formatted = "🔍 **Data Analysis Results**\n\n"
            
            # Clean and split the response
            lines = response.replace('\r', '').split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('=', '-', '_', '*' * 3)):
                    clean_lines.append(line)
            
            # Process lines into structured format
            current_section = []
            section_counter = 1
            
            for line in clean_lines:
                # Check if this is a main point/header
                if (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 
                    line.startswith(('**')) or
                    line.endswith(':') or
                    len(line) < 100 and not line.startswith(('The ', 'This ', 'It ', 'There '))):
                    
                    # Process previous section
                    if current_section:
                        formatted += self._format_clean_section(current_section, section_counter - 1) + "\n"
                    
                    # Start new section
                    current_section = [line]
                    section_counter += 1
                else:
                    # This is content for current section
                    current_section.append(line)
            
            # Process final section
            if current_section:
                formatted += self._format_clean_section(current_section, section_counter - 1)
            
            return formatted
            
        except Exception as e:
            print(f"⚠️ Formatting failed: {e}")
            return self._simple_bullet_format(response)
    
    def _format_clean_section(self, section_lines: list, section_num: int) -> str:
        """Format a section with proper bullet points and structure"""
        if not section_lines:
            return ""
        
        header = section_lines[0]
        content_lines = section_lines[1:] if len(section_lines) > 1 else []
        
        # Clean up header
        header = header.replace('**', '').replace('##', '').strip()
        if header.endswith(':'):
            header = header[:-1]
        
        # Format header based on content
        if header.startswith(('1.', '2.', '3.', '4.', '5.')):
            # Remove number prefix
            header = header.split('.', 1)[1].strip()
        
        formatted_header = f"📊 **{header}**\n"
        
        # Format content as clean bullet points
        formatted_content = []
        for line in content_lines:
            if line.strip():
                clean_line = line.strip()
                # Split long sentences into multiple bullet points
                if len(clean_line) > 150 and '. ' in clean_line:
                    sentences = clean_line.split('. ')
                    for sentence in sentences:
                        if len(sentence.strip()) > 20:
                            formatted_content.append(f"   • {sentence.strip()}{'.' if not sentence.endswith('.') else ''}")
                else:
                    formatted_content.append(f"   • {clean_line}")
        
        result = formatted_header
        if formatted_content:
            result += "\n".join(formatted_content) + "\n"
        
        return result
    
    def _simple_bullet_format(self, response: str) -> str:
        """Simple fallback formatting with bullet points"""
        formatted = "🔍 **Analysis Results**\n\n"
        
        # Split into sentences and create bullet points
        sentences = response.replace('\n', ' ').split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not sentence.startswith(('The analysis', 'This shows')):
                formatted += f"• {sentence}{'.' if not sentence.endswith('.') else ''}\n\n"
        
        return formatted
    
    def _format_section(self, section_lines: list) -> str:
        """Format a section of the response"""
        if not section_lines:
            return ""
        
        header = section_lines[0]
        content = section_lines[1:] if len(section_lines) > 1 else []
        
        # Format header
        if header.startswith(('1.', '2.', '3.', '4.')):
            # Numbered sections
            header_parts = header.split(' ', 1)
            if len(header_parts) == 2:
                number = header_parts[0]
                title = header_parts[1].replace('**', '').strip()
                formatted_header = f"📊 **{number} {title}**"
            else:
                formatted_header = f"📊 **{header}**"
        elif header.endswith(':'):
            # Section titles
            formatted_header = f"🔹 **{header.replace(':', '')}:**"
        else:
            formatted_header = f"• {header}"
        
        # Format content
        formatted_content = []
        for line in content:
            if line.strip():
                # Add bullet points for sub-items
                if not line.startswith(('•', '-', '*')):
                    formatted_content.append(f"  • {line}")
                else:
                    formatted_content.append(f"  {line}")
        
        result = formatted_header
        if formatted_content:
            result += "\n" + "\n".join(formatted_content)
        
        return result
    
    def _store_figure_for_export(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list):
        """Create and store a plotly figure that can be properly exported"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Determine the best visualization type
            question_lower = question.lower()
            
            # Create a proper figure based on data and question
            if any(word in question_lower for word in ['sentiment', 'distribution', 'category']):
                # Create sentiment distribution chart
                if 'airline_sentiment' in df.columns:
                    sentiment_counts = df['airline_sentiment'].value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values, 
                        names=sentiment_counts.index,
                        title="Airline Sentiment Distribution"
                    )
                else:
                    # Generic categorical distribution
                    if categorical_cols:
                        cat_col = categorical_cols[0]
                        value_counts = df[cat_col].value_counts().head(10)
                        fig = px.bar(
                            x=value_counts.index, 
                            y=value_counts.values,
                            title=f"Distribution of {cat_col}",
                            labels={'x': cat_col, 'y': 'Count'}
                        )
                    else:
                        fig = None
                        
            elif any(word in question_lower for word in ['compare', 'difference', 'between']):
                # Create comparison chart
                if categorical_cols and numeric_cols:
                    cat_col = categorical_cols[0]
                    num_col = numeric_cols[0]
                    grouped_data = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(10)
                    fig = px.bar(
                        x=grouped_data.index,
                        y=grouped_data.values,
                        title=f"Average {num_col} by {cat_col}",
                        labels={'x': cat_col, 'y': f'Average {num_col}'}
                    )
                elif categorical_cols and len(categorical_cols) >= 2:
                    # Cross-tabulation
                    crosstab = pd.crosstab(df[categorical_cols[0]], df[categorical_cols[1]])
                    fig = px.imshow(
                        crosstab.values,
                        x=crosstab.columns,
                        y=crosstab.index,
                        title=f"{categorical_cols[0]} vs {categorical_cols[1]}"
                    )
                else:
                    fig = None
                    
            elif numeric_cols:
                # Create numeric distribution
                num_col = numeric_cols[0]
                fig = px.histogram(
                    df, 
                    x=num_col, 
                    title=f"Distribution of {num_col}",
                    nbins=30
                )
                
            else:
                fig = None
                
            # Store the figure in the state for export
            if fig:
                if not hasattr(self, 'current_figure'):
                    self.current_figure = fig
                else:
                    self.current_figure = fig
                    
        except Exception as e:
            print(f"⚠️ Could not create exportable figure: {e}")
    
    def _find_relevant_columns_llm(self, question: str, numeric_cols: list, categorical_cols: list, df: pd.DataFrame) -> list:
        """Use LLM to intelligently find relevant columns for the question"""
        if not self.llm or not self.llm.is_available():
            return self._smart_column_fallback(question, numeric_cols, categorical_cols, df)
        
        try:
            print(f"🎯 Finding relevant columns for: {question[:50]}...")
            
            columns_info = []
            # Provide rich column info to LLM
            for col in numeric_cols[:8]:  # Limit for prompt size
                stats = df[col].describe()
                columns_info.append(f"- {col} (numeric): mean={stats['mean']:.2f}, range=[{stats['min']:.1f}, {stats['max']:.1f}]")
            
            for col in categorical_cols[:8]:
                unique_count = df[col].nunique()
                top_vals = df[col].value_counts().head(3).index.tolist()
                columns_info.append(f"- {col} (categorical): {unique_count} unique values, top: {top_vals}")
            
            prompt = f"""You are analyzing this specific question: "{question}"

Which 2-3 columns are MOST relevant to answer this question?

AVAILABLE COLUMNS:
{chr(10).join(columns_info)}

Think about:
- What specific data does the question ask about?
- Which columns contain that information?
- What comparisons or calculations are needed?

Respond with ONLY the most relevant column names, comma-separated:"""
            
            response = self.llm(prompt).strip()
            print(f"🎯 LLM suggests columns: {response}")
            
            # Parse and validate
            suggested_cols = [col.strip().strip('"\'"`') for col in response.split(',')]
            all_cols = numeric_cols + categorical_cols
            relevant = [col for col in suggested_cols if col in all_cols]
            
            if relevant:
                print(f"✅ Using relevant columns: {relevant}")
                return relevant[:4]
            else:
                print("⚠️ LLM suggestions invalid, using smart fallback")
                return self._smart_column_fallback(question, numeric_cols, categorical_cols, df)
            
        except Exception as e:
            print(f"⚠️ LLM column selection failed: {e}")
            return self._smart_column_fallback(question, numeric_cols, categorical_cols, df)
    
    def _smart_column_fallback(self, question: str, numeric_cols: list, categorical_cols: list, df: pd.DataFrame) -> list:
        """Smart fallback for column selection when LLM unavailable"""
        question_lower = question.lower()
        all_cols = numeric_cols + categorical_cols
        relevant = []
        
        # 1. Direct column name matches
        for col in all_cols:
            if col.lower() in question_lower or any(word in col.lower() for word in question_lower.split()):
                relevant.append(col)
        
        # 2. Question type-based selection
        if not relevant:
            if any(word in question_lower for word in ['compare', 'versus', 'difference', 'between', 'across']):
                if categorical_cols and numeric_cols:
                    relevant = [categorical_cols[0], numeric_cols[0]]
            elif any(word in question_lower for word in ['correlation', 'relationship', 'associated']):
                relevant = numeric_cols[:3]
            elif any(word in question_lower for word in ['distribution', 'spread', 'range']):
                relevant = numeric_cols[:2]
            elif any(word in question_lower for word in ['average', 'mean', 'total', 'sum']):
                relevant = numeric_cols[:2]
            else:
                # Mixed selection
                if categorical_cols:
                    relevant.append(categorical_cols[0])
                if numeric_cols:
                    relevant.extend(numeric_cols[:2])
        
        # 3. Ensure we have something
        if not relevant:
            relevant = all_cols[:3]
        
        return relevant[:4]
    
    def _perform_analysis(self, df: pd.DataFrame, question: str, analysis_type: str, relevant_cols: list, numeric_cols: list, categorical_cols: list) -> dict:
        """Perform the actual analysis based on type and columns"""
        result = {
            "question": question,
            "answer": "",
            "visualization_html": None,
            "method": "intelligent_analysis"
        }
        
        try:
            if analysis_type == "correlation" and len([col for col in relevant_cols if col in numeric_cols]) >= 2:
                numeric_relevant = [col for col in relevant_cols if col in numeric_cols]
                corr_matrix = df[numeric_relevant].corr()
                result["answer"] = self._generate_correlation_insights(corr_matrix, question)
                # Use intelligent visualization decision
                if self._should_create_visualization(question):
                    result["visualization_html"] = plot_correlation_matrix(df[numeric_relevant])
                
            elif analysis_type == "distribution":
                target_col = next((col for col in relevant_cols if col in numeric_cols), numeric_cols[0] if numeric_cols else None)
                if target_col:
                    result["answer"] = self._generate_distribution_insights(df[target_col], target_col, question)
                    # Use intelligent visualization decision
                    if self._should_create_visualization(question):
                        result["visualization_html"] = plot_histogram(df, target_col)
                    
            elif analysis_type == "comparison":
                cat_col = next((col for col in relevant_cols if col in categorical_cols), categorical_cols[0] if categorical_cols else None)
                num_col = next((col for col in relevant_cols if col in numeric_cols), numeric_cols[0] if numeric_cols else None)
                if cat_col and num_col:
                    grouped = df.groupby(cat_col)[num_col].agg(['mean', 'count', 'std']).reset_index()
                    result["answer"] = self._generate_comparison_insights(grouped, cat_col, num_col, question)
                    # Use intelligent visualization decision
                    if self._should_create_visualization(question):
                        result["visualization_html"] = plot_bar(df.groupby(cat_col)[num_col].mean().reset_index(), cat_col, num_col)
                    
            elif analysis_type == "aggregation":
                target_col = next((col for col in relevant_cols if col in numeric_cols), numeric_cols[0] if numeric_cols else None)
                if target_col:
                    stats = df[target_col].describe()
                    result["answer"] = f"📊 Summary statistics for {target_col}:\n" + \
                                     f"• Mean: {stats['mean']:.2f}\n" + \
                                     f"• Median: {stats['50%']:.2f}\n" + \
                                     f"• Std Dev: {stats['std']:.2f}\n" + \
                                     f"• Range: {stats['min']:.2f} to {stats['max']:.2f}"
                    # Use intelligent visualization decision - aggregation usually doesn't need viz
                    if self._should_create_visualization(question):
                        result["visualization_html"] = plot_histogram(df, target_col)
                    
            else:
                # Fallback to basic analysis
                if relevant_cols and relevant_cols[0] in numeric_cols:
                    target_col = relevant_cols[0]
                    result["answer"] = f"📊 Basic analysis of {target_col}:\n" + \
                                     f"• Average: {df[target_col].mean():.2f}\n" + \
                                     f"• Total records: {len(df)}\n" + \
                                     f"• Missing values: {df[target_col].isnull().sum()}"
                    # Use intelligent visualization decision for fallback too
                    if self._should_create_visualization(question):
                        result["visualization_html"] = plot_histogram(df, target_col)
                    
        except Exception as e:
            print(f"⚠️ Analysis execution failed: {e}")
            result["answer"] = f"Could not perform {analysis_type} analysis: {str(e)}"
            
        return result
    
    def _generate_llm_insights(self, df: pd.DataFrame, question: str, result: dict, relevant_cols: list) -> str:
        """Use LLM to enhance insights with natural language understanding"""
        try:
            # Simplified approach to avoid garbled output
            # Just use the statistical results we already generated
            base_answer = result.get("answer", "")
            
            # Only enhance if we have a good base answer and LLM is working
            if not base_answer or "could not perform" in base_answer.lower():
                return base_answer
            
            # Create a very simple, focused prompt
            simple_prompt = f"""Explain this data analysis result in simple terms:

Question: {question}
Result: {base_answer[:200]}

Write a brief explanation in 2-3 sentences about what this means."""
            
            llm_response = self.llm(simple_prompt)
            
            # Strict validation of LLM response
            if (llm_response and 
                len(llm_response.strip()) > 10 and 
                len(llm_response.strip()) < 500 and
                not any(char.isdigit() and llm_response.count(char) > 10 for char in "0123456789")):
                
                # Clean response
                clean_response = llm_response.strip()
                # Remove any data-like patterns (sequences of numbers)
                import re
                if not re.search(r'\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d+', clean_response):
                    return f"{base_answer}\n\n💡 Insight: {clean_response}"
            
            # Fallback to base answer if LLM response looks garbled
            print(f"🔄 Using statistical analysis (LLM response filtered out)")
            return base_answer
                
        except Exception as e:
            print(f"⚠️ LLM insight generation failed: {e}")
            return result.get("answer", "Analysis could not be completed")
    
    
    def _generate_correlation_insights(self, corr_matrix, question: str) -> str:
        """Generate intelligent correlation insights"""
        strong_corr = corr_matrix.abs() > 0.5
        insights = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j and strong_corr.iloc[i, j]:
                    corr_val = corr_matrix.iloc[i, j]
                    strength = "strong" if abs(corr_val) > 0.7 else "moderate"
                    direction = "positive" if corr_val > 0 else "negative"
                    insights.append(f"{col1} has a {strength} {direction} correlation ({corr_val:.3f}) with {col2}")
        
        if insights:
            return "📊 Correlation Analysis:\n" + "\n".join(insights[:3])
        else:
            return "📊 No strong correlations found between variables."
    
    def _generate_distribution_insights(self, series: pd.Series, col_name: str, question: str) -> str:
        """Generate intelligent distribution insights"""
        mean_val = series.mean()
        median_val = series.median()
        std_val = series.std()
        skewness = series.skew()
        
        insights = [f"📈 Distribution Analysis for {col_name}:"]
        insights.append(f"• Mean: {mean_val:.2f}, Median: {median_val:.2f}")
        
        if abs(skewness) > 1:
            direction = "right" if skewness > 0 else "left"
            insights.append(f"• Highly skewed to the {direction} (skewness: {skewness:.2f})")
        elif abs(skewness) > 0.5:
            direction = "right" if skewness > 0 else "left"
            insights.append(f"• Moderately skewed to the {direction}")
        else:
            insights.append("• Approximately normal distribution")
        
        # Outliers
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
        if len(outliers) > 0:
            insights.append(f"• Contains {len(outliers)} outliers ({len(outliers)/len(series)*100:.1f}% of data)")
        
        return "\n".join(insights)
    
    def _generate_comparison_insights(self, grouped_data, cat_col: str, num_col: str, question: str) -> str:
        """Generate intelligent comparison insights"""
        insights = [f"📊 Comparison Analysis: {num_col} by {cat_col}"]
        
        # Sort by mean value
        sorted_data = grouped_data.sort_values('mean', ascending=False)
        
        highest = sorted_data.iloc[0]
        lowest = sorted_data.iloc[-1]
        
        insights.append(f"• Highest average: {highest[cat_col]} ({highest['mean']:.2f})")
        insights.append(f"• Lowest average: {lowest[cat_col]} ({lowest['mean']:.2f})")
        
        # Variation analysis
        overall_std = grouped_data['mean'].std()
        if overall_std > grouped_data['mean'].mean() * 0.1:  # 10% threshold
            insights.append(f"• Significant variation across {cat_col} groups")
        else:
            insights.append(f"• Relatively consistent across {cat_col} groups")
        
        return "\n".join(insights)
    
    def _custom_llm_analysis(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> dict:
        """Let LLM perform custom analysis when standard patterns don't apply"""
        if not self.llm:
            return self._fallback_analysis(df, question, numeric_cols, categorical_cols)
        
        # Create rich context for LLM
        context = f"""
        QUESTION: {question}
        
        DATA SUMMARY:
        - Shape: {df.shape}
        - Numeric columns: {numeric_cols[:5]}
        - Categorical columns: {categorical_cols[:5]}
        
        KEY STATISTICS:
        {df.describe().to_string() if numeric_cols else 'No numeric columns'}
        
        SAMPLE DATA:
        {df.head(3).to_string()}
        """
        
        analysis_prompt = f"""
        You are an expert data analyst. Analyze this dataset to answer the user's question.
        
        {context}
        
        Provide:
        1. Direct answer to the question with specific numbers/findings
        2. Deep Understanding of the question. 
        3. Key insights and patterns you discovered
        4. Any notable observations or recommendations
        
        Be specific, quantitative, qualitiative, and actionable. Focus on what the data actually shows.
        """
        
        try:
            llm_response = self.llm(analysis_prompt)
            return {
                "question": question,
                "answer": f"🤖 AI Analysis:\n{llm_response}",
                "visualization_html": self._suggest_visualization(df, question, numeric_cols, categorical_cols),
                "method": "custom_llm_analysis"
            }
        except Exception as e:
            return {
                "question": question,
                "answer": f"Custom analysis failed: {e}",
                "visualization_html": None,
                "method": "error"
            }
    
    def _suggest_visualization(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> str:
        """Use LLM to intelligently decide if visualization is needed and what type"""
        try:
            # First, ask LLM if visualization is even needed
            if not self._should_create_visualization(question):
                print(f"🧠 LLM decided: No visualization needed for this question")
                return None
            
            # If visualization is needed, determine the best type
            viz_type = self._determine_visualization_type(question, numeric_cols, categorical_cols)
            print(f"📊 LLM suggested visualization: {viz_type}")
            
            # Create the appropriate visualization
            return self._create_visualization(df, viz_type, question, numeric_cols, categorical_cols)
            
        except Exception as e:
            print(f"⚠️ Intelligent visualization creation failed: {e}")
            return None
    
    def _should_create_visualization(self, question: str) -> bool:
        """Use LLM to determine if this question needs a visualization - default to YES for most analytical questions"""
        q_lower = question.lower()
        
        # Questions that definitely DON'T need visualization
        no_viz_keywords = ['how many rows', 'column names', 'columns', 'shape of', 'size of', 
                          'data types', 'info about', 'describe the dataset structure']
        if any(keyword in q_lower for keyword in no_viz_keywords):
            return False
        
        # Questions that DEFINITELY need visualization (aggressive defaults)
        viz_keywords = ['show', 'plot', 'chart', 'graph', 'visualize', 'distribution', 
                       'correlation', 'trend', 'pattern', 'compare', 'comparison',
                       'analyze', 'analysis', 'find', 'calculate', 'what', 'how',
                       'top', 'bottom', 'highest', 'lowest', 'average', 'mean',
                       'performance', 'rate', 'rates', 'across', 'by', 'which',
                       'best', 'worst', 'relationship', 'between', 'among',
                       'exploratory', 'insight', 'insights', 'breakdown']
        
        # Default to YES for most questions - be very generous
        if any(keyword in q_lower for keyword in viz_keywords):
            return True
        
        # Use LLM only if basic keyword detection doesn't match
        if self.llm and self.llm.is_available():
            try:
                prompt = f"""Should this data analysis question include a chart or graph?

Question: "{question}"

✅ Most questions about data analysis benefit from visualizations.
✅ Create charts for: comparisons, averages, distributions, trends, performance metrics, top/bottom rankings.
❌ Skip charts only for: basic info like row count, column names, data structure questions.

Answer: YES or NO"""
                
                response = self.llm(prompt).strip().upper()
                
                # Be VERY generous - if there's any YES indication, create viz
                if 'YES' in response:
                    return True
                elif 'NO' in response:
                    # Double-check: is this really a data analysis question?
                    return len([word for word in ['average', 'mean', 'compare', 'distribution', 
                                                'performance', 'rate', 'top', 'bottom', 'by']
                               if word in q_lower]) >= 1
                else:
                    # Unclear response - default to YES for safety
                    return True
                    
            except Exception as e:
                print(f"⚠️ LLM visualization decision failed: {e}")
                # Generous fallback - default to YES unless clearly not needed
                return True
        
        # Final fallback - be generous, default to YES
        return True
    
    def _determine_visualization_type(self, question: str, numeric_cols: list, categorical_cols: list) -> str:
        """Use LLM to determine the best visualization type"""
        if not self.llm or not self.llm.is_available():
            # Simple fallback logic
            q_lower = question.lower()
            if "correlation" in q_lower and len(numeric_cols) >= 2:
                return "correlation"
            elif "distribution" in q_lower and numeric_cols:
                return "distribution"
            elif ("compare" in q_lower or "difference" in q_lower) and categorical_cols and numeric_cols:
                return "comparison"
            elif numeric_cols:
                return "distribution"
            else:
                return "summary"
        
        try:
            viz_options = [
                "histogram - for showing distribution of a single numeric variable",
                "bar_chart - for comparing values across categories", 
                "correlation_matrix - for showing relationships between multiple numeric variables",
                "scatter_plot - for showing relationship between two numeric variables",
                "line_chart - for showing trends over time",
                "summary - for basic statistics that don't need a chart"
            ]
            
            prompt = f"""What type of visualization best answers this question?

Question: "{question}"

Available data:
- Numeric columns: {numeric_cols[:5] if numeric_cols else 'None'}
- Categorical columns: {categorical_cols[:5] if categorical_cols else 'None'}

Visualization options:
{chr(10).join(viz_options)}

Choose the SINGLE best option. Respond with just the option name (e.g., "histogram", "bar_chart", etc.)"""
            
            response = self.llm(prompt).strip().lower()
            
            # Map LLM response to our visualization types
            viz_mapping = {
                'histogram': 'distribution',
                'bar_chart': 'comparison', 
                'bar': 'comparison',
                'correlation_matrix': 'correlation',
                'correlation': 'correlation',
                'scatter_plot': 'scatter',
                'scatter': 'scatter',
                'line_chart': 'trend',
                'line': 'trend',
                'summary': 'summary'
            }
            
            for key, value in viz_mapping.items():
                if key in response:
                    return value
            
            # Fallback
            return 'distribution' if numeric_cols else 'summary'
            
        except Exception as e:
            print(f"⚠️ LLM visualization type detection failed: {e}")
            return 'distribution' if numeric_cols else 'summary'
    
    def _create_visualization(self, df: pd.DataFrame, viz_type: str, question: str, numeric_cols: list, categorical_cols: list) -> str:
        """Use LLM to create intelligent, question-specific visualizations"""
        try:
            print(f"🧠 Using LLM to create intelligent visualization for: {question[:60]}...")
            
            # Get LLM recommendation for visualization
            viz_recommendation = self._get_llm_visualization_recommendation(df, question, numeric_cols, categorical_cols)
            
            # Create the recommended visualization
            return self._create_llm_recommended_visualization(df, viz_recommendation, question, numeric_cols, categorical_cols)
            
        except Exception as e:
            print(f"⚠️ LLM visualization creation failed: {e}")
            # Simple fallback
            try:
                from utils.plotting_utils import plot_histogram, plot_bar
                if categorical_cols and numeric_cols:
                    # Group by first categorical, average first numeric
                    cat_col = categorical_cols[0]
                    num_col = numeric_cols[0] 
                    grouped = df.groupby(cat_col)[num_col].mean().reset_index()
                    return plot_bar(grouped, cat_col, num_col)
                elif numeric_cols:
                    return plot_histogram(df, numeric_cols[0])
                return None
            except:
                return None
    
    def _get_llm_visualization_recommendation(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> dict:
        """Use LLM to intelligently recommend the best visualization for the question"""
        if not self.llm or not self.llm.is_available():
            # Fallback without LLM
            return {
                'type': 'bar_chart' if categorical_cols and numeric_cols else 'histogram',
                'columns': [categorical_cols[0] if categorical_cols else None, numeric_cols[0] if numeric_cols else None],
                'reasoning': 'Basic recommendation (LLM unavailable)'
            }
        
        try:
            # Create data context for LLM
            sample_data = df.head(3).to_string() if len(df) > 0 else "No data"
            
            prompt = f"""You are a data visualization expert. Analyze this question and dataset to recommend the perfect visualization.

QUESTION: "{question}"

DATASET INFO:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Sample data:
{sample_data}

VISUALIZATION OPTIONS:
1. bar_chart - Compare values across categories
2. histogram - Show distribution of single variable
3. scatter_plot - Show relationship between two variables
4. line_chart - Show trends over time or sequence
5. pie_chart - Show parts of a whole
6. box_plot - Show distribution with quartiles
7. heatmap - Show correlation matrix
8. multiple_charts - Dashboard with 2-4 related charts

TASK: Recommend the BEST visualization that directly answers the question.

Respond in this EXACT format:
TYPE: [chart_type]
COLUMNS: [column1, column2, etc.]
REASONING: [why this visualization best answers the question]
MESSAGE: [what you'll tell the user about why this visualization helps]

Example:
TYPE: bar_chart
COLUMNS: [device, conversion_rate]
REASONING: Question asks about performance by device type, bar chart clearly shows conversion differences
MESSAGE: A bar chart will clearly show how conversion rates vary across different devices, making it easy to identify which device performs best."""
            
            response = self.llm(prompt)
            
            # Parse LLM response
            return self._parse_viz_recommendation(response)
            
        except Exception as e:
            print(f"⚠️ LLM viz recommendation failed: {e}")
            # Fallback
            return {
                'type': 'bar_chart' if categorical_cols and numeric_cols else 'histogram',
                'columns': [categorical_cols[0] if categorical_cols else None, numeric_cols[0] if numeric_cols else None],
                'reasoning': 'Fallback recommendation (LLM error)',
                'message': 'Using a standard chart to visualize the data.'
            }
    
    def _parse_viz_recommendation(self, llm_response: str) -> dict:
        """Parse LLM visualization recommendation response"""
        try:
            lines = llm_response.strip().split('\n')
            result = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('TYPE:'):
                    result['type'] = line.replace('TYPE:', '').strip().lower()
                elif line.startswith('COLUMNS:'):
                    columns_str = line.replace('COLUMNS:', '').strip()
                    # Parse [col1, col2] or col1, col2 and remove quotes/backticks
                    columns_str = columns_str.strip('[]')
                    raw_cols = [col.strip() for col in columns_str.split(',') if col.strip()]
                    cleaned_cols = [col.strip("'\"` ") for col in raw_cols]
                    result['columns'] = cleaned_cols
                elif line.startswith('REASONING:'):
                    result['reasoning'] = line.replace('REASONING:', '').strip()
                elif line.startswith('MESSAGE:'):
                    result['message'] = line.replace('MESSAGE:', '').strip()
            
            # Ensure we have required fields
            if 'type' not in result:
                result['type'] = 'bar_chart'
            if 'message' not in result:
                result['message'] = 'This visualization will help answer your question.'
                
            print(f"📊 LLM recommends: {result['type']} - {result.get('reasoning', 'No reasoning provided')}")
            return result
            
        except Exception as e:
            print(f"⚠️ Failed to parse LLM recommendation: {e}")
            return {
                'type': 'bar_chart',
                'columns': [],
                'reasoning': 'Parse error',
                'message': 'Using a standard visualization.'
            }
    
    def _create_llm_recommended_visualization(self, df: pd.DataFrame, recommendation: dict, question: str, numeric_cols: list, categorical_cols: list) -> str:
        """Create visualization based on LLM recommendation and store Plotly JSON for reporting"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import plotly.io as pio
            
            viz_type = recommendation.get('type', 'bar_chart')
            columns = recommendation.get('columns', [])
            message = recommendation.get('message', 'Visualization created.')
            
            print(f"🎨 Creating {viz_type} with columns: {columns}")
            print(f"💡 LLM says: {message}")
            
            # Clean and normalize column names
            normalized_map = {str(c).lower(): c for c in df.columns}
            candidate_cols = [c for c in (col.strip("'\"` ") for col in columns) if c]
            mapped_cols = [normalized_map.get(c.lower()) for c in candidate_cols if normalized_map.get(c.lower())]
            valid_columns = [col for col in mapped_cols if col]
            
            def is_true_numeric(col_name: str) -> bool:
                try:
                    import pandas as pd
                    return pd.api.types.is_numeric_dtype(df[col_name]) and not pd.api.types.is_datetime64_any_dtype(df[col_name])
                except Exception:
                    return False
            def is_categorical(col_name: str) -> bool:
                try:
                    import pandas as pd
                    return pd.api.types.is_categorical_dtype(df[col_name]) or df[col_name].dtype == 'object'
                except Exception:
                    return False
            
            fig = None
            # Build figures by dtype and intent
            if viz_type == 'bar_chart':
                cat_col = next((c for c in valid_columns if is_categorical(c)), (categorical_cols[0] if categorical_cols else None))
                num_col = next((c for c in valid_columns if is_true_numeric(c)), None)
                ql = question.lower()
                count_requested = any(k in ql for k in ['count', 'frequency', 'frequencies', 'occurrence', 'occurrences']) or any(str(c).lower() == 'count' for c in columns)
                if cat_col and num_col and not count_requested:
                    grouped = df.groupby(cat_col)[num_col].mean().reset_index()
                    fig = px.bar(grouped, x=cat_col, y=num_col, title=f"Average {num_col} by {cat_col}")
                elif cat_col:
                    top_n = self._extract_top_n_from_question(question, default=10)
                    counts = df[cat_col].value_counts().head(top_n).reset_index()
                    counts.columns = [cat_col, 'count']
                    title = f"Top {top_n} {cat_col} by count" if top_n else f"Count of {cat_col}"
                    fig = px.bar(counts, x=cat_col, y='count', title=title)
            elif viz_type == 'histogram':
                candidates = [c for c in valid_columns if is_true_numeric(c)] or [c for c in numeric_cols if is_true_numeric(c)]
                if candidates:
                    target_col = candidates[0]
                    fig = px.histogram(df, x=target_col, nbins=30, title=f"Distribution of {target_col}")
                elif categorical_cols:
                    cat_col = categorical_cols[0]
                    counts = df[cat_col].value_counts().head(self._extract_top_n_from_question(question, 10)).reset_index()
                    counts.columns = [cat_col, 'count']
                    fig = px.bar(counts, x=cat_col, y='count', title=f"Count of {cat_col}")
            elif viz_type == 'scatter_plot':
                num_list = [c for c in valid_columns if is_true_numeric(c)]
                if len(num_list) < 2:
                    num_list = [c for c in numeric_cols if is_true_numeric(c)][:2]
                if len(num_list) >= 2:
                    color_by = next((c for c in valid_columns if is_categorical(c)), (categorical_cols[0] if categorical_cols else None))
                    fig = px.scatter(df, x=num_list[0], y=num_list[1], color=color_by, title=f"Scatter: {num_list[0]} vs {num_list[1]}")
            elif viz_type == 'pie_chart' and categorical_cols:
                cat_col = next((c for c in valid_columns if is_categorical(c)), categorical_cols[0])
                value_counts = df[cat_col].value_counts().head(10)
                fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Distribution of {cat_col}")
            elif viz_type == 'box_plot':
                num_col = next((c for c in valid_columns if is_true_numeric(c)), (numeric_cols[0] if numeric_cols else None))
                cat_col = next((c for c in valid_columns if is_categorical(c)), None)
                if num_col and cat_col:
                    fig = px.box(df, x=cat_col, y=num_col, title=f"Box Plot of {num_col} by {cat_col}")
                elif num_col:
                    fig = px.box(df, y=num_col, title=f"Box Plot of {num_col}")
            elif viz_type == 'heatmap' and len(numeric_cols) >= 2:
                import numpy as np
                num_subset = [c for c in numeric_cols if is_true_numeric(c)][:8]
                if len(num_subset) >= 2:
                    corr = df[num_subset].corr()
                    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu', zmid=0))
                    fig.update_layout(title='Correlation Heatmap')
            elif viz_type == 'multiple_charts':
                # Simple default: histogram of first numeric
                if numeric_cols:
                    target_col = next((c for c in numeric_cols if is_true_numeric(c)), None)
                    if target_col:
                        fig = px.histogram(df, x=target_col, nbins=30, title=f"Distribution of {target_col}")
            
            if fig is None:
                # Final fallback by available dtypes
                if categorical_cols and numeric_cols:
                    grouped = df.groupby(categorical_cols[0])[numeric_cols[0]].mean().reset_index()
                    fig = px.bar(grouped, x=categorical_cols[0], y=numeric_cols[0], title=f"Average {numeric_cols[0]} by {categorical_cols[0]}")
                elif numeric_cols:
                    fig = px.histogram(df, x=numeric_cols[0], nbins=30, title=f"Distribution of {numeric_cols[0]}")
                else:
                    return None
            
            # Store JSON for reports
            self.last_figure_json = pio.to_json(fig)
            return fig.to_html(full_html=False)
            
        except Exception as e:
            print(f"⚠️ Failed to create recommended visualization: {e}")
            return None
    
    def _create_simple_dashboard(self, df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> str:
        """Create a simple multi-chart dashboard"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create 2x2 subplot dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Distribution', 'Comparison', 'Relationship', 'Summary'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Chart 1: Distribution (top-left)
            if numeric_cols:
                num_col = numeric_cols[0]
                fig.add_trace(
                    go.Histogram(x=df[num_col], name=f'{num_col} Distribution', opacity=0.7),
                    row=1, col=1
                )
            
            # Chart 2: Comparison (top-right)
            if categorical_cols and numeric_cols:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                grouped = df.groupby(cat_col)[num_col].mean().head(8)  # Top 8
                fig.add_trace(
                    go.Bar(x=grouped.index, y=grouped.values, name=f'{num_col} by {cat_col}'),
                    row=1, col=2
                )
            
            # Chart 3: Relationship (bottom-left)
            if len(numeric_cols) >= 2:
                fig.add_trace(
                    go.Scatter(x=df[numeric_cols[0]], y=df[numeric_cols[1]], 
                              mode='markers', name='Relationship', opacity=0.6),
                    row=2, col=1
                )
            
            # Chart 4: Summary stats (bottom-right)
            if numeric_cols:
                stats = df[numeric_cols[0]].describe()
                fig.add_trace(
                    go.Bar(x=['Mean', 'Median', 'Std'], 
                          y=[stats['mean'], stats['50%'], stats['std']], 
                          name='Summary Stats'),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                title_text="Data Overview Dashboard",
                showlegend=True
            )
            
            return fig.to_html(full_html=False)
            
        except Exception as e:
            print(f"⚠️ Dashboard creation failed: {e}")
            return None
    
    def _get_visualization_message(self, question: str) -> str:
        """Generate a message explaining why visualization helps answer the question"""
        try:
            if not self.llm or not self.llm.is_available():
                return "A visualization has been created to help illustrate the data patterns and make the analysis easier to understand."
            
            prompt = f"""Explain briefly why a visualization would help answer this question better than just numbers.

Question: "{question}"

Write 1-2 sentences explaining:
1. Why visual representation helps
2. What insights the chart makes clearer

Be specific about how the visualization adds value beyond just the statistical analysis.

Example: "A bar chart makes it easy to visually compare performance across different categories and quickly identify which ones stand out, rather than just reading through a list of numbers."

Response:"""
            
            response = self.llm(prompt)
            
            if response and len(response.strip()) > 20:
                return f"**Why this visualization helps:** {response.strip()}"
            else:
                return "This visualization makes the data patterns more intuitive and easier to interpret than numerical results alone."
                
        except Exception as e:
            print(f"⚠️ Visualization message generation failed: {e}")
            return "A visualization has been created to complement the analysis and make the insights more accessible."
            
            # Distribution analysis - enhanced with statistical overlays
            if viz_type == 'distribution' and numeric_cols:
                target_col = self._find_target_column(question, numeric_cols)
                group_col = None
                
                # Check if question mentions grouping
                if any(word in question_lower for word in ['by', 'across', 'different', 'each']):
                    group_col = self._find_target_column(question, categorical_cols)
                    
                return plot_distribution_advanced(df, target_col, group_col)
                
            # Advanced correlation analysis
            elif viz_type == 'correlation' and len(numeric_cols) >= 2:
                return plot_correlation_matrix(df)
                
            # Advanced scatter plot with regression analysis
            elif viz_type == 'scatter' and len(numeric_cols) >= 2:
                x_col = self._find_target_column(question, numeric_cols)
                y_cols = [col for col in numeric_cols if col != x_col]
                y_col = y_cols[0] if y_cols else numeric_cols[1]
                
                # Check for additional dimensions
                color_by = None
                size_by = None
                
                if categorical_cols and len(categorical_cols) > 0:
                    color_by = categorical_cols[0]
                    
                if len(numeric_cols) > 2:
                    remaining_numeric = [col for col in numeric_cols if col not in [x_col, y_col]]
                    if remaining_numeric:
                        size_by = remaining_numeric[0]
                
                return plot_scatter(df, x_col, y_col, color_by, size_by)
                
            # Advanced comparison/bar chart
            elif viz_type == 'comparison' and categorical_cols and numeric_cols:
                cat_col = self._find_target_column(question, categorical_cols)
                num_col = self._find_target_column(question, numeric_cols)
                
                return plot_categorical_analysis(df, cat_col, num_col)
                
            # Correlation between categorical and numeric (special case)
            elif viz_type == 'correlation' and categorical_cols and numeric_cols:
                cat_col = self._find_target_column(question, categorical_cols)
                num_col = self._find_target_column(question, numeric_cols)
                
                return plot_categorical_analysis(df, cat_col, num_col)
                
            # Time series analysis
            elif viz_type == 'trend' and numeric_cols:
                target_col = self._find_target_column(question, numeric_cols)
                time_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'year', 'month', 'day'])]
                
                if time_cols:
                    return plot_timeseries(df, time_cols[0], target_col)
                else:
                    # Use index as time if no time column found
                    return plot_line(df.reset_index(), 'index', target_col)
                    
            # Fallback to basic histogram for numeric questions
            elif numeric_cols:
                target_col = self._find_target_column(question, numeric_cols)
                return plot_histogram(df, target_col)
                    
            else:
                # viz_type == 'summary' or fallback - no visualization
                return None
                
        except Exception as e:
            print(f"⚠️ Advanced visualization creation failed: {e}")
            # Fallback to basic visualization
            try:
                if numeric_cols:
                    return plot_histogram(df, numeric_cols[0])
                return None
            except:
                return None
    
    def _find_target_column(self, question: str, columns: list) -> str:
        """Find the most relevant column for the question"""
        if not columns:
            return None
        
        q_lower = question.lower()
        
        # Look for column names mentioned in the question
        for col in columns:
            if col.lower() in q_lower:
                return col
        
        # Return first column as fallback
        return columns[0]
    
    def _fallback_analysis(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> dict:
        """Fallback analysis when LLM is not available"""
        # Even in fallback, use simple keyword-based visualization decision
        viz_html = None
        if numeric_cols and self._should_create_visualization(question):
            viz_html = plot_histogram(df, numeric_cols[0])
        
        return {
            "question": question,
            "answer": f"Basic analysis: Dataset has {df.shape[0]} rows and {df.shape[1]} columns. Available for analysis: {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns.",
            "visualization_html": viz_html,
            "method": "fallback_analysis"
        }

    # Old methods removed - now using intelligent LLM-driven analysis

    # ================= MAIN ENTRY =================
    def answer(self, data, question: str, dataset_name: str = None) -> dict:
        """
        Unified entrypoint:
        - Structured (DataFrame)
        - Text (list[str])
        """
        if isinstance(data, pd.DataFrame):
            return self._answer_structured(data, question, dataset_name)
        elif isinstance(data, list) and all(isinstance(d, str) for d in data):
            return self._answer_text(data, question, dataset_name)
        else:
            return {
                "answer": "⚠️ Unsupported data type. Provide DataFrame or list[str].",
                "visualization_html": None,
            }

    # ================= STRUCTURED =================
    def _answer_structured(self, df: pd.DataFrame, question: str, dataset_name: str = None) -> dict:
        # Ensure dataset is cached for performance
        if dataset_name and dataset_name not in self.dataset_cache:
            self.set_dataset(dataset_name, df)
        
        # Get column information
        if dataset_name and dataset_name in self.dataset_cache:
            cache = self.dataset_cache[dataset_name]
            numeric_cols, categorical_cols = cache["numeric_cols"], cache["categorical_cols"]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # 🚀 Use intelligent LLM-driven analysis instead of primitive keyword matching
        analysis_result = self._intelligent_analysis(df, question, numeric_cols, categorical_cols)

        # Store in history
        if dataset_name and dataset_name in self.analysis_history:
            self.analysis_history[dataset_name].append({
                "question": question,
                "result": analysis_result,
                "timestamp": pd.Timestamp.now(),
            })

        return analysis_result

    # ================= TEXT =================
    def _answer_text(self, docs: list[str], question: str, dataset_name: str = None) -> dict:
        q_lower = question.lower()
        answer, viz_html = f"📝 Text Analysis for: '{question}'\n", None

        try:
            if any(word in q_lower for word in ["word", "keyword", "frequency"]):
                viz_html = plot_word_frequencies(docs)
                answer += "📊 Word frequency plot generated."
            elif any(word in q_lower for word in ["cloud", "overview", "summary"]):
                viz_html = plot_wordcloud(docs)
                answer += "☁️ Word cloud generated."
            elif any(word in q_lower for word in ["sentiment", "tone", "positive", "negative"]):
                viz_html = sentiment_distribution(docs)
                answer += "😊 Sentiment distribution generated."
            else:
                stats = {
                    "num_docs": len(docs),
                    "avg_len": np.mean([len(d.split()) for d in docs]) if docs else 0
                }
                answer += f"Corpus has {stats['num_docs']} docs, avg length {stats['avg_len']:.2f} words."
                viz_html = plot_word_frequencies(docs)
        except Exception as e:
            answer += f"\n⚠️ Error during text analysis: {e}"

        result = {
            "question": question,
            "answer": answer,
            "visualization_html": viz_html,
            "method": "text_analysis",
        }

        if dataset_name and dataset_name in self.analysis_history:
            self.analysis_history[dataset_name].append({
                "question": question,
                "result": result,
                "timestamp": pd.Timestamp.now(),
            })

        return result

    # ================= MULTI-Q =================
    def answer_multiple(self, state: STATE, data, questions: list, dataset_name: str) -> STATE:
        results = []
        print(f"\n🔄 Processing {len(questions)} questions...")

        # precompute structured or text cache
        if isinstance(data, pd.DataFrame) and dataset_name and dataset_name not in self.dataset_cache:
            self.set_dataset(dataset_name, data)
        elif isinstance(data, list) and dataset_name and dataset_name not in self.text_cache:
            self.set_text_dataset(dataset_name, data)

        for i, question in enumerate(questions, 1):
            print(f"   Q{i}/{len(questions)}: {question[:50]}...")
            try:
                result = self.answer(data, question, dataset_name)
                results.append(result)
            except Exception as e:
                print(f"   ⚠️ Failed Q{i}: {e}")
                results.append({
                    "question": question,
                    "answer": f"⚠️ Error processing question: {e}",
                    "visualization_html": None,
                    "method": "error"
                })

        print(f"✅ Completed {len(results)} questions")

        # update state
        state.insights[dataset_name] = results
        return state
    
    # ================= INTELLIGENT ANALYSIS METHODS =================
    
    def _create_intelligent_data_context(self, df: pd.DataFrame, question: str, relevant_cols: list, numeric_cols: list, categorical_cols: list) -> str:
        """Create intelligent data context with smart column interpretation and comprehensive data analysis"""
        
        # Start with dataset overview
        context_parts = [
            f"**DATASET OVERVIEW:**",
            f"- Size: {df.shape[0]:,} records × {df.shape[1]} columns",
            f"- Data appears to be: {self._infer_dataset_type(df, categorical_cols)}"
        ]
        
        # Intelligent column interpretation
        context_parts.append("\n**COLUMN INTERPRETATION:**")
        all_cols_to_analyze = list(set(relevant_cols + numeric_cols[:3] + categorical_cols[:3]))
        
        for col in all_cols_to_analyze[:8]:  # Analyze up to 8 most relevant columns
            if col in df.columns:
                col_analysis = self._analyze_column_intelligently(df, col)
                context_parts.append(f"- **{col}**: {col_analysis}")
        
        # Data patterns and relationships
        context_parts.append("\n**DATA PATTERNS:**")
        
        # Analyze temporal patterns if time data exists
        temporal_cols = self._identify_temporal_columns(df, all_cols_to_analyze)
        if temporal_cols:
            context_parts.append(f"- Temporal data detected: {temporal_cols}")
            context_parts.extend(self._analyze_temporal_patterns(df, temporal_cols[0]))
        
        # Analyze numeric relationships
        if len(numeric_cols) >= 2:
            context_parts.extend(self._analyze_numeric_relationships(df, numeric_cols[:4]))
        
        # Sample data with intelligent selection
        context_parts.append("\n**SAMPLE DATA:**")
        if len(all_cols_to_analyze) > 0:
            sample_cols = all_cols_to_analyze[:6]  # Show up to 6 most relevant columns
            sample_data = df[sample_cols].head(5)
            context_parts.append(f"First 5 rows of key columns:\n{sample_data.to_string()}")
        
        return "\n".join(context_parts)
    
    def _infer_dataset_type(self, df: pd.DataFrame, categorical_cols: list) -> str:
        """Intelligently infer what type of dataset this is based on column names and structure"""
        all_cols_lower = [col.lower() for col in df.columns]
        all_cols_text = ' '.join(all_cols_lower)
        
        # Economic/Financial data
        if any(term in all_cols_text for term in ['gdp', 'gross domestic product', 'expenditure', 'consumption', 'investment', 'exports', 'imports']):
            return "Economic/Macroeconomic time series data"
        
        # Business/Sales data
        elif any(term in all_cols_text for term in ['revenue', 'sales', 'profit', 'customer', 'product', 'order']):
            return "Business/Commercial data"
        
        # Health/Medical data
        elif any(term in all_cols_text for term in ['patient', 'medical', 'health', 'treatment', 'diagnosis']):
            return "Healthcare/Medical data"
        
        # Time series if many numeric columns
        elif len(df.select_dtypes(include=[np.number]).columns) > df.shape[1] * 0.6:
            return "Quantitative/Numeric dataset"
        
        else:
            return "Mixed structured data"
    
    def _analyze_column_intelligently(self, df: pd.DataFrame, col: str) -> str:
        """Analyze a column intelligently to understand what it represents and its key characteristics"""
        try:
            col_lower = col.lower()
            
            # Interpret column meaning based on name patterns
            if 'gdp' in col_lower:
                if '.2' in col or '.3' in col or '.6' in col or '.7' in col:
                    interpretation = "Likely GDP growth rates (quarterly or annual %)"
                elif '.1' in col or '.5' in col:
                    interpretation = "Likely GDP values (seasonally adjusted)"
                else:
                    interpretation = "GDP-related economic indicator"
            elif 'consumption' in col_lower:
                interpretation = "Consumer spending measure"
            elif 'expenditure' in col_lower:
                interpretation = "Expenditure/spending category"
            elif 'export' in col_lower or 'import' in col_lower:
                interpretation = "Trade-related measure"
            elif 'investment' in col_lower or 'capital formation' in col_lower:
                interpretation = "Investment/capital formation measure"
            else:
                interpretation = "Economic indicator"
            
            # Analyze the data values
            if df[col].dtype in ['int64', 'float64']:
                stats = df[col].describe()
                non_null = df[col].count()
                return f"{interpretation}. Range: [{stats['min']:.1f}, {stats['max']:.1f}], Mean: {stats['mean']:.1f}, Non-null: {non_null}/{len(df)}"
            else:
                # Try to convert to numeric for mixed columns
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.count() > len(df) * 0.5:  # More than 50% are numeric
                    stats = numeric_series.describe()
                    return f"{interpretation}. Range: [{stats['min']:.1f}, {stats['max']:.1f}], Mean: {stats['mean']:.1f}, {numeric_series.count()}/{len(df)} numeric values"
                else:
                    unique_vals = df[col].nunique()
                    top_val = df[col].value_counts().index[0] if unique_vals > 0 else "None"
                    return f"{interpretation}. {unique_vals} unique values, most common: '{top_val}'"
                
        except Exception as e:
            return f"Analysis error: {str(e)[:50]}"
    
    def _identify_temporal_columns(self, df: pd.DataFrame, cols_to_check: list) -> list:
        """Identify columns that likely contain temporal/time information"""
        temporal_cols = []
        
        for col in cols_to_check:
            if col in df.columns:
                col_lower = col.lower()
                # Check for datetime columns or columns with temporal names
                if (df[col].dtype in ['datetime64[ns]', 'datetime64'] or 
                    any(term in col_lower for term in ['date', 'time', 'year', 'quarter', 'month', 'unnamed: 0']) or
                    col == 'Unnamed: 0'):  # Often index columns contain dates
                    temporal_cols.append(col)
                    
        return temporal_cols
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, temporal_col: str) -> list:
        """Analyze patterns in temporal data"""
        patterns = []
        try:
            # Check if we can identify any time patterns
            unique_vals = df[temporal_col].nunique()
            total_rows = len(df)
            
            patterns.append(f"- Time dimension: {unique_vals} unique time points across {total_rows} observations")
            
            # If it's a reasonable time series, check for patterns
            if 10 <= unique_vals <= total_rows * 0.8:
                patterns.append(f"- Appears to be time series data with potential temporal trends")
                
        except Exception as e:
            patterns.append(f"- Temporal analysis limited: {str(e)[:50]}")
            
        return patterns
    
    def _analyze_numeric_relationships(self, df: pd.DataFrame, numeric_cols: list) -> list:
        """Analyze relationships between numeric columns"""
        relationships = []
        
        try:
            # Convert columns to numeric, handling mixed types
            clean_numeric_data = {}
            for col in numeric_cols[:4]:
                if col in df.columns:
                    # Try to convert to numeric, coercing errors to NaN
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if numeric_series.count() > len(df) * 0.3:  # At least 30% valid numeric values
                        clean_numeric_data[col] = numeric_series
            
            if len(clean_numeric_data) >= 2:
                # Create correlation matrix
                corr_df = pd.DataFrame(clean_numeric_data).corr()
                
                # Find strongest correlations
                strong_corrs = []
                for i, col1 in enumerate(corr_df.columns):
                    for j, col2 in enumerate(corr_df.columns[i+1:], i+1):
                        corr_val = corr_df.iloc[i, j]
                        if abs(corr_val) > 0.3:  # Meaningful correlation
                            strong_corrs.append((col1, col2, corr_val))
                
                if strong_corrs:
                    relationships.append("- **Key Correlations:**")
                    for col1, col2, corr in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)[:3]:
                        relationships.append(f"  - {col1} ↔ {col2}: {corr:.3f}")
                else:
                    relationships.append("- Numeric variables show weak correlations (< 0.3)")
            
        except Exception as e:
            relationships.append(f"- Relationship analysis limited: {str(e)[:50]}")
            
        return relationships
    
    def _is_intelligent_response(self, response: str) -> bool:
        """Check if response shows intelligent data analysis"""
        if not response or len(response.strip()) < 50:
            return False
            
        response_lower = response.lower()
        
        # Look for data-driven analysis indicators
        intelligent_indicators = [
            # Specific data references
            'based on the data', 'the dataset shows', 'analysis reveals', 'data indicates',
            # Quantitative insights
            'correlation', 'trend', 'pattern', 'growth', 'increase', 'decrease', 'volatility',
            # Statistical terms
            'mean', 'average', 'median', 'standard deviation', 'range', 'percentage',
            # Economic/domain terms
            'gdp', 'expenditure', 'consumption', 'investment', 'exports', 'imports',
            # Analytical terms
            'comparison', 'relationship', 'significant', 'period', 'component'
        ]
        
        # Avoid generic/evasive responses
        evasive_phrases = [
            'unfortunately', 'lacks the necessary', 'insufficient data', 'cannot determine',
            'not enough information', 'would need', 'does not contain', 'missing data'
        ]
        
        # Check for intelligent indicators
        has_intelligent_content = sum(1 for indicator in intelligent_indicators if indicator in response_lower) >= 3
        
        # Check for evasive content
        has_evasive_content = any(phrase in response_lower for phrase in evasive_phrases)
        
        return has_intelligent_content and not has_evasive_content
    
    def _format_intelligent_response(self, response: str) -> str:
        """Format intelligent response for better readability"""
        try:
            formatted = "🔍 **Intelligent Data Analysis**\n\n"
            
            # Clean and structure the response
            lines = response.replace('\r', '').split('\n')
            clean_lines = [line.strip() for line in lines if line.strip()]
            
            formatted_content = []
            for line in clean_lines:
                # Add bullet points to main insights
                if (line and not line.startswith(('�', '•', '-', '*')) and 
                    len(line) > 20 and not line.endswith(':')):
                    formatted_content.append(f"• {line}")
                elif line.endswith(':') and len(line) < 100:
                    formatted_content.append(f"\n📊 **{line.replace(':', '')}**")
                else:
                    formatted_content.append(line)
            
            formatted += "\n".join(formatted_content)
            return formatted
            
        except Exception as e:
            return f"🔍 **Analysis Results**\n\n• {response}"
    
    def _suggest_intelligent_visualization(self, df: pd.DataFrame, question: str, relevant_cols: list, numeric_cols: list, categorical_cols: list) -> str:
        """Use LLM to intelligently create question-specific visualizations"""
        try:
            print(f"🎨 Creating intelligent visualization for: {question}")
            
            # Use LLM to get smart visualization recommendation
            if self.llm and self.llm.is_available():
                viz_recommendation = self._get_smart_visualization_recommendation(df, question, relevant_cols, numeric_cols, categorical_cols)
                
                # Create the visualization based on LLM recommendation
                if viz_recommendation:
                    return self._create_smart_visualization(df, viz_recommendation, question, numeric_cols, categorical_cols)
            
            # Fallback to basic logic if LLM unavailable
            return self._create_fallback_visualization(df, question, relevant_cols, numeric_cols, categorical_cols)
            
        except Exception as e:
            print(f"⚠️ Smart visualization creation failed: {e}")
            return self._create_fallback_visualization(df, question, relevant_cols, numeric_cols, categorical_cols)
    
    def _get_smart_visualization_recommendation(self, df: pd.DataFrame, question: str, relevant_cols: list, numeric_cols: list, categorical_cols: list) -> dict:
        """Use LLM to get intelligent visualization recommendation based on question context"""
        try:
            # Create contextual data summary
            data_context = self._create_visualization_context(df, relevant_cols, numeric_cols, categorical_cols)
            
            prompt = f"""You are a data visualization expert. Analyze this question and recommend the PERFECT visualization.

QUESTION: "{question}"

DATA CONTEXT:
{data_context}

VISUALIZATION OPTIONS:
1. histogram - Distribution of single numeric variable
2. bar_chart - Compare values across categories  
3. scatter_plot - Relationship between two numeric variables
4. line_chart - Trends over time or sequence
5. box_plot - Distribution with quartiles and outliers
6. heatmap - Correlation matrix for multiple variables
7. pie_chart - Parts of a whole (for categorical data)
8. none - No visualization needed (for basic info questions)

TASK: Choose the BEST visualization that directly answers this specific question.

Respond in this EXACT format:
TYPE: [visualization_type]
COLUMNS: [relevant_column_names]
REASONING: [why this visualization answers the question]
MESSAGE: [brief explanation for user]

Example:
TYPE: bar_chart
COLUMNS: category, value
REASONING: Question asks for comparison across categories
MESSAGE: Bar chart shows clear differences between categories"""
            
            response = self.llm(prompt)
            return self._parse_smart_viz_recommendation(response)
            
        except Exception as e:
            print(f"⚠️ Smart visualization recommendation failed: {e}")
            return None
    
    def _create_visualization_context(self, df: pd.DataFrame, relevant_cols: list, numeric_cols: list, categorical_cols: list) -> str:
        """Create focused data context for visualization decisions"""
        context_parts = [
            f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns",
            f"Relevant columns for question: {relevant_cols[:5]}",
            f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:5]}",
            f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}"
        ]
        
        # Add sample data for context
        if len(df) > 0:
            sample_cols = relevant_cols[:3] if relevant_cols else (numeric_cols + categorical_cols)[:3]
            if sample_cols:
                context_parts.append(f"\nSample data for {sample_cols}:")
                context_parts.append(df[sample_cols].head(3).to_string())
        
        return "\n".join(context_parts)
    
    def _parse_smart_viz_recommendation(self, response: str) -> dict:
        """Parse LLM visualization recommendation"""
        try:
            lines = response.strip().split('\n')
            result = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('TYPE:'):
                    result['type'] = line.replace('TYPE:', '').strip().lower()
                elif line.startswith('COLUMNS:'):
                    columns_str = line.replace('COLUMNS:', '').strip()
                    result['columns'] = [col.strip() for col in columns_str.split(',') if col.strip()]
                elif line.startswith('REASONING:'):
                    result['reasoning'] = line.replace('REASONING:', '').strip()
                elif line.startswith('MESSAGE:'):
                    result['message'] = line.replace('MESSAGE:', '').strip()
            
            if 'type' not in result:
                result['type'] = 'bar_chart'
            if 'columns' not in result:
                result['columns'] = []
            
            print(f"🎯 LLM recommends: {result['type']} with columns {result.get('columns', [])}")
            return result
            
        except Exception as e:
            print(f"⚠️ Failed to parse visualization recommendation: {e}")
            return {'type': 'bar_chart', 'columns': [], 'reasoning': 'Parse error'}
    
    def _create_smart_visualization(self, df: pd.DataFrame, recommendation: dict, question: str, numeric_cols: list, categorical_cols: list) -> str:
        """Create visualization based on intelligent LLM recommendation"""
        try:
            from utils.plotting_utils import (
                plot_histogram, plot_bar, plot_scatter, plot_line, 
                plot_correlation_matrix, plot_box
            )
            import plotly.express as px
            
            viz_type = recommendation.get('type', 'bar_chart')
            suggested_columns = recommendation.get('columns', [])
            
            # Normalize and map suggested columns to real df columns (case-insensitive, strip quotes)
            normalized_map = {str(c).lower(): c for c in df.columns}
            candidate_cols = [c for c in (col.strip("'\"` ") for col in suggested_columns) if c]
            valid_columns = [normalized_map.get(c.lower()) for c in candidate_cols if normalized_map.get(c.lower())]
            
            print(f"🎨 Creating {viz_type} with columns: {valid_columns}")
            
            if viz_type == 'histogram':
                import pandas as pd
                def is_true_numeric(col_name: str) -> bool:
                    try:
                        return pd.api.types.is_numeric_dtype(df[col_name]) and not pd.api.types.is_datetime64_any_dtype(df[col_name])
                    except Exception:
                        return False
                candidates = [col for col in valid_columns if is_true_numeric(col)] or [col for col in numeric_cols if is_true_numeric(col)]
                if candidates:
                    target_col = candidates[0]
                    print(f"✅ Using numeric column for histogram: {target_col}")
                    return plot_histogram(df, target_col)
                # No numeric columns; if categorical present, show counts
                if categorical_cols:
                    import plotly.express as px
                    cat_col = categorical_cols[0]
                    counts = df[cat_col].value_counts().reset_index()
                    counts.columns = [cat_col, 'count']
                    fig = px.bar(counts, x=cat_col, y='count', title=f"Count of {cat_col}")
                    return fig.to_html(full_html=False)
                return None
                
            elif viz_type == 'bar_chart':
                import pandas as pd
                def is_true_numeric(col_name: str) -> bool:
                    try:
                        return pd.api.types.is_numeric_dtype(df[col_name]) and not pd.api.types.is_datetime64_any_dtype(df[col_name])
                    except Exception:
                        return False
                def is_categorical(col_name: str) -> bool:
                    try:
                        return pd.api.types.is_categorical_dtype(df[col_name]) or df[col_name].dtype == 'object'
                    except Exception:
                        return False
                cat_col = next((c for c in valid_columns if is_categorical(c)), (categorical_cols[0] if categorical_cols else None))
                num_col = next((c for c in valid_columns if is_true_numeric(c)), (numeric_cols[0] if numeric_cols else None))
                if cat_col and num_col:
                    grouped = df.groupby(cat_col)[num_col].mean().reset_index()
                    return plot_bar(grouped, cat_col, num_col)
                elif cat_col:
                    # Pure categorical count bar
                    import plotly.express as px
                    counts = df[cat_col].value_counts().reset_index()
                    counts.columns = [cat_col, 'count']
                    fig = px.bar(counts, x=cat_col, y='count', title=f"Count of {cat_col}")
                    return fig.to_html(full_html=False)
                elif num_col:
                    return plot_histogram(df, num_col)
                return None
                    
            elif viz_type == 'scatter_plot':
                # ensure numeric-numeric
                import pandas as pd
                def is_true_numeric(col_name: str) -> bool:
                    try:
                        return pd.api.types.is_numeric_dtype(df[col_name]) and not pd.api.types.is_datetime64_any_dtype(df[col_name])
                    except Exception:
                        return False
                num_list = [c for c in valid_columns if is_true_numeric(c)]
                if len(num_list) < 2:
                    num_list = [c for c in numeric_cols if is_true_numeric(c)][:2]
                if len(num_list) >= 2:
                    color_by = next((c for c in valid_columns if (c in categorical_cols)), None)
                    return plot_scatter(df, num_list[0], num_list[1], color_by)
                return None
                
            elif viz_type == 'box_plot':
                if valid_columns and valid_columns[0] in numeric_cols:
                    num_col = valid_columns[0]
                    cat_col = next((col for col in valid_columns[1:] if col in categorical_cols), None)
                    return plot_box(df, num_col, cat_col)
                elif numeric_cols:
                    return plot_box(df, numeric_cols[0])
                    
            elif viz_type == 'heatmap' and len(numeric_cols) >= 2:
                return plot_correlation_matrix(df[numeric_cols])
                
            elif viz_type == 'pie_chart' and categorical_cols:
                cat_col = valid_columns[0] if valid_columns and valid_columns[0] in categorical_cols else categorical_cols[0]
                value_counts = df[cat_col].value_counts().head(10)
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f"Distribution of {cat_col}")
                return fig.to_html(full_html=False)
                
            elif viz_type == 'none':
                return None
            
            # Fallback to basic visualization
            return self._create_basic_fallback(df, numeric_cols, categorical_cols)
            
        except Exception as e:
            print(f"⚠️ Smart visualization creation failed: {e}")
            return self._create_basic_fallback(df, numeric_cols, categorical_cols)
    
    def _create_fallback_visualization(self, df: pd.DataFrame, question: str, relevant_cols: list, numeric_cols: list, categorical_cols: list) -> str:
        """Create fallback visualization when LLM is unavailable"""
        try:
            from utils.plotting_utils import plot_histogram, plot_bar, plot_scatter
            
            question_lower = question.lower()
            
            # Question-specific logic
            if any(word in question_lower for word in ['compare', 'versus', 'difference', 'between', 'across']):
                if categorical_cols and numeric_cols:
                    grouped = df.groupby(categorical_cols[0])[numeric_cols[0]].mean().reset_index()
                    return plot_bar(grouped, categorical_cols[0], numeric_cols[0])
                    
            elif any(word in question_lower for word in ['relationship', 'correlation', 'versus']):
                if len(numeric_cols) >= 2:
                    return plot_scatter(df, numeric_cols[0], numeric_cols[1])
                    
            elif any(word in question_lower for word in ['distribution', 'spread', 'range']):
                if numeric_cols:
                    return plot_histogram(df, numeric_cols[0])
            
            # Default fallback
            return self._create_basic_fallback(df, numeric_cols, categorical_cols)
            
        except Exception:
            return self._create_basic_fallback(df, numeric_cols, categorical_cols)
    
    def _create_basic_fallback(self, df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> str:
        """Most basic visualization fallback"""
        try:
            from utils.plotting_utils import plot_histogram, plot_bar
            
            if categorical_cols and numeric_cols:
                grouped = df.groupby(categorical_cols[0])[numeric_cols[0]].mean().reset_index()
                return plot_bar(grouped, categorical_cols[0], numeric_cols[0])
            elif numeric_cols:
                return plot_histogram(df, numeric_cols[0])
            else:
                return None
        except Exception:
            return None
