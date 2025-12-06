#!/usr/bin/env python3
"""
LLM-Powered Question Intent Analyzer
Uses Groq and Gemma LLMs to intelligently analyze question intent and route to appropriate engines
"""

import json
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import our existing LLMs
from groq_llm import GroqLLM
from gemma_llm import GemmaLLM

class QuestionIntent(Enum):
    STATISTICAL = "statistical"
    DESCRIPTIVE = "descriptive" 
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"
    EXPLORATORY = "exploratory"

@dataclass
class IntentClassification:
    """Result of LLM-based intent classification"""
    intent: QuestionIntent
    confidence: float
    recommended_engine: str  # "groq" or "gemma"
    reasoning: str
    sub_category: str
    complexity: str  # "simple", "moderate", "complex"

class LLMQuestionClassifier:
    """
    Advanced question classifier using LLMs for intelligent intent analysis
    """
    
    def __init__(self):
        # Initialize LLMs for intent analysis
        try:
            self.groq = GroqLLM(model_name="openai/gpt-oss-120b", temperature=0.0, max_tokens=800)
            print("✅ Groq LLM initialized for intent analysis")
        except Exception as e:
            print(f"⚠️ Groq LLM failed to initialize: {e}")
            self.groq = None
            
        try:
            self.gemma = GemmaLLM(model_name="google/gemma-2-9b-it", temperature=0.1, max_tokens=800)
            print("✅ Gemma LLM initialized for intent analysis")
        except Exception as e:
            print(f"⚠️ Gemma LLM failed to initialize: {e}")
            self.gemma = None
        
        # Engine routing rules
        self.engine_routing = {
            QuestionIntent.STATISTICAL: "groq",      # Fast calculations
            QuestionIntent.COMPARATIVE: "groq",     # Fast comparisons
            QuestionIntent.DESCRIPTIVE: "gemma",    # Rich descriptions
            QuestionIntent.ANALYTICAL: "gemma",     # Deep analysis
            QuestionIntent.PREDICTIVE: "gemma",     # Complex modeling
            QuestionIntent.EXPLORATORY: "gemma"     # Pattern discovery
        }
    
    def analyze_question_intent(self, question: str, dataset_context: Optional[str] = None) -> IntentClassification:
        """
        Use LLMs to analyze question intent and determine routing
        """
        # Try Groq first for fast intent classification
        if self.groq and self.groq.is_available():
            return self._analyze_with_groq(question, dataset_context)
        
        # Fallback to Gemma if Groq unavailable
        elif self.gemma and self.gemma.is_available():
            return self._analyze_with_gemma(question, dataset_context)
        
        # Final fallback to rule-based classification
        else:
            return self._fallback_classification(question)
    
    def _analyze_with_groq(self, question: str, dataset_context: Optional[str] = None) -> IntentClassification:
        """Use Groq for ultra-fast intent analysis"""
        
        intent_prompt = f"""You are an expert data analysis question classifier. Analyze the user's question intent and provide a precise classification.

QUESTION: "{question}"

{f"DATASET CONTEXT: {dataset_context}" if dataset_context else ""}

CLASSIFICATION CATEGORIES:
- STATISTICAL: Requires mathematical calculations, aggregations, statistical operations (sum, avg, min, max, count, correlation)
- COMPARATIVE: Involves comparing data points, ranking, finding differences, best/worst analysis
- DESCRIPTIVE: Seeks to describe, summarize, or explain what the data shows
- ANALYTICAL: Requires deep analysis, insights, pattern interpretation, cause-effect relationships
- PREDICTIVE: Involves forecasting, predicting future outcomes, trend projections
- EXPLORATORY: Aims to discover hidden patterns, anomalies, or explore data relationships

ENGINE RECOMMENDATIONS:
- GROQ: Best for STATISTICAL and COMPARATIVE (ultra-fast mathematical processing)
- GEMMA: Best for DESCRIPTIVE, ANALYTICAL, PREDICTIVE, EXPLORATORY (rich contextual insights)

COMPLEXITY LEVELS:
- SIMPLE: Basic single-step operations
- MODERATE: Multi-step analysis or moderate complexity
- COMPLEX: Advanced analysis requiring multiple approaches

Respond with ONLY a JSON object in this exact format:
{{
    "intent": "statistical|comparative|descriptive|analytical|predictive|exploratory",
    "confidence": 0.95,
    "recommended_engine": "groq|gemma",
    "reasoning": "Clear explanation of why this classification was chosen",
    "sub_category": "specific type within the main category",
    "complexity": "simple|moderate|complex"
}}"""

        try:
            print(f"🎯 Using Groq for question intent analysis...")
            response = self.groq.generate(intent_prompt)
            if response and len(response.strip()) > 10:
                print(f"✅ Groq response received: {response[:100]}...")
                return self._parse_llm_response(response, "groq")
            else:
                print("⚠️ Groq returned empty/short response")
        except Exception as e:
            print(f"⚠️ Groq analysis failed: {e}")
        
        # Fallback to Gemma
        return self._analyze_with_gemma(question, dataset_context)
    
    def _analyze_with_gemma(self, question: str, dataset_context: Optional[str] = None) -> IntentClassification:
        """Use Gemma for detailed intent analysis"""
        
        intent_prompt = f"""As a data analysis expert, classify this user question to determine the best analysis approach.

USER QUESTION: "{question}"

{f"DATASET INFORMATION: {dataset_context}" if dataset_context else ""}

ANALYSIS CATEGORIES:
🔢 STATISTICAL: Mathematical calculations, statistics, aggregations
   Examples: "Calculate average", "Find maximum", "Count records", "Sum values"

📊 COMPARATIVE: Comparing, ranking, finding differences
   Examples: "Compare regions", "Top 10 products", "Which is better", "Rank by performance"

📝 DESCRIPTIVE: Describing, summarizing, explaining data characteristics
   Examples: "Describe the data", "What does this show", "Summarize findings"

🧠 ANALYTICAL: Deep analysis, insights, pattern interpretation
   Examples: "Analyze trends", "Why did this happen", "What factors influence", "Insights about"

🔮 PREDICTIVE: Forecasting, predicting future outcomes
   Examples: "Predict sales", "Future trends", "What will happen", "Forecast"

🔍 EXPLORATORY: Discovering patterns, exploring relationships
   Examples: "Explore data", "Find patterns", "Any correlations", "Hidden insights"

ROUTING LOGIC:
⚡ GROQ GPT-OSS-120B: STATISTICAL + COMPARATIVE (ultra-fast calculations)
🧠 GEMMA-2-9B-IT: DESCRIPTIVE + ANALYTICAL + PREDICTIVE + EXPLORATORY (rich insights)

Provide response as JSON:
{{
    "intent": "category_name",
    "confidence": confidence_score,
    "recommended_engine": "groq_or_gemma",
    "reasoning": "explanation",
    "sub_category": "specific_type",
    "complexity": "simple_moderate_or_complex"
}}"""

        try:
            print(f"🎯 Using Gemma for detailed question intent analysis...")
            response = self.gemma.generate(intent_prompt)
            if response and len(response.strip()) > 10:
                print(f"✅ Gemma response received: {response[:100]}...")
                return self._parse_llm_response(response, "gemma")
            else:
                print("⚠️ Gemma returned empty/short response")
        except Exception as e:
            print(f"⚠️ Gemma analysis failed: {e}")
        
        # Final fallback
        return self._fallback_classification(question)
    
    def _parse_llm_response(self, response: str, source_llm: str) -> IntentClassification:
        """Parse LLM response and create IntentClassification"""
        try:
            # Clean the response to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # Validate and create classification
                intent_str = result.get('intent', 'descriptive').lower()
                intent = QuestionIntent(intent_str)
                confidence = float(result.get('confidence', 0.8))
                recommended_engine = result.get('recommended_engine', 'gemma').lower()
                reasoning = result.get('reasoning', f'Classified by {source_llm.upper()}')
                sub_category = result.get('sub_category', 'general')
                complexity = result.get('complexity', 'moderate')
                
                # Validate engine recommendation against our routing rules
                if intent in self.engine_routing:
                    recommended_engine = self.engine_routing[intent]
                
                return IntentClassification(
                    intent=intent,
                    confidence=min(confidence, 1.0),
                    recommended_engine=recommended_engine,
                    reasoning=reasoning,
                    sub_category=sub_category,
                    complexity=complexity
                )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠️ Failed to parse {source_llm} response: {e}")
        
        # Fallback parsing attempt
        return self._fallback_classification(response if response else "")
    
    def _fallback_classification(self, question: str) -> IntentClassification:
        """Simple rule-based fallback classification"""
        question_lower = question.lower()
        
        # Statistical indicators
        if any(word in question_lower for word in ['calculate', 'compute', 'average', 'sum', 'count', 'total', 'min', 'max']):
            intent = QuestionIntent.STATISTICAL
        # Comparative indicators  
        elif any(word in question_lower for word in ['compare', 'top', 'best', 'worst', 'rank', 'versus', 'vs']):
            intent = QuestionIntent.COMPARATIVE
        # Analytical indicators
        elif any(word in question_lower for word in ['analyze', 'why', 'how', 'relationship', 'correlation', 'insight']):
            intent = QuestionIntent.ANALYTICAL
        # Predictive indicators
        elif any(word in question_lower for word in ['predict', 'forecast', 'future', 'will', 'trend']):
            intent = QuestionIntent.PREDICTIVE
        # Exploratory indicators
        elif any(word in question_lower for word in ['explore', 'discover', 'find', 'pattern', 'hidden']):
            intent = QuestionIntent.EXPLORATORY
        else:
            intent = QuestionIntent.DESCRIPTIVE
        
        return IntentClassification(
            intent=intent,
            confidence=0.6,  # Lower confidence for rule-based
            recommended_engine=self.engine_routing[intent],
            reasoning=f"Rule-based classification based on keywords",
            sub_category="fallback",
            complexity="moderate"
        )
    
    def get_classification_summary(self, classification: IntentClassification) -> str:
        """Generate a detailed summary of the classification"""
        
        engine_details = {
            "groq": {
                "name": "Groq GPT-OSS-120B",
                "icon": "⚡",
                "strength": "Ultra-fast mathematical processing",
                "speed": "~1000+ tokens/sec"
            },
            "gemma": {
                "name": "Gemma-2-9B-IT", 
                "icon": "🧠",
                "strength": "Rich contextual analysis",
                "speed": "Standard LLM speeds"
            }
        }
        
        intent_descriptions = {
            QuestionIntent.STATISTICAL: "Mathematical calculations and statistical operations",
            QuestionIntent.COMPARATIVE: "Comparisons, rankings, and relative analysis",
            QuestionIntent.DESCRIPTIVE: "Data description and summarization",
            QuestionIntent.ANALYTICAL: "Deep insights and pattern analysis",
            QuestionIntent.PREDICTIVE: "Forecasting and future projections", 
            QuestionIntent.EXPLORATORY: "Pattern discovery and data exploration"
        }
        
        engine = engine_details[classification.recommended_engine]
        
        summary = f"""
🎯 Question Intent Analysis Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Intent: {classification.intent.value.upper()}
🎯 Confidence: {classification.confidence:.1%}
📊 Sub-category: {classification.sub_category}
⚙️ Complexity: {classification.complexity.upper()}

🤖 Recommended Engine:
   {engine['icon']} {engine['name']}
   → {engine['strength']}
   → Performance: {engine['speed']}

💡 Reasoning: {classification.reasoning}

🔍 Intent Description: {intent_descriptions[classification.intent]}
"""
        
        return summary
    
    def is_available(self) -> bool:
        """Check if classifier is available"""
        return (self.groq and self.groq.is_available()) or (self.gemma and self.gemma.is_available())


# Test function
def test_llm_classifier():
    """Test the LLM-based question classifier"""
    classifier = LLMQuestionClassifier()
    
    if not classifier.is_available():
        print("❌ No LLMs available for testing")
        return
    
    test_questions = [
        "Calculate the average sales for each product category",
        "Find the top 10 customers by total purchase amount", 
        "What are the main characteristics of our customer base?",
        "Analyze the relationship between marketing spend and revenue",
        "Compare performance across different sales regions",
        "Predict next quarter's sales based on current trends",
        "Explore any unusual patterns in customer behavior data"
    ]
    
    print("🧪 Testing LLM-Powered Question Classification:")
    print("=" * 70)
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("⏳ Analyzing intent with LLMs...")
        
        classification = classifier.analyze_question_intent(question)
        
        engine_icon = "⚡" if classification.recommended_engine == "groq" else "🧠"
        print(f"   {engine_icon} {classification.intent.value.upper()} → {classification.recommended_engine.upper()}")
        print(f"   📊 Confidence: {classification.confidence:.1%} | Complexity: {classification.complexity}")
        print(f"   💡 {classification.reasoning}")

if __name__ == "__main__":
    test_llm_classifier()