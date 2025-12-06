#!/usr/bin/env python3
"""
Groq LLM wrapper with ultra-fast inference for GPT-OSS-120B.
Optimized for data analysis tasks with high-speed processing.
"""

import os
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()


@dataclass
class GroqUsageTracker:
    """Track Groq API usage"""
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_time_taken: float = 0.0
    
    def add_usage(self, input_tokens: int, output_tokens: int, time_taken: float) -> None:
        """Add usage statistics"""
        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_time_taken += time_taken
    
    def get_summary(self) -> str:
        """Get usage summary"""
        avg_time = self.total_time_taken / max(self.total_requests, 1)
        tokens_per_sec = (self.total_input_tokens + self.total_output_tokens) / max(self.total_time_taken, 0.01)
        
        return f"""⚡ Groq Usage Summary:
   📊 Requests: {self.total_requests}
   📥 Input tokens: {self.total_input_tokens:,}
   📤 Output tokens: {self.total_output_tokens:,}
   ⏱️ Total time: {self.total_time_taken:.2f}s
   🚀 Avg speed: {tokens_per_sec:.0f} tokens/sec
   💫 Avg response time: {avg_time:.2f}s"""


class GroqLLM:
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.0,
        max_tokens: int = 2500,
        timeout: int = 30,
    ):
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library not installed. Run: pip install groq")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Initialize API key
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize client
        self.client = Groq(api_key=self.api_key)
        self.usage_tracker = GroqUsageTracker()
        self._available = False
        
        # Test availability
        self._test_model_availability()
    
    def _test_model_availability(self):
        """Test if the model is available and working"""
        try:
            test_prompt = "Calculate 3+5 and explain briefly."
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=50,
                temperature=0.0,
                timeout=self.timeout
            )
            
            end_time = time.time()
            
            if response.choices and len(response.choices) > 0:
                response_time = end_time - start_time
                self._available = True
                print(f"✅ Groq {self.model_name} available - Response time: {response_time:.2f}s")
                
                # Track test usage
                if hasattr(response, 'usage') and response.usage:
                    self.usage_tracker.add_usage(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                        response_time
                    )
            else:
                print(f"⚠️ Groq {self.model_name} responded but no content generated")
                
        except Exception as e:
            print(f"⚠️ Groq model test failed: {e}")
    
    def _make_request(self, prompt: str, retries: int = 3) -> Optional[str]:
        """Make request to Groq API with performance tracking"""
        for attempt in range(1, retries + 1):
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout
                )
                
                end_time = time.time()
                request_time = end_time - start_time
                
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    
                    # Track usage and performance
                    if hasattr(response, 'usage') and response.usage:
                        input_tokens = response.usage.prompt_tokens
                        output_tokens = response.usage.completion_tokens
                        self.usage_tracker.add_usage(input_tokens, output_tokens, request_time)
                        
                        tokens_per_sec = (input_tokens + output_tokens) / max(request_time, 0.01)
                        print(f"⚡ Groq speed: {tokens_per_sec:.0f} tokens/sec | Time: {request_time:.2f}s")
                    
                    return content.strip() if content else None
                
            except Exception as e:
                print(f"⚠️ Groq API error (attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(0.5)  # Brief pause before retry
                    continue
                break
        
        return None
    
    def is_available(self) -> bool:
        """Check if the model is available"""
        return self._available
    
    def calculate(self, mathematical_query: str) -> Optional[str]:
        """Perform mathematical calculations and reasoning at ultra-high speed"""
        prompt = f"""You are GPT-OSS-120B, an advanced open-source model with exceptional mathematical and analytical capabilities. Provide rapid, precise analysis:

QUERY: {mathematical_query}

INSTRUCTIONS:
1. **Rapid Analysis**: Break down the problem efficiently
2. **Precise Calculations**: Perform all mathematical operations accurately
3. **Clear Methodology**: Show your calculation steps clearly
4. **Exact Results**: Provide specific numerical answers
5. **High Confidence**: Validate your results and provide confidence levels
6. **Business Context**: Relate findings to practical applications

SOLUTION:"""
        return self._make_request(prompt)
    
    def analyze_data_pattern(self, data_context: str, analysis_question: str) -> Optional[str]:
        """Analyze data patterns with lightning-fast processing"""
        prompt = f"""You are GPT-OSS-120B performing rapid data analysis. Provide comprehensive insights at maximum speed:

DATA CONTEXT:
{data_context}

ANALYSIS QUESTION: {analysis_question}

PROVIDE COMPREHENSIVE ANALYSIS:
1. **Statistical Insights**: Key patterns and numerical findings
2. **Business Intelligence**: Strategic implications and opportunities
3. **Data Relationships**: Correlations and dependencies discovered
4. **Quantitative Results**: Specific calculations and metrics
5. **Actionable Recommendations**: Strategic next steps and decisions
6. **Performance Indicators**: Success metrics and benchmarks

RAPID ANALYSIS:"""
        return self._make_request(prompt)
    
    def verify_calculation(self, calculation_steps: str) -> Optional[str]:
        """Verify calculations with high-speed processing"""
        prompt = f"""Rapidly verify this calculation with precision:

CALCULATION TO VERIFY:
{calculation_steps}

VERIFICATION REQUIREMENTS:
1. **Step-by-Step Check**: Validate each calculation step
2. **Recomputation**: Independently recalculate all results
3. **Error Detection**: Identify any mistakes or inconsistencies
4. **Corrected Results**: Provide verified final answers
5. **Confidence Rating**: Rate accuracy confidence (1-10)
6. **Method Validation**: Confirm calculation approach is optimal

VERIFICATION RESULT:"""
        return self._make_request(prompt)
    
    def generate(self, prompt: str) -> Optional[str]:
        """General purpose ultra-fast text generation"""
        return self._make_request(prompt)
    
    def get_usage_summary(self) -> str:
        """Get detailed usage and performance summary"""
        return self.usage_tracker.get_summary()
    
    def __call__(self, prompt: str) -> str:
        """Make the class callable"""
        return self.generate(prompt) or ""


# Quick test function
def test_groq_llm():
    """Test the Groq LLM wrapper"""
    try:
        llm = GroqLLM()
        if llm.is_available():
            result = llm.calculate("What is 25% of 180? Show your calculation and explain the business relevance.")
            print(f"Test result: {result}")
            print(llm.get_usage_summary())
        else:
            print("Groq LLM not available for testing")
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_groq_llm()