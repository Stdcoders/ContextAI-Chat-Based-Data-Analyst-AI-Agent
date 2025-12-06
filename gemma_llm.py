#!/usr/bin/env python3
"""
Gemma LLM wrapper for data analysis tasks.
Replaces FLAN-T5 throughout the system with the superior Gemma-2-2b-it model.
"""

import os
from typing import Optional
from huggingface_hub import InferenceClient

class GemmaLLM:
    """
    Gemma-2-2b-it LLM wrapper for data analysis tasks.
    Drop-in replacement for the old HuggingFaceLLM with FLAN-T5.
    """
    
    def __init__(self, 
                 model_name: str = "google/gemma-2-9b-it",
                 api_token: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 1000):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Get API token from environment or parameter
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        
        self.client = None
        self.llm = None  # For compatibility with existing code
        self._init_client()
    
    def _init_client(self):
        """Initialize the Gemma client."""
        if not self.api_token:
            print("⚠️ No HuggingFace API token found. Set HUGGINGFACE_API_TOKEN environment variable.")
            return
        
        try:
            print(f"🔄 Initializing Gemma LLM: {self.model_name}")
            self.client = InferenceClient(
                model=self.model_name,
                token=self.api_token
            )
            self.llm = "ready"  # Flag for compatibility
            
        except Exception as e:
            self.client = None
            self.llm = None
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Main inference method - compatible with existing code.
        This method is called when the LLM is used like: llm(prompt)
        """
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
    
    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Generate response using Gemma model."""
        if not self.client:
            return "❌ Gemma LLM not available. Please check API token and connection."
        
        try:
            # Use provided parameters or defaults
            tokens = max_tokens or self.max_tokens
            temp = temperature or self.temperature
            
            # Use chat completion for best results
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=tokens,
                temperature=temp
            )
            
            result = response.choices[0].message.content.strip()
            
            # Validate response
            if not result or len(result) < 5:
                return "❌ Empty or invalid response from Gemma LLM"
            
            return result
            
        except Exception as e:
            print(f"⚠️ Gemma generation failed: {e}")
            return f"❌ Generation error: {str(e)[:100]}..."
    
    def chat_completion(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Alternative method for chat completion."""
        return self.generate(prompt, max_tokens, temperature)
    
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        return self.client is not None and self.llm is not None

# Test function
def test_gemma_llm():
    """Test the Gemma LLM wrapper."""
    print("🧪 Testing Gemma LLM Wrapper")
    print("=" * 40)
    
    # Initialize
    llm = GemmaLLM()
    
    if not llm.is_available():
        print("❌ Gemma LLM not available for testing")
        return
    
    # Test domain detection
    print("\n📊 Test 1: Domain Detection")
    domain_prompt = """Analyze this dataset and determine its domain:

Dataset columns: Sleep Duration, Quality of Sleep, Stress Level, BMI Category, Blood Pressure, Heart Rate, Sleep Disorder

What domain does this belong to?"""
    
    result = llm(domain_prompt)
    print(f"Result: {result}")
    
    # Test question generation  
    print("\n📊 Test 2: Question Generation")
    question_prompt = """Generate 2 analytical questions for a sleep health dataset:

Columns: Sleep Duration, Quality of Sleep, Physical Activity Level, Stress Level

Questions:"""
    
    result = llm(question_prompt)
    print(f"Result: {result}")
    
    print("\n✅ Gemma LLM testing completed!")

if __name__ == "__main__":
    test_gemma_llm()
