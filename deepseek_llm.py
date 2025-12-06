#!/usr/bin/env python3
"""
Improved LLM wrapper: uses huggingface_hub.InferenceClient when available
and falls back to the HTTP Inference API. Handles provider selection,
parses typical provider response shapes, and prints actionable errors.
"""

import os
import json
import time
from typing import Optional, Any

# optional: install with `pip install huggingface_hub`
try:
    from huggingface_hub import InferenceClient, model_info
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

import requests
from dotenv import load_dotenv
load_dotenv()


class DeepSeekLLM:
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        use_inference_client: bool = True,
        provider: str = "nscale",  # nscale provider for DeepSeek models
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        # keep legacy REST URL as a fallback
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}

        self.use_inference_client = use_inference_client and HF_AVAILABLE
        self.client = None
        self._available = False

        if self.use_inference_client:
            try:
                # initialize client; provider can be "auto" or a specific provider
                self.client = InferenceClient(token=self.api_token, provider=self.provider)
            except Exception as e:
                print(f"⚠️ could not initialize InferenceClient: {e} — falling back to HTTP requests")
                self.client = None
                self.use_inference_client = False

        # test availability
        self._test_model_availability()

    # ---------- utilities ----------
    def _extract_text(self, resp: Any) -> Optional[str]:
        """Best-effort extractor for returned objects from HF providers/clients."""
        try:
            if resp is None:
                return None
            # If it's already a string
            if isinstance(resp, str):
                return resp
            # If library returned a list (common)
            if isinstance(resp, list) and len(resp) > 0:
                item = resp[0]
                if isinstance(item, dict):
                    for k in ("generated_text", "text", "content"):
                        if k in item:
                            return item[k]
                    # fallback to JSON string
                    return json.dumps(item)
                else:
                    # dataclass-like object (dataclass attrs)
                    for attr in ("generated_text", "text", "content"):
                        if hasattr(item, attr):
                            return getattr(item, attr)
                    return str(item)
            # If it's a dict
            if isinstance(resp, dict):
                for k in ("generated_text", "text", "content"):
                    if k in resp:
                        return resp[k]
                return json.dumps(resp)

            # fallback to str()
            return str(resp)
        except Exception as e:
            print(f"⚠️ error extracting text from response: {e}")
            return None

    # ---------- availability test ----------
    def _test_model_availability(self):
        """Try a quick inference to detect availability and helpful failure reasons."""
        test_prompt = "Calculate 2+2 and explain the result in one short sentence."
        try:
            if self.use_inference_client and self.client:
                # try chat completions for DeepSeek-R1
                if "deepseek-ai" in self.model_name.lower() and self.provider == "nscale":
                    try:
                        chat_result = self._make_chat_completion(test_prompt)
                        if chat_result and len(chat_result.strip()) > 0:
                            self._available = True
                            print(f"✅ DeepSeek-R1 available via nscale chat completions: {self.model_name}")
                            return
                    except Exception as e:
                        print(f"⚠️ Chat completions failed: {e}")
                
                # fallback to standard text generation
                try:
                    response = self.client.text_generation(
                        test_prompt,
                        model=self.model_name,
                        max_new_tokens=32,
                        temperature=0.0,
                    )
                    text = self._extract_text(response)
                    if text and len(text.strip()) > 0:
                        self._available = True
                        print(f"✅ Model reachable via InferenceClient: {self.model_name} (provider={self.provider})")
                        return
                except Exception as e:
                    print(f"⚠️ InferenceClient call failed: {e}")

            # fallback: direct HTTP to HF Inference API
            if not self.api_token:
                print("⚠️ No Hugging Face API token found; direct HTTP requests may be rate-limited or blocked.")
            resp = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "inputs": test_prompt,
                    "parameters": {"temperature": self.temperature, "max_new_tokens": 32},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                body = resp.json()
                text = self._extract_text(body)
                if text and len(text.strip()) > 0:
                    self._available = True
                    print(f"✅ Model reachable via HF Inference HTTP: {self.model_name}")
                    return
            else:
                # print clear reason
                print(f"⚠️ HTTP test returned {resp.status_code}: {resp.text}")
                # 401/403 -> token or private model; 404 -> model not hosted; 503 -> loading
                if resp.status_code == 404:
                    print("→ 404 likely means the model is not hosted on the default Hugging Face Inference API. Check the model card for 'Inference Providers'.")
                elif resp.status_code in (401, 403):
                    print("→ 401/403 — token missing/invalid or model is private. Check your token and model permissions.")
                elif resp.status_code == 503:
                    print("→ 503 — model is loading on the provider; try again after a short wait.")
        except requests.exceptions.Timeout:
            print("⏱️ Model availability test timed out")
        except Exception as e:
            print(f"⚠️ model availability check error: {e}")

    # ---------- low level call ----------
    def _make_chat_completion(self, prompt: str, retries: int = 2) -> Optional[str]:
        """Use chat completions API for DeepSeek-R1 with nscale provider."""
        if not self.use_inference_client or not self.client:
            return None
            
        for attempt in range(1, retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                if completion and completion.choices and len(completion.choices) > 0:
                    message = completion.choices[0].message
                    if hasattr(message, 'content') and message.content:
                        return message.content.strip()
                    elif hasattr(message, 'text') and message.text:
                        return message.text.strip()
                    else:
                        text_content = str(message)
                        if text_content and len(text_content.strip()) > 5:
                            return text_content.strip()
                        
            except Exception as e:
                print(f"⚠️ Chat completion attempt {attempt} error: {e}")
                if attempt < retries:
                    time.sleep(1)  # Brief pause before retry
                    continue
                    
        return None
    
    def _make_request(self, prompt: str, retries: int = 3) -> Optional[str]:
        """Unified inference call. Tries InferenceClient (if available) then HTTP fallback."""
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                # For DeepSeek-R1 with nscale: ONLY use chat completions
                if "deepseek-ai" in self.model_name.lower() and self.provider == "nscale":
                    chat_result = self._make_chat_completion(prompt)
                    if chat_result:
                        return chat_result
                    else:
                        # If chat completions failed, don't try other methods for nscale
                        print(f"⚠️ Chat completions failed for DeepSeek-R1 (nscale provider)")
                        break
                
                # For other models: try standard InferenceClient
                if self.use_inference_client and self.client:
                    try:
                        resp = self.client.text_generation(
                            prompt,
                            model=self.model_name,
                            max_new_tokens=self.max_tokens,
                            temperature=self.temperature,
                        )
                        text = self._extract_text(resp)
                        if text:
                            return text.strip()
                        # if client returned nothing meaningful, continue to fallback
                    except Exception as e:
                        last_exc = e
                        print(f"⚠️ InferenceClient attempt {attempt} failed: {e}")
                        # sometimes provider mismatch -> try different provider or fallback
                        # keep trying; don't immediately break

                # Fallback: direct HF Inference HTTP
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_tokens,
                        "return_full_text": False,
                    },
                }
                resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    body = resp.json()
                    text = self._extract_text(body)
                    if text:
                        return text.strip()
                elif resp.status_code == 503:
                    print(f"🔄 Model loading (attempt {attempt}/{retries}) — sleeping briefly")
                    time.sleep(5 + attempt * 2)
                    continue
                else:
                    print(f"⚠️ HTTP {resp.status_code} error: {resp.text}")
                    # do not retry for 401/403/404 unless you expect token changes
                    if resp.status_code in (401, 403, 404):
                        break
            except requests.exceptions.Timeout:
                print(f"⏱️ Request timeout (attempt {attempt}/{retries})")
                time.sleep(2)
                continue
            except Exception as e:
                last_exc = e
                print(f"⚠️ Unexpected request error: {e}")
                break

        if last_exc:
            print("⚠️ Final error (see above).")
        return None

    # ---------- public API ----------
    def is_available(self) -> bool:
        return self._available

    def calculate(self, mathematical_query: str) -> Optional[str]:
        prompt = f"""You are a mathematical reasoning expert. Solve step-by-step:

QUERY: {mathematical_query}

INSTRUCTIONS:
1. Break down the problem into clear steps
2. Perform calculations precisely
3. Show all work
4. Provide final numerical result and confidence estimate

SOLUTION:"""
        return self._make_request(prompt)

    def analyze_data_pattern(self, data_context: str, analysis_question: str) -> Optional[str]:
        prompt = f"""Analyze this data pattern:

DATA:
{data_context}

QUESTION: {analysis_question}

REQUIREMENTS:
- identify relationships
- compute relevant statistics
- show calculations
- summarize conclusions
ANALYSIS:"""
        return self._make_request(prompt)

    def verify_calculation(self, calculation_steps: str) -> Optional[str]:
        prompt = f"""Verify this calculation:

CALCULATION:
{calculation_steps}

STEPS:
1. Check each line
2. Recompute
3. Report errors and final verified result
VERIFICATION:"""
        return self._make_request(prompt)

    def generate(self, prompt: str) -> Optional[str]:
        return self._make_request(prompt)

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt) or ""
