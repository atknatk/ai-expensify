"""
Alternative LLM services for cost optimization.
"""

import logging
import json
import httpx
from typing import Dict, Any, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class AlternativeLLMService:
    """Alternative LLM services for cost optimization."""
    
    def __init__(self):
        """Initialize alternative LLM service."""
        self.groq_api_key = getattr(settings, 'groq_api_key', None)
        self.anthropic_api_key = getattr(settings, 'anthropic_api_key', None)
        self.ollama_base_url = getattr(settings, 'ollama_base_url', 'http://localhost:11434')
    
    async def generate_text_groq(self, prompt: str, model: str = "llama3-8b-8192") -> str:
        """
        Generate text using Groq API (very fast and cheap).
        
        Models:
        - llama3-8b-8192: $0.05/1M input tokens, $0.08/1M output tokens
        - llama3-70b-8192: $0.59/1M input tokens, $0.79/1M output tokens
        - mixtral-8x7b-32768: $0.24/1M input tokens, $0.24/1M output tokens
        """
        if not self.groq_api_key:
            raise ValueError("Groq API key not configured")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.1
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Groq API error: {response.text}")
    
    async def generate_text_anthropic(self, prompt: str, model: str = "claude-3-haiku-20240307") -> str:
        """
        Generate text using Anthropic Claude API.
        
        Models:
        - claude-3-haiku-20240307: $0.25/1M input tokens, $1.25/1M output tokens
        - claude-3-sonnet-20240229: $3/1M input tokens, $15/1M output tokens
        """
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": model,
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["content"][0]["text"]
            else:
                raise Exception(f"Anthropic API error: {response.text}")
    
    async def generate_text_ollama(self, prompt: str, model: str = "llama3.1:8b") -> str:
        """
        Generate text using local Ollama (FREE!).
        
        Popular models:
        - llama3.1:8b: Good balance of speed and quality
        - llama3.1:70b: High quality but slower
        - mistral: Fast and efficient
        - codellama: Good for code-related tasks
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1500
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["response"]
            else:
                raise Exception(f"Ollama API error: {response.text}")
    
    async def generate_text_smart_routing(self, prompt: str, task_type: str = "general") -> str:
        """
        Smart routing to choose the best/cheapest model for the task.
        
        Args:
            prompt: Input prompt
            task_type: Type of task (ocr, validation, summary, categorization)
        """
        try:
            # For simple tasks, use cheapest options first
            if task_type in ["validation", "categorization"]:
                # Try Ollama first (free)
                try:
                    return await self.generate_text_ollama(prompt, "llama3.1:8b")
                except:
                    pass
                
                # Try Groq (very cheap)
                try:
                    return await self.generate_text_groq(prompt, "llama3-8b-8192")
                except:
                    pass
            
            # For complex tasks, use better models
            elif task_type in ["summary", "ocr"]:
                # Try Groq with better model
                try:
                    return await self.generate_text_groq(prompt, "llama3-70b-8192")
                except:
                    pass
                
                # Try Anthropic Haiku (good quality, reasonable price)
                try:
                    return await self.generate_text_anthropic(prompt, "claude-3-haiku-20240307")
                except:
                    pass
            
            # Fallback to OpenAI if all else fails
            raise Exception("All alternative LLM services failed")
            
        except Exception as e:
            logger.error(f"Smart routing failed: {e}")
            raise


class CostOptimizedLLMService:
    """Cost-optimized LLM service that combines multiple providers."""
    
    def __init__(self):
        """Initialize cost-optimized service."""
        self.alternative_service = AlternativeLLMService()
        self.use_alternatives = getattr(settings, 'use_alternative_llms', False)
    
    async def generate_text(self, prompt: str, task_type: str = "general") -> str:
        """
        Generate text using the most cost-effective method.
        
        Cost hierarchy (cheapest to most expensive):
        1. Ollama (Local, FREE)
        2. Groq (Very cheap, fast)
        3. Anthropic Haiku (Cheap, good quality)
        4. OpenAI GPT-3.5-turbo (Moderate cost)
        5. OpenAI GPT-4 (Expensive)
        """
        if self.use_alternatives:
            try:
                return await self.alternative_service.generate_text_smart_routing(prompt, task_type)
            except Exception as e:
                logger.warning(f"Alternative LLM failed, falling back to OpenAI: {e}")
        
        # Fallback to OpenAI
        from app.services.llm_service import LLMService
        llm_service = LLMService()
        return await llm_service.generate_text(prompt)


# Cost comparison (per 1M tokens):
COST_COMPARISON = {
    "ollama": {"input": 0.0, "output": 0.0, "note": "Local, requires GPU"},
    "groq_llama3_8b": {"input": 0.05, "output": 0.08, "note": "Very fast"},
    "groq_llama3_70b": {"input": 0.59, "output": 0.79, "note": "High quality"},
    "groq_mixtral": {"input": 0.24, "output": 0.24, "note": "Balanced"},
    "anthropic_haiku": {"input": 0.25, "output": 1.25, "note": "Good quality"},
    "anthropic_sonnet": {"input": 3.0, "output": 15.0, "note": "Very high quality"},
    "openai_gpt35": {"input": 0.50, "output": 1.50, "note": "OpenAI standard"},
    "openai_gpt4": {"input": 30.0, "output": 60.0, "note": "OpenAI premium"},
    "openai_gpt4o": {"input": 5.0, "output": 15.0, "note": "OpenAI vision"},
    "openai_gpt4o_mini": {"input": 0.15, "output": 0.60, "note": "OpenAI cheap vision"}
}
