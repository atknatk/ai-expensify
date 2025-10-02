"""
Cost tracking models for API usage monitoring.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ServiceProvider(str, Enum):
    """Service providers for cost tracking."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"
    AWS_TEXTRACT = "aws_textract"


class ModelUsage(BaseModel):
    """Model usage tracking."""
    provider: ServiceProvider
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 1
    duration_ms: int = 0
    cost_usd: float = 0.0
    
    
class CostBreakdown(BaseModel):
    """Detailed cost breakdown."""
    ocr_cost: float = 0.0
    categorization_cost: float = 0.0
    validation_cost: float = 0.0
    summary_cost: float = 0.0
    total_cost: float = 0.0
    
    
class ProcessingMetadata(BaseModel):
    """Processing metadata including costs and model usage."""
    
    # Model usage tracking
    models_used: List[ModelUsage] = Field(default_factory=list)
    
    # Cost breakdown
    cost_breakdown: CostBreakdown = Field(default_factory=CostBreakdown)
    
    # Performance metrics
    total_processing_time_ms: int = 0
    ocr_method_used: str = "unknown"
    fallback_used: bool = False
    
    # File metadata
    original_filename: Optional[str] = None
    file_size_bytes: Optional[int] = None
    
    # Processing timestamp
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def add_model_usage(self, usage: ModelUsage):
        """Add model usage to tracking."""
        self.models_used.append(usage)
        
        # Update cost breakdown based on phase
        if "gpt-4" in usage.model_name.lower() or "vision" in usage.model_name.lower():
            self.cost_breakdown.ocr_cost += usage.cost_usd
        elif "categoriz" in usage.model_name.lower():
            self.cost_breakdown.categorization_cost += usage.cost_usd
        elif "validat" in usage.model_name.lower():
            self.cost_breakdown.validation_cost += usage.cost_usd
        elif "summary" in usage.model_name.lower():
            self.cost_breakdown.summary_cost += usage.cost_usd
            
        # Update total cost
        self.cost_breakdown.total_cost = (
            self.cost_breakdown.ocr_cost +
            self.cost_breakdown.categorization_cost +
            self.cost_breakdown.validation_cost +
            self.cost_breakdown.summary_cost
        )


# Cost per 1M tokens (USD)
MODEL_COSTS = {
    # OpenAI Models
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    
    # Groq Models
    "llama3-8b-8192": {"input": 0.05, "output": 0.08},
    "llama3-70b-8192": {"input": 0.59, "output": 0.79},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    
    # Anthropic Models
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
    
    # AWS Textract (per 1000 pages)
    "textract-analyze-expense": {"input": 1.50, "output": 0.0},
    
    # Ollama (Free)
    "llama3.1:8b": {"input": 0.0, "output": 0.0},
    "llama3.1:70b": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost for model usage.
    
    Args:
        model_name: Name of the model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        float: Cost in USD
    """
    if model_name not in MODEL_COSTS:
        return 0.0
    
    costs = MODEL_COSTS[model_name]
    
    # Calculate cost per million tokens
    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    
    return input_cost + output_cost


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Rough approximation: 1 token ≈ 4 characters
    
    Args:
        text: Input text
        
    Returns:
        int: Estimated token count
    """
    return len(text) // 4


class CostTracker:
    """Cost tracking utility."""
    
    def __init__(self):
        self.session_costs: List[ModelUsage] = []
    
    def track_usage(self, provider: ServiceProvider, model_name: str, 
                   input_text: str, output_text: str, duration_ms: int) -> ModelUsage:
        """
        Track model usage and calculate cost.
        
        Args:
            provider: Service provider
            model_name: Model name
            input_text: Input text
            output_text: Output text
            duration_ms: Processing duration
            
        Returns:
            ModelUsage: Usage tracking object
        """
        input_tokens = estimate_tokens(input_text)
        output_tokens = estimate_tokens(output_text)
        cost = calculate_cost(model_name, input_tokens, output_tokens)
        
        usage = ModelUsage(
            provider=provider,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            cost_usd=cost
        )
        
        self.session_costs.append(usage)
        return usage
    
    def get_session_total(self) -> float:
        """Get total cost for current session."""
        return sum(usage.cost_usd for usage in self.session_costs)
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by provider."""
        breakdown = {}
        for usage in self.session_costs:
            provider = usage.provider.value
            if provider not in breakdown:
                breakdown[provider] = 0.0
            breakdown[provider] += usage.cost_usd
        return breakdown


# Global cost tracker instance
cost_tracker = CostTracker()
