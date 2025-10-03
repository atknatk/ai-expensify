"""
Application configuration settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None

    # Model Configuration (Cost Optimization)
    ocr_model: str = "gpt-4o-mini"  # Cheaper alternative to gpt-4o
    text_model: str = "gpt-3.5-turbo"  # Much cheaper for text processing
    vision_model: str = "gpt-4o-mini"  # Cheaper vision model

    # Use cheaper models for non-critical tasks
    use_cheap_models: bool = True

    # Alternative LLM Services (Even Cheaper!)
    use_alternative_llms: bool = False
    groq_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Ollama Configuration (Tier 1 - Primary)
    ollama_base_url: str = "http://192.168.0.201:11434"
    ollama_model: str = "llava:7b"
    ollama_timeout: int = 30
    ollama_enabled: bool = True
    
    # AWS Configuration (Optional)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    
    # Application Configuration
    debug: bool = False
    log_level: str = "INFO"
    max_file_size_mb: int = 10
    
    # API Configuration
    api_title: str = "Invoice Analyzer API"
    api_version: str = "1.0.0"
    api_description: str = "AI-powered invoice analysis with OCR and LLM integration"
    
    # Processing Configuration
    max_retries: int = 3
    timeout_seconds: int = 30

    # AI Processing Configuration - 3-Tier Fallback System
    # Tier 1: Ollama LLaVA confidence threshold
    ollama_confidence_threshold: float = 0.80
    # Tier 2: OpenAI GPT-4 Vision confidence threshold
    openai_confidence_threshold: float = 0.75
    # Tier 3: AWS Textract (always used as last resort)
    high_value_threshold: float = 10000.0  # EUR - triggers direct Textract for critical invoices
    enable_textract_fallback: bool = True

    # Legacy settings (for backward compatibility)
    confidence_threshold: float = 0.80
    ocr_quality_threshold: float = 0.70
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
