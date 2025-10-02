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
    ollama_base_url: str = "http://localhost:11434"
    
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

    # AI Processing Configuration
    confidence_threshold: float = 0.80
    ocr_quality_threshold: float = 0.70
    high_value_threshold: float = 10000.0
    enable_textract_fallback: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
