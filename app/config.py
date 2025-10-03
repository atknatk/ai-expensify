"""
Application configuration settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
from pydantic import validator


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
    ollama_model: str = "llava:13b"
    ollama_timeout: int = 30
    # Note: ollama_enabled is defined in the tier configuration section below
    
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

    # AI Processing Configuration - Dynamic 3-Tier Fallback System

    # Tier Configuration - Define order and enabled tiers
    ocr_tier_order: str = "ollama,openai,textract"  # Comma-separated tier order
    ocr_enabled_tiers: str = "ollama,openai,textract"  # Comma-separated enabled tiers

    # Individual Tier Enable/Disable Flags
    ollama_enabled: bool = True
    openai_ocr_enabled: bool = True
    textract_enabled: bool = True

    # Confidence Thresholds per Tier
    ollama_confidence_threshold: float = 0.80
    openai_confidence_threshold: float = 0.75
    textract_confidence_threshold: float = 0.85

    # Legacy/Additional Configuration
    high_value_threshold: float = 10000.0  # EUR - triggers direct Textract for critical invoices
    enable_textract_fallback: bool = True

    # Legacy settings (for backward compatibility)
    confidence_threshold: float = 0.80
    ocr_quality_threshold: float = 0.70

    @validator('ocr_tier_order')
    def validate_tier_order(cls, v):
        """Validate OCR tier order configuration."""
        if not v:
            return "ollama,openai,textract"  # Default fallback

        valid_tiers = {"ollama", "openai", "textract"}
        tiers = [tier.strip().lower() for tier in v.split(",")]

        # Check for invalid tiers
        invalid_tiers = set(tiers) - valid_tiers
        if invalid_tiers:
            raise ValueError(f"Invalid OCR tiers: {invalid_tiers}. Valid tiers: {valid_tiers}")

        # Check for duplicates
        if len(tiers) != len(set(tiers)):
            raise ValueError(f"Duplicate tiers found in OCR_TIER_ORDER: {v}")

        return v.lower()

    @validator('ocr_enabled_tiers')
    def validate_enabled_tiers(cls, v):
        """Validate OCR enabled tiers configuration."""
        if not v:
            return "ollama,openai,textract"  # Default fallback

        valid_tiers = {"ollama", "openai", "textract"}
        tiers = [tier.strip().lower() for tier in v.split(",")]

        # Check for invalid tiers
        invalid_tiers = set(tiers) - valid_tiers
        if invalid_tiers:
            raise ValueError(f"Invalid OCR enabled tiers: {invalid_tiers}. Valid tiers: {valid_tiers}")

        # Check for duplicates
        if len(tiers) != len(set(tiers)):
            raise ValueError(f"Duplicate tiers found in OCR_ENABLED_TIERS: {v}")

        return v.lower()

    def get_tier_order(self) -> List[str]:
        """Get the ordered list of OCR tiers to try."""
        return [tier.strip().lower() for tier in self.ocr_tier_order.split(",")]

    def get_enabled_tiers(self) -> List[str]:
        """Get the list of enabled OCR tiers."""
        return [tier.strip().lower() for tier in self.ocr_enabled_tiers.split(",")]

    def get_active_tier_order(self) -> List[str]:
        """Get the ordered list of active (enabled) OCR tiers to try."""
        tier_order = self.get_tier_order()
        enabled_tiers = set(self.get_enabled_tiers())

        # Return tiers in order, but only if they're enabled
        active_tiers = [tier for tier in tier_order if tier in enabled_tiers]

        if not active_tiers:
            raise ValueError("No active OCR tiers found. Check OCR_TIER_ORDER and OCR_ENABLED_TIERS configuration.")

        return active_tiers

    def is_tier_enabled(self, tier: str) -> bool:
        """Check if a specific tier is enabled."""
        tier = tier.lower()
        enabled_tiers = self.get_enabled_tiers()

        # Check both the enabled tiers list and individual flags
        if tier == "ollama":
            return tier in enabled_tiers and self.ollama_enabled
        elif tier == "openai":
            return tier in enabled_tiers and self.openai_ocr_enabled
        elif tier == "textract":
            return tier in enabled_tiers and self.textract_enabled
        else:
            return False

    def get_tier_confidence_threshold(self, tier: str) -> float:
        """Get the confidence threshold for a specific tier."""
        tier = tier.lower()
        if tier == "ollama":
            return self.ollama_confidence_threshold
        elif tier == "openai":
            return self.openai_confidence_threshold
        elif tier == "textract":
            return self.textract_confidence_threshold
        else:
            return 0.75  # Default threshold

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
