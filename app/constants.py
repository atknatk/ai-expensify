"""
Application constants and configuration values.
"""

from typing import Final

# Processing thresholds
CONFIDENCE_THRESHOLD_LOW: Final[float] = 0.6
"""Confidence threshold below which fallback processing is triggered."""

VALIDATION_ISSUE_TOLERANCE: Final[float] = 0.05
"""Tolerance percentage for validation calculations (5%)."""

TAX_TOLERANCE: Final[float] = 0.3
"""Tolerance percentage for tax calculations when no subtotal available (30%)."""

# Processing phase names
class ProcessingPhase:
    """Processing phase identifiers."""
    OCR_EXTRACTION: Final[str] = "OCR Data Extraction"
    CATEGORIZATION: Final[str] = "Expense Categorization"
    VALIDATION: Final[str] = "Data Validation"
    SUMMARY_GENERATION: Final[str] = "Final Summary Generation"

# Fallback reasons
class FallbackReason:
    """Standardized fallback trigger reasons."""
    LOW_CONFIDENCE: Final[str] = "low_confidence"
    SUMMARY_RECOMMENDATION: Final[str] = "summary_recommendation"
    MANUAL_OVERRIDE: Final[str] = "manual_override"

# Processing decision methods
class ProcessingDecisionMethod:
    """Processing decision method identifiers."""
    STANDARD: Final[str] = "standard"
    FALLBACK_REQUIRED: Final[str] = "fallback_required"
    MANUAL_REVIEW: Final[str] = "manual_review"

# Logging context keys
class LogContext:
    """Structured logging context keys."""
    REQUEST_ID: Final[str] = "request_id"
    FILENAME: Final[str] = "invoice_filename"
    PHASE: Final[str] = "processing_phase"
    CONFIDENCE: Final[str] = "confidence_score"
    PROCESSING_TIME: Final[str] = "processing_time_ms"
    FALLBACK_REASON: Final[str] = "fallback_reason"
    COST: Final[str] = "cost_usd"

# File processing limits
MAX_RETRY_ATTEMPTS: Final[int] = 3
"""Maximum number of retry attempts for failed operations."""

PROCESSING_TIMEOUT_SECONDS: Final[int] = 300
"""Maximum processing time in seconds before timeout."""

# Cost tracking
COST_PRECISION_DECIMALS: Final[int] = 8
"""Number of decimal places for cost calculations."""
