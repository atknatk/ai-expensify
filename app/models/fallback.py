"""
Fallback strategy models and enums for invoice processing.
"""

from enum import Enum
from typing import Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
from pydantic import BaseModel

if TYPE_CHECKING:
    from app.models.invoice import ProcessedInvoice
    from app.models.response import InvoiceAnalysisResponse

from app.constants import FallbackReason


class FallbackTrigger(str, Enum):
    """Enumeration of fallback trigger types."""
    CONFIDENCE_BASED = "confidence_based"
    SUMMARY_BASED = "summary_based"
    MANUAL = "manual"


class FallbackOutcome(str, Enum):
    """Enumeration of fallback execution outcomes."""
    IMPROVED = "improved"
    NO_IMPROVEMENT = "no_improvement"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FallbackDecision:
    """Represents a decision to execute fallback processing."""
    
    trigger: FallbackTrigger
    reason: str
    confidence_score: Optional[float] = None
    validation_issues_count: Optional[int] = None
    summary_recommendation: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """String representation of fallback decision."""
        return f"FallbackDecision(trigger={self.trigger.value}, reason='{self.reason}')"


@dataclass
class FallbackResult:
    """Represents the result of fallback processing execution."""

    decision: FallbackDecision
    outcome: FallbackOutcome
    original_confidence: float
    fallback_confidence: Optional[float] = None
    original_issues_count: int = 0
    fallback_issues_count: Optional[int] = None
    execution_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    validated_data: Optional["ProcessedInvoice"] = None  # ProcessedInvoice instance if successful
    final_result: Optional["InvoiceAnalysisResponse"] = None    # InvoiceAnalysisResponse instance if successful
    
    @property
    def was_successful(self) -> bool:
        """Check if fallback was successful and improved results."""
        return self.outcome == FallbackOutcome.IMPROVED
    
    @property
    def confidence_improvement(self) -> Optional[float]:
        """Calculate confidence improvement from fallback."""
        if self.fallback_confidence is None:
            return None
        return self.fallback_confidence - self.original_confidence
    
    @property
    def issues_improvement(self) -> Optional[int]:
        """Calculate validation issues improvement from fallback."""
        if self.fallback_issues_count is None:
            return None
        return self.original_issues_count - self.fallback_issues_count
    
    def __str__(self) -> str:
        """String representation of fallback result."""
        return (f"FallbackResult(outcome={self.outcome.value}, "
                f"confidence_improvement={self.confidence_improvement}, "
                f"issues_improvement={self.issues_improvement})")


class FallbackMetrics(BaseModel):
    """Metrics for fallback processing performance."""
    
    total_attempts: int = 0
    successful_improvements: int = 0
    failed_attempts: int = 0
    average_execution_time_ms: float = 0.0
    confidence_based_triggers: int = 0
    summary_based_triggers: int = 0
    manual_triggers: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate fallback success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_improvements / self.total_attempts
    
    @property
    def failure_rate(self) -> float:
        """Calculate fallback failure rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.failed_attempts / self.total_attempts


def create_confidence_fallback_decision(
    confidence_score: float,
    threshold: float
) -> FallbackDecision:
    """Create a confidence-based fallback decision."""
    return FallbackDecision(
        trigger=FallbackTrigger.CONFIDENCE_BASED,
        reason=f"Confidence score {confidence_score:.2f} below threshold {threshold}",
        confidence_score=confidence_score
    )


def create_summary_fallback_decision(
    summary_recommendation: Dict[str, Any],
    validation_issues_count: int
) -> FallbackDecision:
    """Create a summary-based fallback decision."""
    reason = summary_recommendation.get('reason', 'Summary recommends fallback processing')
    return FallbackDecision(
        trigger=FallbackTrigger.SUMMARY_BASED,
        reason=reason,
        validation_issues_count=validation_issues_count,
        summary_recommendation=summary_recommendation
    )


def create_manual_fallback_decision(reason: str = "Manual fallback requested") -> FallbackDecision:
    """Create a manual fallback decision."""
    return FallbackDecision(
        trigger=FallbackTrigger.MANUAL,
        reason=reason
    )
