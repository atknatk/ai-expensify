"""
API response models.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from app.models.invoice import ProcessedInvoice, ConfidenceLevel


class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())


class HealthResponse(BaseResponse):
    """Health check response."""
    status: str
    version: str


class ErrorResponse(BaseModel):
    """Error response model."""
    error: bool = True
    message: str
    status_code: int
    timestamp: str = Field(default_factory=lambda: __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    details: Optional[Dict[str, Any]] = None


class InvoiceAnalysisResponse(BaseResponse):
    """Invoice analysis response."""
    
    # Analysis results
    invoice: ProcessedInvoice
    
    # Summary information
    summary: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing statistics
    processing_stats: Dict[str, Any] = Field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Auto-generate summary if not provided
        if not self.summary and hasattr(self, 'invoice'):
            self.summary = self._generate_summary()
        
        # Auto-generate processing stats if not provided
        if not self.processing_stats and hasattr(self, 'invoice'):
            self.processing_stats = self._generate_processing_stats()
        
        # Auto-generate recommendations if not provided
        if not self.recommendations and hasattr(self, 'invoice'):
            self.recommendations = self._generate_recommendations()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary information from invoice data."""
        invoice_data = self.invoice.invoice_data
        
        return {
            "vendor_name": invoice_data.vendor.name,
            "invoice_number": invoice_data.invoice_number,
            "invoice_date": invoice_data.invoice_date.isoformat() if invoice_data.invoice_date else None,
            "total_amount": float(invoice_data.total_amount) if invoice_data.total_amount else None,
            "currency": invoice_data.currency,
            "line_items_count": len(invoice_data.line_items),
            "confidence_level": self.invoice.confidence_level,
            "overall_confidence": self.invoice.overall_confidence,
            "validation_issues_count": len(self.invoice.validation_issues),
            "is_valid": self.invoice.is_valid
        }
    
    def _generate_processing_stats(self) -> Dict[str, Any]:
        """Generate processing statistics."""
        return {
            "total_processing_time_ms": self.invoice.total_processing_time_ms,
            "ocr_method_used": self.invoice.ocr_method_used,
            "phases_completed": len(self.invoice.processing_phases),
            "phases_with_errors": len([p for p in self.invoice.processing_phases if p.errors]),
            "phases_with_warnings": len([p for p in self.invoice.processing_phases if p.warnings]),
            "original_filename": self.invoice.original_filename,
            "file_size_bytes": self.invoice.file_size_bytes
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Confidence-based recommendations
        if self.invoice.confidence_level == ConfidenceLevel.LOW:
            recommendations.append("Consider manual review due to low confidence score")
        elif self.invoice.confidence_level == ConfidenceLevel.VERY_LOW:
            recommendations.append("Manual review strongly recommended due to very low confidence")
        
        # Validation-based recommendations
        error_issues = [issue for issue in self.invoice.validation_issues if issue.severity == "error"]
        if error_issues:
            recommendations.append("Address validation errors before processing")
        
        warning_issues = [issue for issue in self.invoice.validation_issues if issue.severity == "warning"]
        if warning_issues:
            recommendations.append("Review validation warnings for accuracy")
        
        # Data completeness recommendations
        invoice_data = self.invoice.invoice_data
        
        if not invoice_data.vendor.name:
            recommendations.append("Vendor name is missing - consider manual entry")
        
        if not invoice_data.invoice_number:
            recommendations.append("Invoice number is missing - verify document completeness")
        
        if not invoice_data.invoice_date:
            recommendations.append("Invoice date is missing - check document quality")
        
        if not invoice_data.total_amount:
            recommendations.append("Total amount is missing - verify calculation accuracy")
        
        # Line items recommendations
        if not invoice_data.line_items:
            recommendations.append("No line items detected - consider manual entry")
        else:
            uncategorized_items = [item for item in invoice_data.line_items if not item.category]
            if uncategorized_items:
                recommendations.append(f"{len(uncategorized_items)} line items need category assignment")
        
        return recommendations


class BatchAnalysisResponse(BaseResponse):
    """Batch analysis response for multiple invoices."""
    
    results: List[InvoiceAnalysisResponse]
    batch_stats: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Auto-generate batch stats if not provided
        if not self.batch_stats and self.results:
            self.batch_stats = self._generate_batch_stats()
    
    def _generate_batch_stats(self) -> Dict[str, Any]:
        """Generate batch processing statistics."""
        total_invoices = len(self.results)
        successful_analyses = len([r for r in self.results if r.success])
        
        confidence_levels = [r.invoice.confidence_level for r in self.results if r.success]
        avg_confidence = sum(r.invoice.overall_confidence for r in self.results if r.success) / successful_analyses if successful_analyses > 0 else 0
        
        return {
            "total_invoices": total_invoices,
            "successful_analyses": successful_analyses,
            "failed_analyses": total_invoices - successful_analyses,
            "success_rate": successful_analyses / total_invoices if total_invoices > 0 else 0,
            "average_confidence": avg_confidence,
            "confidence_distribution": {
                level: confidence_levels.count(level) for level in ConfidenceLevel
            }
        }
