"""
Invoice data models using Pydantic.
"""

from datetime import datetime, date, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class InvoiceType(str, Enum):
    """Invoice type enumeration."""
    PURCHASE = "purchase"
    SALES = "sales"
    SERVICE = "service"
    UTILITY = "utility"
    RENT = "rent"
    OTHER = "other"


class Currency(str, Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    TRY = "TRY"
    INR = "INR"
    OTHER = "OTHER"


class ExpenseCategory(str, Enum):
    """Expense categories for accounting."""
    OFFICE_SUPPLIES = "office_supplies"
    TRAVEL = "travel"
    MEALS = "meals"
    UTILITIES = "utilities"
    RENT = "rent"
    MARKETING = "marketing"
    PROFESSIONAL_SERVICES = "professional_services"
    EQUIPMENT = "equipment"
    SOFTWARE = "software"
    INSURANCE = "insurance"
    TAXES = "taxes"
    OTHER = "other"


class ConfidenceLevel(str, Enum):
    """Confidence levels for extracted data."""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"      # 80-89%
    MEDIUM = "medium"  # 70-79%
    LOW = "low"        # 50-69%
    VERY_LOW = "very_low"  # <50%


class VendorInfo(BaseModel):
    """Vendor/supplier information."""
    name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    tax_id: Optional[str] = None
    website: Optional[str] = None


class LineItem(BaseModel):
    """Individual line item on invoice."""
    description: str
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    total_price: Optional[Decimal] = None
    category: Optional[ExpenseCategory] = None
    category_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @field_validator('quantity', 'unit_price', 'total_price', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert numeric values to Decimal for precision."""
        if v is None:
            return v
        return Decimal(str(v))


class TaxInfo(BaseModel):
    """Tax information."""
    tax_rate: Optional[Decimal] = None
    tax_amount: Optional[Decimal] = None
    tax_type: Optional[str] = None  # VAT, GST, Sales Tax, etc.
    
    @field_validator('tax_rate', 'tax_amount', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert numeric values to Decimal for precision."""
        if v is None:
            return v
        return Decimal(str(v))


class PaymentInfo(BaseModel):
    """Payment information."""
    payment_method: Optional[str] = None
    payment_terms: Optional[str] = None
    due_date: Optional[date] = None
    payment_status: Optional[str] = None


class ProcessingPhase(BaseModel):
    """Information about a processing phase."""
    phase_name: str
    status: str  # success, error, warning
    duration_ms: Optional[int] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    details: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CategoryData(BaseModel):
    """Expense categorization data."""
    primary_category: str
    sub_categories: List[str] = Field(default_factory=list)
    expense_type: str
    tax_category: str
    priority: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class ValidationResult(BaseModel):
    """Data validation results."""
    is_valid: bool
    validation_errors: List[Dict[str, Any]] = Field(default_factory=list)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    should_use_fallback: bool = False
    fallback_reason: Optional[str] = None
    overall_quality_score: float = Field(ge=0.0, le=1.0)


class ProcessingDecision(BaseModel):
    """Processing method decision."""
    method: str  # "standard" or "fallback_required"
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class InvoiceData(BaseModel):
    """Complete invoice data structure."""
    
    # Basic Information
    invoice_number: Optional[str] = None
    invoice_date: Optional[date] = None
    invoice_type: Optional[InvoiceType] = None
    
    # Vendor Information
    vendor: VendorInfo = Field(default_factory=VendorInfo)
    
    # Financial Information
    currency: Optional[Currency] = None
    subtotal: Optional[Decimal] = None
    tax_info: Optional[TaxInfo] = None
    total_amount: Optional[Decimal] = None
    
    # Line Items
    line_items: List[LineItem] = Field(default_factory=list)
    
    # Payment Information
    payment_info: Optional[PaymentInfo] = None
    
    # Additional Information
    notes: Optional[str] = None
    purchase_order_number: Optional[str] = None
    
    # Metadata
    extracted_text: Optional[str] = None  # Raw OCR text
    processing_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    original_filename: Optional[str] = None
    processing_phases: List[ProcessingPhase] = Field(default_factory=list)
    
    @field_validator('subtotal', 'total_amount', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert numeric values to Decimal for precision."""
        if v is None:
            return v
        return Decimal(str(v))


class ValidationIssue(BaseModel):
    """Validation issue found during processing."""
    field: str
    issue_type: str
    message: str
    severity: str  # error, warning, info
    suggested_fix: Optional[str] = None


class ProcessedInvoice(BaseModel):
    """Fully processed invoice with metadata."""
    
    # Core invoice data
    invoice_data: InvoiceData
    
    # Processing metadata
    processing_phases: List[ProcessingPhase] = Field(default_factory=list)
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.VERY_LOW
    
    # Validation results
    validation_issues: List[ValidationIssue] = Field(default_factory=list)
    is_valid: bool = True
    
    # Processing statistics
    total_processing_time_ms: Optional[int] = None
    ocr_method_used: Optional[str] = None  # gpt4_vision, textract
    
    # File information
    original_filename: Optional[str] = None
    file_size_bytes: Optional[int] = None

    # Cost and usage metadata
    processing_metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('overall_confidence')
    @classmethod
    def set_confidence_level(cls, v, info):
        """Set confidence level based on overall confidence score."""
        # Note: In Pydantic V2, we handle this differently
        return v

    def update_confidence_level(self):
        """Update confidence level based on overall confidence score."""
        if self.overall_confidence >= 0.9:
            self.confidence_level = ConfidenceLevel.VERY_HIGH
        elif self.overall_confidence >= 0.8:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.overall_confidence >= 0.7:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.overall_confidence >= 0.5:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.VERY_LOW


class FinalResponse(BaseModel):
    """Final API response with all analysis results."""
    invoice_summary: Dict[str, Any]
    invoice_data: InvoiceData
    category_data: CategoryData
    validation_result: ValidationResult
    key_insights: List[str] = Field(default_factory=list)
    action_required: Dict[str, Any] = Field(default_factory=dict)
    processing_decision: ProcessingDecision
    metadata: Dict[str, Any] = Field(default_factory=dict)
