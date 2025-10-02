"""
Validation service for invoice data accuracy and completeness.
"""

import logging
import time
import json
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Dict, Any

from app.models.invoice import (
    InvoiceData,
    ProcessedInvoice,
    ValidationIssue,
    ProcessingPhase,
    Currency,
    InvoiceType
)
from app.prompts.validation import VALIDATION_PROMPT
from app.services.llm_service import LLMService


logger = logging.getLogger(__name__)


class ValidationService:
    """Service for validating invoice data accuracy and completeness."""
    
    def __init__(self):
        """Initialize validation service."""
        self.llm_service = LLMService()
    
    async def validate_data(self, invoice_data: InvoiceData) -> ProcessedInvoice:
        """
        Validate invoice data for accuracy and completeness.
        
        Args:
            invoice_data: Invoice data to validate
            
        Returns:
            ProcessedInvoice: Processed invoice with validation results
        """
        start_time = time.time()
        
        try:
            logger.info("Starting invoice data validation")
            
            # Initialize processed invoice
            processed_invoice = ProcessedInvoice(
                invoice_data=invoice_data,
                validation_issues=[],
                is_valid=True
            )
            
            # Run validation checks
            validation_issues = []
            
            # Basic data validation
            validation_issues.extend(self._validate_basic_fields(invoice_data))
            
            # Vendor validation
            validation_issues.extend(self._validate_vendor_info(invoice_data))
            
            # Financial validation
            validation_issues.extend(self._validate_financial_data(invoice_data))
            
            # Line items validation
            validation_issues.extend(self._validate_line_items(invoice_data))
            
            # Date validation
            validation_issues.extend(self._validate_dates(invoice_data))
            
            # Cross-field validation
            validation_issues.extend(self._validate_cross_fields(invoice_data))

            # LLM-based advanced validation (temporarily disabled for stability)
            # llm_validation_issues = await self._validate_with_llm(invoice_data)
            # validation_issues.extend(llm_validation_issues)

            # Set validation results
            processed_invoice.validation_issues = validation_issues
            
            # Determine if invoice is valid (no error-level issues)
            error_issues = [issue for issue in validation_issues if issue.severity == "error"]
            processed_invoice.is_valid = len(error_issues) == 0
            
            # Calculate validation confidence
            validation_confidence = self._calculate_validation_confidence(validation_issues, invoice_data)
            processed_invoice.overall_confidence = validation_confidence

            # Update confidence level based on overall confidence
            processed_invoice.update_confidence_level()

            # Add processing phase
            processing_time = int((time.time() - start_time) * 1000)
            processed_invoice.processing_phases.append(
                ProcessingPhase(
                    phase_name="Validation",
                    status="success" if processed_invoice.is_valid else "warning",
                    duration_ms=processing_time,
                    confidence=validation_confidence,
                    details={
                        "total_issues": len(validation_issues),
                        "error_issues": len(error_issues),
                        "warning_issues": len([issue for issue in validation_issues if issue.severity == "warning"]),
                        "info_issues": len([issue for issue in validation_issues if issue.severity == "info"])
                    },
                    warnings=[issue.message for issue in validation_issues if issue.severity == "warning"],
                    errors=[issue.message for issue in validation_issues if issue.severity == "error"]
                )
            )
            
            logger.info(f"Validation completed: {len(validation_issues)} issues found, valid: {processed_invoice.is_valid}")
            return processed_invoice
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            
            # Create processed invoice with error
            processed_invoice = ProcessedInvoice(
                invoice_data=invoice_data,
                validation_issues=[
                    ValidationIssue(
                        field="validation_process",
                        issue_type="system_error",
                        message=f"Validation process failed: {str(e)}",
                        severity="error"
                    )
                ],
                is_valid=False
            )
            
            # Add error phase
            processing_time = int((time.time() - start_time) * 1000)
            processed_invoice.processing_phases.append(
                ProcessingPhase(
                    phase_name="Validation",
                    status="error",
                    duration_ms=processing_time,
                    confidence=0.0,
                    errors=[str(e)]
                )
            )
            
            return processed_invoice
    
    def _validate_basic_fields(self, invoice_data: InvoiceData) -> List[ValidationIssue]:
        """Validate basic required fields."""
        issues = []
        
        # Invoice number validation
        if not invoice_data.invoice_number:
            issues.append(ValidationIssue(
                field="invoice_number",
                issue_type="missing_required",
                message="Invoice number is missing",
                severity="error",
                suggested_fix="Manually enter invoice number from document"
            ))
        elif len(invoice_data.invoice_number.strip()) < 2:
            issues.append(ValidationIssue(
                field="invoice_number",
                issue_type="invalid_format",
                message="Invoice number appears too short",
                severity="warning",
                suggested_fix="Verify invoice number accuracy"
            ))
        
        # Invoice type validation
        if invoice_data.invoice_type and invoice_data.invoice_type not in InvoiceType:
            issues.append(ValidationIssue(
                field="invoice_type",
                issue_type="invalid_value",
                message=f"Invalid invoice type: {invoice_data.invoice_type}",
                severity="warning",
                suggested_fix="Select valid invoice type from available options"
            ))
        
        return issues
    
    def _validate_vendor_info(self, invoice_data: InvoiceData) -> List[ValidationIssue]:
        """Validate vendor information."""
        issues = []
        
        # Vendor name validation
        if not invoice_data.vendor.name:
            issues.append(ValidationIssue(
                field="vendor.name",
                issue_type="missing_required",
                message="Vendor name is missing",
                severity="error",
                suggested_fix="Extract vendor name from document header"
            ))
        elif len(invoice_data.vendor.name.strip()) < 2:
            issues.append(ValidationIssue(
                field="vendor.name",
                issue_type="invalid_format",
                message="Vendor name appears too short",
                severity="warning"
            ))
        
        # Email validation
        if invoice_data.vendor.email:
            if "@" not in invoice_data.vendor.email or "." not in invoice_data.vendor.email:
                issues.append(ValidationIssue(
                    field="vendor.email",
                    issue_type="invalid_format",
                    message="Vendor email format appears invalid",
                    severity="warning",
                    suggested_fix="Verify email address format"
                ))
        
        return issues
    
    def _validate_financial_data(self, invoice_data: InvoiceData) -> List[ValidationIssue]:
        """Validate financial data."""
        issues = []
        
        # Total amount validation
        if not invoice_data.total_amount:
            issues.append(ValidationIssue(
                field="total_amount",
                issue_type="missing_required",
                message="Total amount is missing",
                severity="error",
                suggested_fix="Extract total amount from document"
            ))
        elif invoice_data.total_amount <= 0:
            issues.append(ValidationIssue(
                field="total_amount",
                issue_type="invalid_value",
                message="Total amount must be greater than zero",
                severity="error",
                suggested_fix="Verify total amount calculation"
            ))
        
        # Currency validation
        if invoice_data.currency and invoice_data.currency not in Currency:
            issues.append(ValidationIssue(
                field="currency",
                issue_type="invalid_value",
                message=f"Unsupported currency: {invoice_data.currency}",
                severity="warning",
                suggested_fix="Use supported currency codes or select OTHER"
            ))
        
        # Subtotal vs total validation
        if invoice_data.subtotal and invoice_data.total_amount:
            if invoice_data.subtotal > invoice_data.total_amount:
                issues.append(ValidationIssue(
                    field="subtotal",
                    issue_type="logical_error",
                    message="Subtotal cannot be greater than total amount",
                    severity="error",
                    suggested_fix="Verify subtotal and total amount calculations"
                ))
        
        # Tax validation
        if invoice_data.tax_info:
            if invoice_data.tax_info.tax_rate and (invoice_data.tax_info.tax_rate < 0 or invoice_data.tax_info.tax_rate > 1):
                issues.append(ValidationIssue(
                    field="tax_info.tax_rate",
                    issue_type="invalid_value",
                    message="Tax rate should be between 0 and 1 (0% to 100%)",
                    severity="warning",
                    suggested_fix="Convert tax rate to decimal format"
                ))
            
            if invoice_data.tax_info and invoice_data.tax_info.tax_amount and invoice_data.tax_info.tax_amount < 0:
                issues.append(ValidationIssue(
                    field="tax_info.tax_amount",
                    issue_type="invalid_value",
                    message="Tax amount cannot be negative",
                    severity="error"
                ))
        
        return issues
    
    def _validate_line_items(self, invoice_data: InvoiceData) -> List[ValidationIssue]:
        """Validate line items."""
        issues = []
        
        if not invoice_data.line_items:
            issues.append(ValidationIssue(
                field="line_items",
                issue_type="missing_data",
                message="No line items found",
                severity="warning",
                suggested_fix="Extract individual line items from document"
            ))
            return issues
        
        total_calculated = Decimal('0')
        
        for i, item in enumerate(invoice_data.line_items):
            item_prefix = f"line_items[{i}]"
            
            # Description validation
            if not item.description or len(item.description.strip()) < 2:
                issues.append(ValidationIssue(
                    field=f"{item_prefix}.description",
                    issue_type="missing_required",
                    message=f"Line item {i+1} description is missing or too short",
                    severity="warning"
                ))
            
            # Quantity validation
            if item.quantity is not None and item.quantity <= 0:
                issues.append(ValidationIssue(
                    field=f"{item_prefix}.quantity",
                    issue_type="invalid_value",
                    message=f"Line item {i+1} quantity must be greater than zero",
                    severity="warning"
                ))
            
            # Price validation
            if item.unit_price is not None and item.unit_price < 0:
                issues.append(ValidationIssue(
                    field=f"{item_prefix}.unit_price",
                    issue_type="invalid_value",
                    message=f"Line item {i+1} unit price cannot be negative",
                    severity="warning"
                ))
            
            if item.total_price is not None and item.total_price < 0:
                issues.append(ValidationIssue(
                    field=f"{item_prefix}.total_price",
                    issue_type="invalid_value",
                    message=f"Line item {i+1} total price cannot be negative",
                    severity="warning"
                ))
            
            # Cross-validation: quantity * unit_price = total_price
            if all([item.quantity, item.unit_price, item.total_price]):
                calculated_total = item.quantity * item.unit_price
                if abs(calculated_total - item.total_price) > Decimal('0.01'):
                    issues.append(ValidationIssue(
                        field=f"{item_prefix}.total_price",
                        issue_type="calculation_error",
                        message=f"Line item {i+1} total price doesn't match quantity × unit price",
                        severity="warning",
                        suggested_fix="Verify line item calculations"
                    ))
            
            # Add to total calculation
            if item.total_price:
                total_calculated += item.total_price
        
        # Validate line items total vs invoice total
        if invoice_data.total_amount and total_calculated > 0:
            # Allow for small rounding differences and tax
            difference = abs(invoice_data.total_amount - total_calculated)
            if difference > invoice_data.total_amount * Decimal('0.1'):  # 10% tolerance
                issues.append(ValidationIssue(
                    field="line_items_total",
                    issue_type="calculation_error",
                    message="Sum of line items doesn't match invoice total (considering tax)",
                    severity="info",
                    suggested_fix="Verify line item totals and tax calculations"
                ))
        
        return issues
    
    def _validate_dates(self, invoice_data: InvoiceData) -> List[ValidationIssue]:
        """Validate date fields."""
        issues = []
        
        # Invoice date validation
        if not invoice_data.invoice_date:
            issues.append(ValidationIssue(
                field="invoice_date",
                issue_type="missing_required",
                message="Invoice date is missing",
                severity="error",
                suggested_fix="Extract invoice date from document"
            ))
        elif invoice_data.invoice_date > date.today():
            issues.append(ValidationIssue(
                field="invoice_date",
                issue_type="invalid_value",
                message="Invoice date is in the future",
                severity="warning",
                suggested_fix="Verify invoice date accuracy"
            ))
        
        # Due date validation
        if invoice_data.payment_info and invoice_data.payment_info.due_date:
            if invoice_data.invoice_date and invoice_data.payment_info.due_date < invoice_data.invoice_date:
                issues.append(ValidationIssue(
                    field="payment_info.due_date",
                    issue_type="logical_error",
                    message="Due date cannot be before invoice date",
                    severity="error",
                    suggested_fix="Verify due date accuracy"
                ))
        
        return issues
    
    def _validate_cross_fields(self, invoice_data: InvoiceData) -> List[ValidationIssue]:
        """Validate relationships between fields."""
        issues = []
        
        # If we have tax info and subtotal, validate total calculation
        if all([invoice_data.subtotal, invoice_data.tax_info, invoice_data.tax_info and invoice_data.tax_info.tax_amount, invoice_data.total_amount]):
            expected_total = invoice_data.subtotal + invoice_data.tax_info.tax_amount
            if abs(expected_total - invoice_data.total_amount) > Decimal('0.01'):
                issues.append(ValidationIssue(
                    field="total_amount",
                    issue_type="calculation_error",
                    message="Total amount doesn't match subtotal + tax",
                    severity="warning",
                    suggested_fix="Verify total amount calculation"
                ))
        
        return issues
    
    def _calculate_validation_confidence(self, validation_issues: List[ValidationIssue], invoice_data: InvoiceData) -> float:
        """Calculate validation confidence score."""
        if not validation_issues:
            return 1.0
        
        # Count issues by severity
        error_count = len([issue for issue in validation_issues if issue.severity == "error"])
        warning_count = len([issue for issue in validation_issues if issue.severity == "warning"])
        info_count = len([issue for issue in validation_issues if issue.severity == "info"])
        
        # Calculate penalty based on issue severity
        error_penalty = error_count * 0.3
        warning_penalty = warning_count * 0.1
        info_penalty = info_count * 0.05
        
        total_penalty = error_penalty + warning_penalty + info_penalty
        
        # Base confidence starts at 1.0 and decreases with issues
        confidence = max(0.0, 1.0 - total_penalty)
        
        return confidence

    async def _validate_with_llm(self, invoice_data: InvoiceData) -> List[ValidationIssue]:
        """
        Use LLM for advanced validation and anomaly detection.

        Args:
            invoice_data: Invoice data to validate

        Returns:
            List[ValidationIssue]: List of validation issues found by LLM
        """
        try:
            # Prepare combined data for LLM
            combined_data = {
                "invoice_number": invoice_data.invoice_number,
                "invoice_date": str(invoice_data.invoice_date) if invoice_data.invoice_date else None,
                "vendor": {
                    "name": invoice_data.vendor.name if invoice_data.vendor else None,
                    "address": invoice_data.vendor.address if invoice_data.vendor else None,
                    "tax_id": invoice_data.vendor.tax_id if invoice_data.vendor else None,
                } if invoice_data.vendor else None,
                "currency": invoice_data.currency,
                "subtotal": float(invoice_data.subtotal) if invoice_data.subtotal else None,
                "tax_info": {
                    "tax_rate": float(invoice_data.tax_info.tax_rate) if invoice_data.tax_info and invoice_data.tax_info.tax_rate else None,
                    "tax_amount": float(invoice_data.tax_info.tax_amount) if invoice_data.tax_info and invoice_data.tax_info.tax_amount else None,
                } if invoice_data.tax_info else None,
                "total_amount": float(invoice_data.total_amount) if invoice_data.total_amount else None,
                "line_items": [
                    {
                        "description": item.description,
                        "quantity": float(item.quantity) if item.quantity else None,
                        "unit_price": float(item.unit_price) if item.unit_price else None,
                        "total_price": float(item.total_price) if item.total_price else None,
                    }
                    for item in invoice_data.line_items
                ],
                "overall_confidence": getattr(invoice_data, 'overall_confidence', None),
                "ocr_quality": getattr(invoice_data, 'ocr_quality', None),
            }

            # Format prompt with data
            prompt = VALIDATION_PROMPT.format(
                combined_data=json.dumps(combined_data, indent=2, ensure_ascii=False)
            )

            # Get LLM validation
            validation_result = await self.llm_service.generate_text(prompt)

            # Parse LLM response
            try:
                # Clean up the response
                validation_result = validation_result.strip()
                if validation_result.startswith('```json'):
                    validation_result = validation_result[7:]
                if validation_result.endswith('```'):
                    validation_result = validation_result[:-3]
                validation_result = validation_result.strip()

                # Find JSON content between braces
                start_idx = validation_result.find('{')
                end_idx = validation_result.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    validation_result = validation_result[start_idx:end_idx+1]

                validation_data = json.loads(validation_result)

                validation_issues = []

                # Convert validation errors to ValidationIssue objects
                if "validation_errors" in validation_data:
                    for error in validation_data["validation_errors"]:
                        validation_issues.append(
                            ValidationIssue(
                                field=error.get("field", "unknown"),
                                issue_type=error.get("error_type", "validation_error"),
                                message=error.get("message", "Validation error detected"),
                                severity=error.get("severity", "medium")
                            )
                        )

                # Convert anomalies to ValidationIssue objects
                if "anomalies" in validation_data:
                    for anomaly in validation_data["anomalies"]:
                        validation_issues.append(
                            ValidationIssue(
                                field="anomaly_detection",
                                issue_type=anomaly.get("type", "anomaly"),
                                message=f"{anomaly.get('message', 'Anomaly detected')} - {anomaly.get('recommendation', '')}",
                                severity=anomaly.get("severity", "medium")
                            )
                        )

                return validation_issues

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM validation response: {e}")
                return [
                    ValidationIssue(
                        field="llm_validation",
                        issue_type="parsing_error",
                        message="Failed to parse LLM validation response",
                        severity="warning"
                    )
                ]

        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return [
                ValidationIssue(
                    field="llm_validation",
                    issue_type="service_error",
                    message=f"LLM validation service error: {str(e)}",
                    severity="info"
                )
            ]
