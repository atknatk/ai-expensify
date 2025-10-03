"""
Enterprise-level processing orchestrator for invoice analysis pipeline.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional

from app.constants import (
    CONFIDENCE_THRESHOLD_LOW,
    ProcessingPhase,
    FallbackReason,
    ProcessingDecisionMethod,
    LogContext
)
from app.models.fallback import (
    FallbackDecision,
    FallbackResult,
    FallbackOutcome,
    create_confidence_fallback_decision,
    create_summary_fallback_decision
)
from app.models.response import InvoiceAnalysisResponse
from app.models.invoice import ProcessedInvoice, ConfidenceLevel
from app.services.ocr_service import OCRService
from app.services.llm_service import LLMService
from app.services.validator import ValidationService
from app.services.categorizer import CategorizationService


class ProcessingOrchestrator:
    """
    Enterprise-level orchestrator for multi-phase invoice processing pipeline.
    
    Manages the complete processing workflow including fallback strategies,
    cost tracking, and structured logging with enterprise standards.
    """
    
    def __init__(
        self,
        ocr_service: OCRService,
        llm_service: LLMService,
        validation_service: ValidationService,
        categorization_service: CategorizationService,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the processing orchestrator.
        
        Args:
            ocr_service: OCR service for data extraction
            llm_service: LLM service for summary generation
            validation_service: Validation service for data validation
            categorization_service: Categorization service for expense categorization
            logger: Optional logger instance
        """
        self.ocr_service = ocr_service
        self.llm_service = llm_service
        self.validation_service = validation_service
        self.categorization_service = categorization_service
        self.logger = logger or logging.getLogger(__name__)
        
    async def process_invoice(
        self,
        file_content: bytes,
        filename: str,
        use_fallback: bool = False,
        request_id: Optional[str] = None
    ) -> InvoiceAnalysisResponse:
        """
        Execute the complete invoice processing pipeline.
        
        Args:
            file_content: Raw file content bytes
            filename: Original filename
            use_fallback: Whether to force fallback processing
            request_id: Optional request ID for tracking
            
        Returns:
            InvoiceAnalysisResponse: Complete analysis results
            
        Raises:
            Exception: If processing fails at any stage
        """
        if not request_id:
            request_id = str(uuid.uuid4())
            
        context = {
            LogContext.REQUEST_ID: request_id,
            LogContext.FILENAME: filename
        }
        
        self.logger.info("Starting invoice processing pipeline", extra=context)
        pipeline_start_time = time.time()
        
        try:
            # Phase 1: OCR Data Extraction
            extracted_data = await self._execute_ocr_phase(
                file_content, filename, use_fallback, context
            )
            
            # Phase 2: Expense Categorization
            categorized_data = await self._execute_categorization_phase(
                extracted_data, context
            )
            
            # Phase 3: Data Validation
            validated_data = await self._execute_validation_phase(
                categorized_data, context
            )
            
            # Check for confidence-based fallback
            fallback_result = None
            if not use_fallback and validated_data.overall_confidence < CONFIDENCE_THRESHOLD_LOW:
                fallback_result = await self._execute_confidence_fallback(
                    file_content, filename, validated_data, context
                )
                if fallback_result.was_successful and fallback_result.validated_data:
                    validated_data = fallback_result.validated_data

            # Phase 4: Summary Generation
            if validated_data is None:
                raise ValueError("No validated data available for summary generation")
            final_result = await self._execute_summary_phase(validated_data, context)

            # Check for summary-based fallback
            if not use_fallback and not fallback_result:
                summary_fallback_result = await self._check_summary_fallback(
                    file_content, filename, validated_data, final_result, context
                )
                if summary_fallback_result and summary_fallback_result.was_successful:
                    if summary_fallback_result.validated_data:
                        validated_data = summary_fallback_result.validated_data
                    if summary_fallback_result.final_result:
                        final_result = summary_fallback_result.final_result

            # Add cost tracking metadata
            if validated_data is None:
                raise ValueError("No validated data available for cost tracking")
            self._add_cost_tracking_metadata(validated_data)
            
            pipeline_time = int((time.time() - pipeline_start_time) * 1000)
            context[LogContext.PROCESSING_TIME] = str(pipeline_time)
            
            self.logger.info("Invoice processing pipeline completed successfully", extra=context)
            
            return final_result
            
        except Exception as e:
            pipeline_time = int((time.time() - pipeline_start_time) * 1000)
            context[LogContext.PROCESSING_TIME] = str(pipeline_time)
            self.logger.error(f"Invoice processing pipeline failed: {str(e)}", extra=context)
            raise
    
    async def _execute_ocr_phase(
        self,
        file_content: bytes,
        filename: str,
        use_fallback: bool,
        context: Dict[str, Any]
    ) -> ProcessedInvoice:
        """Execute OCR data extraction phase."""
        context[LogContext.PHASE] = ProcessingPhase.OCR_EXTRACTION
        self.logger.info("Executing OCR phase", extra=context)
        
        phase_start_time = time.time()
        invoice_data = await self.ocr_service.extract_data(
            file_content, filename, use_fallback=use_fallback
        )

        # Create ProcessedInvoice wrapper
        extracted_data = ProcessedInvoice(
            invoice_data=invoice_data,
            processing_phases=[],
            overall_confidence=0.0,
            confidence_level=ConfidenceLevel.LOW,
            validation_issues=[],
            is_valid=True
        )

        phase_time = int((time.time() - phase_start_time) * 1000)

        context[LogContext.PROCESSING_TIME] = str(phase_time)
        self.logger.info("OCR phase completed", extra=context)

        return extracted_data
    
    async def _execute_categorization_phase(
        self,
        extracted_data: ProcessedInvoice,
        context: Dict[str, Any]
    ) -> ProcessedInvoice:
        """Execute expense categorization phase."""
        context[LogContext.PHASE] = ProcessingPhase.CATEGORIZATION
        self.logger.info("Executing categorization phase", extra=context)
        
        phase_start_time = time.time()
        categorized_data = await self.categorization_service.categorize_expenses(extracted_data.invoice_data)
        # Update the ProcessedInvoice with categorized data
        extracted_data.invoice_data = categorized_data
        phase_time = int((time.time() - phase_start_time) * 1000)
        
        context[LogContext.PROCESSING_TIME] = str(phase_time)
        self.logger.info("Categorization phase completed", extra=context)

        return extracted_data
    
    async def _execute_validation_phase(
        self,
        categorized_data: ProcessedInvoice,
        context: Dict[str, Any]
    ) -> ProcessedInvoice:
        """Execute data validation phase."""
        context[LogContext.PHASE] = ProcessingPhase.VALIDATION
        self.logger.info("Executing validation phase", extra=context)
        
        phase_start_time = time.time()
        validated_data = await self.validation_service.validate_data(categorized_data.invoice_data)
        validated_data.update_confidence_level()
        phase_time = int((time.time() - phase_start_time) * 1000)
        
        context[LogContext.PROCESSING_TIME] = str(phase_time)
        context[LogContext.CONFIDENCE] = str(validated_data.overall_confidence)
        self.logger.info("Validation phase completed", extra=context)
        
        return validated_data
    
    async def _execute_summary_phase(
        self,
        validated_data: ProcessedInvoice,
        context: Dict[str, Any]
    ) -> InvoiceAnalysisResponse:
        """Execute summary generation phase."""
        context[LogContext.PHASE] = ProcessingPhase.SUMMARY_GENERATION
        self.logger.info("Executing summary phase", extra=context)
        
        phase_start_time = time.time()
        final_result = await self.llm_service.generate_summary(validated_data)
        phase_time = int((time.time() - phase_start_time) * 1000)
        
        context[LogContext.PROCESSING_TIME] = str(phase_time)
        self.logger.info("Summary phase completed", extra=context)

        return final_result

    async def _execute_confidence_fallback(
        self,
        file_content: bytes,
        filename: str,
        original_data: ProcessedInvoice,
        context: Dict[str, Any]
    ) -> FallbackResult:
        """
        Execute confidence-based fallback processing.

        Args:
            file_content: Raw file content bytes
            filename: Original filename
            original_data: Original processing results
            context: Logging context

        Returns:
            FallbackResult: Result of fallback processing
        """
        decision = create_confidence_fallback_decision(
            original_data.overall_confidence,
            CONFIDENCE_THRESHOLD_LOW
        )

        context[LogContext.FALLBACK_REASON] = FallbackReason.LOW_CONFIDENCE
        self.logger.warning(f"Executing confidence-based fallback: {decision}", extra=context)

        return await self._execute_fallback_pipeline(
            file_content, filename, original_data, decision, context
        )

    async def _check_summary_fallback(
        self,
        file_content: bytes,
        filename: str,
        validated_data: ProcessedInvoice,
        final_result: InvoiceAnalysisResponse,
        context: Dict[str, Any]
    ) -> Optional[FallbackResult]:
        """
        Check if summary recommends fallback and execute if needed.

        Args:
            file_content: Raw file content bytes
            filename: Original filename
            validated_data: Validated processing results
            final_result: Final analysis results with summary
            context: Logging context

        Returns:
            Optional[FallbackResult]: Result of fallback processing if executed
        """
        if not self._should_execute_summary_fallback(final_result):
            return None

        processing_decision = final_result.summary.get('processing_decision', {})
        decision = create_summary_fallback_decision(
            processing_decision,
            len(validated_data.validation_issues) if validated_data.validation_issues else 0
        )

        context[LogContext.FALLBACK_REASON] = FallbackReason.SUMMARY_RECOMMENDATION
        self.logger.warning(f"Executing summary-based fallback: {decision}", extra=context)

        fallback_result = await self._execute_fallback_pipeline(
            file_content, filename, validated_data, decision, context
        )

        if fallback_result.was_successful and fallback_result.validated_data:
            # Generate new summary for fallback data
            fallback_summary = await self.llm_service.generate_summary(fallback_result.validated_data)
            fallback_result.final_result = fallback_summary

        return fallback_result

    def _should_execute_summary_fallback(self, final_result: InvoiceAnalysisResponse) -> bool:
        """Check if summary recommends fallback processing."""
        if not final_result.summary:
            return False

        processing_decision = final_result.summary.get('processing_decision')
        if not processing_decision:
            return False

        return processing_decision.get('method') == ProcessingDecisionMethod.FALLBACK_REQUIRED

    async def _execute_fallback_pipeline(
        self,
        file_content: bytes,
        filename: str,
        original_data: ProcessedInvoice,
        decision: FallbackDecision,
        context: Dict[str, Any]
    ) -> FallbackResult:
        """
        Execute the complete fallback processing pipeline.

        Args:
            file_content: Raw file content bytes
            filename: Original filename
            original_data: Original processing results
            decision: Fallback decision details
            context: Logging context

        Returns:
            FallbackResult: Result of fallback processing
        """
        fallback_start_time = time.time()

        try:
            # Execute fallback OCR with Textract
            fallback_invoice_data = await self.ocr_service.extract_data(
                file_content, filename, use_fallback=True
            )

            # Create ProcessedInvoice wrapper
            fallback_data = ProcessedInvoice(
                invoice_data=fallback_invoice_data,
                processing_phases=[],
                overall_confidence=0.0,
                confidence_level=ConfidenceLevel.LOW,
                validation_issues=[],
                is_valid=True
            )

            # Process through categorization and validation
            fallback_categorized_data = await self.categorization_service.categorize_expenses(fallback_data.invoice_data)
            fallback_data.invoice_data = fallback_categorized_data
            fallback_validated = await self.validation_service.validate_data(fallback_data.invoice_data)
            fallback_validated.update_confidence_level()

            # Determine if fallback improved results
            outcome = self._evaluate_fallback_improvement(original_data, fallback_validated)

            execution_time = int((time.time() - fallback_start_time) * 1000)

            result = FallbackResult(
                decision=decision,
                outcome=outcome,
                original_confidence=original_data.overall_confidence,
                fallback_confidence=fallback_validated.overall_confidence,
                original_issues_count=len(original_data.validation_issues) if original_data.validation_issues else 0,
                fallback_issues_count=len(fallback_validated.validation_issues) if fallback_validated.validation_issues else 0,
                execution_time_ms=execution_time
            )

            if outcome == FallbackOutcome.IMPROVED:
                # Add fallback metadata
                if not fallback_validated.processing_metadata:
                    fallback_validated.processing_metadata = {}
                fallback_validated.processing_metadata.update({
                    "fallback_used": True,
                    "fallback_reason": decision.trigger.value,
                    "fallback_trigger_details": decision.reason
                })
                result.validated_data = fallback_validated

                self.logger.info(f"Fallback improved results: {result}", extra=context)
            else:
                self.logger.info(f"Fallback did not improve results: {result}", extra=context)

            return result

        except Exception as e:
            execution_time = int((time.time() - fallback_start_time) * 1000)
            self.logger.error(f"Fallback processing failed: {str(e)}", extra=context)

            return FallbackResult(
                decision=decision,
                outcome=FallbackOutcome.FAILED,
                original_confidence=original_data.overall_confidence,
                execution_time_ms=execution_time,
                error_message=str(e)
            )

    def _evaluate_fallback_improvement(
        self,
        original_data: ProcessedInvoice,
        fallback_data: ProcessedInvoice
    ) -> FallbackOutcome:
        """
        Evaluate whether fallback processing improved results.

        Args:
            original_data: Original processing results
            fallback_data: Fallback processing results

        Returns:
            FallbackOutcome: Whether fallback improved results
        """
        original_issues = len(original_data.validation_issues) if original_data.validation_issues else 0
        fallback_issues = len(fallback_data.validation_issues) if fallback_data.validation_issues else 0

        # Fallback is considered improved if:
        # 1. Fewer validation issues, OR
        # 2. Higher confidence score
        if (fallback_issues < original_issues or
            fallback_data.overall_confidence > original_data.overall_confidence):
            return FallbackOutcome.IMPROVED
        else:
            return FallbackOutcome.NO_IMPROVEMENT

    def _add_cost_tracking_metadata(self, validated_data: ProcessedInvoice) -> None:
        """
        Add cost tracking metadata to processing results.

        Args:
            validated_data: Validated processing results to update
        """
        if not (hasattr(self.llm_service, 'cost_tracker') or hasattr(self.ocr_service, 'cost_tracker')):
            return

        combined_cost_breakdown = {}
        combined_models_used = []
        total_cost = 0.0

        # Add LLM service costs
        if hasattr(self.llm_service, 'cost_tracker'):
            llm_cost_breakdown = self.llm_service.cost_tracker.get_cost_breakdown()
            llm_total_cost = self.llm_service.cost_tracker.get_session_total()
            combined_cost_breakdown.update(llm_cost_breakdown)
            combined_models_used.extend([
                usage.model_dump() for usage in self.llm_service.cost_tracker.session_costs
            ])
            total_cost += llm_total_cost

        # Add OCR service costs
        if hasattr(self.ocr_service, 'cost_tracker'):
            ocr_cost_breakdown = self.ocr_service.cost_tracker.get_cost_breakdown()
            ocr_total_cost = self.ocr_service.cost_tracker.get_session_total()

            # Merge cost breakdowns
            for provider, cost in ocr_cost_breakdown.items():
                if provider in combined_cost_breakdown:
                    combined_cost_breakdown[provider] += cost
                else:
                    combined_cost_breakdown[provider] = cost

            combined_models_used.extend([
                usage.model_dump() for usage in self.ocr_service.cost_tracker.session_costs
            ])
            total_cost += ocr_total_cost

        # Update metadata
        if not validated_data.processing_metadata:
            validated_data.processing_metadata = {}

        validated_data.processing_metadata.update({
            "cost_breakdown": combined_cost_breakdown,
            "total_cost_usd": total_cost,
            "models_used": combined_models_used
        })
