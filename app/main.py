"""
FastAPI main application module.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.response import (
    InvoiceAnalysisResponse,
    HealthResponse,
    ErrorResponse
)
from app.services.ocr_service import OCRService
from app.services.llm_service import LLMService
from app.services.validator import ValidationService
from app.services.categorizer import CategorizationService
from app.utils.helpers import validate_file_type, validate_file_size


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting Invoice Analyzer API...")
    yield
    logger.info("Shutting down Invoice Analyzer API...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection for services
def get_ocr_service() -> OCRService:
    """Get OCR service instance."""
    return OCRService()


def get_llm_service() -> LLMService:
    """Get LLM service instance."""
    return LLMService()


def get_validation_service() -> ValidationService:
    """Get validation service instance."""
    return ValidationService()


def get_categorization_service() -> CategorizationService:
    """Get categorization service instance."""
    return CategorizationService()


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        message="Invoice Analyzer API is running"
    )


@app.post("/analyze-invoice", response_model=InvoiceAnalysisResponse)
async def analyze_invoice(
    file: UploadFile = File(...),
    use_fallback: bool = Query(False, description="Force use AWS Textract fallback instead of GPT-4 Vision"),
    ocr_service: OCRService = Depends(get_ocr_service),
    llm_service: LLMService = Depends(get_llm_service),
    validation_service: ValidationService = Depends(get_validation_service),
    categorization_service: CategorizationService = Depends(get_categorization_service)
) -> InvoiceAnalysisResponse:
    """
    Analyze uploaded invoice image through 4-phase processing pipeline.

    Args:
        file: Uploaded invoice image file
        use_fallback: Force use AWS Textract fallback instead of GPT-4 Vision

    Returns:
        InvoiceAnalysisResponse: Complete analysis results

    Raises:
        HTTPException: If file validation fails or processing errors occur
    """
    try:
        # Validate file type
        validate_file_type(file)

        # Read file content
        file_content = await file.read()

        # Validate file size
        validate_file_size(file_content, settings.max_file_size_mb, file.filename)
        
        logger.info(f"Starting analysis for file: {file.filename}")
        
        # Phase 1: OCR Data Extraction
        logger.info("Phase 1: OCR Data Extraction")
        extracted_data = await ocr_service.extract_data(file_content, file.filename, use_fallback=use_fallback)
        
        # Phase 2: Categorization
        logger.info("Phase 2: Expense Categorization")
        categorized_data = await categorization_service.categorize_expenses(extracted_data)
        
        # Phase 3: Validation
        logger.info("Phase 3: Data Validation")
        validated_data = await validation_service.validate_data(categorized_data)

        # Update confidence level
        validated_data.update_confidence_level()

        # Check if fallback is needed (low confidence)
        if validated_data.overall_confidence < 0.6:
            logger.warning(f"Low confidence ({validated_data.overall_confidence:.2f}), trying AWS Textract fallback")
            try:
                # Retry with Textract
                fallback_data = await ocr_service.extract_data(file_content, file.filename, use_fallback=True)
                fallback_categorized = await categorization_service.categorize_expenses(fallback_data)
                fallback_validated = await validation_service.validate_data(fallback_categorized)
                fallback_validated.update_confidence_level()

                # Use fallback if it's better
                if fallback_validated.overall_confidence > validated_data.overall_confidence:
                    logger.info(f"Fallback improved confidence: {fallback_validated.overall_confidence:.2f}")
                    validated_data = fallback_validated
                    if not validated_data.processing_metadata:
                        validated_data.processing_metadata = {}
                    validated_data.processing_metadata["fallback_used"] = True
                else:
                    logger.info("Fallback didn't improve confidence, using original")

            except Exception as e:
                logger.error(f"Fallback failed: {e}")

        # Phase 4: Final Summary
        logger.info("Phase 4: Final Summary Generation")
        final_result = await llm_service.generate_summary(validated_data)

        # Check if summary recommends fallback
        if (final_result.summary and
            final_result.summary.get('processing_decision') and
            final_result.summary.get('processing_decision', {}).get('method') == 'fallback_required' and
            not use_fallback):  # Don't retry if already using fallback

            logger.warning(f"Summary recommends fallback: {final_result.summary.get('processing_decision', {}).get('reason')}")
            try:
                # Retry with Textract
                fallback_data = await ocr_service.extract_data(file_content, file.filename, use_fallback=True)
                fallback_categorized = await categorization_service.categorize_expenses(fallback_data)
                fallback_validated = await validation_service.validate_data(fallback_categorized)
                fallback_validated.update_confidence_level()

                # Generate summary for fallback data
                fallback_result = await llm_service.generate_summary(fallback_validated)

                # Use fallback if it's better (fewer validation issues or higher confidence)
                fallback_issues = len(fallback_validated.validation_issues) if fallback_validated.validation_issues else 0
                original_issues = len(validated_data.validation_issues) if validated_data.validation_issues else 0

                if (fallback_issues < original_issues or
                    fallback_validated.overall_confidence > validated_data.overall_confidence):
                    logger.info(f"Fallback improved results: {fallback_issues} issues vs {original_issues}, confidence: {fallback_validated.overall_confidence:.2f}")
                    validated_data = fallback_validated
                    final_result = fallback_result
                    if not validated_data.processing_metadata:
                        validated_data.processing_metadata = {}
                    validated_data.processing_metadata["fallback_used"] = True
                    validated_data.processing_metadata["fallback_reason"] = "summary_recommendation"
                else:
                    logger.info("Fallback didn't improve results, using original")

            except Exception as e:
                logger.error(f"Summary-recommended fallback failed: {e}")

        # Add cost tracking metadata
        if hasattr(llm_service, 'cost_tracker') or hasattr(ocr_service, 'cost_tracker'):
            # Combine costs from both services
            combined_cost_breakdown = {}
            combined_models_used = []
            total_cost = 0.0

            # Add LLM service costs
            if hasattr(llm_service, 'cost_tracker'):
                llm_cost_breakdown = llm_service.cost_tracker.get_cost_breakdown()
                llm_total_cost = llm_service.cost_tracker.get_session_total()
                combined_cost_breakdown.update(llm_cost_breakdown)
                combined_models_used.extend([usage.model_dump() for usage in llm_service.cost_tracker.session_costs])
                total_cost += llm_total_cost

            # Add OCR service costs
            if hasattr(ocr_service, 'cost_tracker'):
                ocr_cost_breakdown = ocr_service.cost_tracker.get_cost_breakdown()
                ocr_total_cost = ocr_service.cost_tracker.get_session_total()
                # Merge cost breakdowns
                for provider, cost in ocr_cost_breakdown.items():
                    if provider in combined_cost_breakdown:
                        combined_cost_breakdown[provider] += cost
                    else:
                        combined_cost_breakdown[provider] = cost
                combined_models_used.extend([usage.model_dump() for usage in ocr_service.cost_tracker.session_costs])
                total_cost += ocr_total_cost

            if not validated_data.processing_metadata:
                validated_data.processing_metadata = {}

            validated_data.processing_metadata.update({
                "cost_breakdown": combined_cost_breakdown,
                "total_cost_usd": total_cost,
                "models_used": combined_models_used
            })
        
        logger.info(f"Analysis completed successfully for file: {file.filename}")
        
        return final_result
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during invoice analysis: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=True,
            message=exc.detail,
            status_code=exc.status_code
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle general exceptions with structured error response."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=True,
            message="Internal server error",
            status_code=500
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
