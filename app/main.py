"""
Enterprise-level FastAPI main application module.

This module provides the HTTP API layer for the invoice analysis system,
following enterprise standards with proper separation of concerns,
structured logging, and comprehensive error handling.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.constants import LogContext
from app.models.response import (
    InvoiceAnalysisResponse,
    HealthResponse,
    ErrorResponse
)
from app.services.ocr_service import OCRService
from app.services.llm_service import LLMService
from app.services.validator import ValidationService
from app.services.categorizer import CategorizationService
from app.services.processing_orchestrator import ProcessingOrchestrator
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


def get_processing_orchestrator(
    ocr_service: OCRService = Depends(get_ocr_service),
    llm_service: LLMService = Depends(get_llm_service),
    validation_service: ValidationService = Depends(get_validation_service),
    categorization_service: CategorizationService = Depends(get_categorization_service)
) -> ProcessingOrchestrator:
    """
    Get processing orchestrator instance with injected dependencies.

    Args:
        ocr_service: OCR service instance
        llm_service: LLM service instance
        validation_service: Validation service instance
        categorization_service: Categorization service instance

    Returns:
        ProcessingOrchestrator: Configured orchestrator instance
    """
    return ProcessingOrchestrator(
        ocr_service=ocr_service,
        llm_service=llm_service,
        validation_service=validation_service,
        categorization_service=categorization_service,
        logger=logger
    )


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
    orchestrator: ProcessingOrchestrator = Depends(get_processing_orchestrator)
) -> InvoiceAnalysisResponse:
    """
    Analyze uploaded invoice image through enterprise-level processing pipeline.

    This endpoint provides a clean HTTP interface to the invoice processing system,
    delegating all business logic to the ProcessingOrchestrator for proper
    separation of concerns.

    Args:
        file: Uploaded invoice image file
        use_fallback: Force use AWS Textract fallback instead of GPT-4 Vision
        orchestrator: Processing orchestrator with injected dependencies

    Returns:
        InvoiceAnalysisResponse: Complete analysis results with cost tracking

    Raises:
        HTTPException: If file validation fails or processing errors occur
    """
    request_id = str(uuid.uuid4())

    try:
        # Input validation
        _validate_upload_file(file)

        # Read and validate file content
        file_content = await file.read()
        validate_file_size(file_content, settings.max_file_size_mb, file.filename)

        # Log request start with structured context
        context = {
            LogContext.REQUEST_ID: request_id,
            LogContext.FILENAME: file.filename or "unknown"
        }
        logger.info("Starting invoice analysis request", extra=context)

        # Delegate to orchestrator for business logic
        result = await orchestrator.process_invoice(
            file_content=file_content,
            filename=file.filename or "unknown",
            use_fallback=use_fallback,
            request_id=request_id
        )

        logger.info("Invoice analysis request completed successfully", extra=context)
        return result

    except ValueError as e:
        logger.error(f"Validation error for request {request_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Processing error for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during invoice analysis: {str(e)}"
        )


def _validate_upload_file(file: UploadFile) -> None:
    """
    Validate uploaded file meets requirements.

    Args:
        file: Uploaded file to validate

    Raises:
        ValueError: If file validation fails
    """
    if not file.filename:
        raise ValueError("Filename is required")

    validate_file_type(file)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions with structured error response.

    Args:
        _: FastAPI request object (unused)
        exc: HTTP exception that occurred

    Returns:
        JSONResponse: Structured error response
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=True,
            message=exc.detail,
            status_code=exc.status_code
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """
    Handle general exceptions with structured error response.

    Args:
        _: FastAPI request object (unused)
        exc: Exception that occurred

    Returns:
        JSONResponse: Structured error response
    """
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
