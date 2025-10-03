"""
Helper functions for invoice processing.
"""

import mimetypes
from typing import List, Optional
from fastapi import UploadFile, HTTPException
from PIL import Image
import io

from app.config import settings


# Supported image formats
SUPPORTED_IMAGE_TYPES = {
    'image/jpeg',
    'image/jpg', 
    'image/png',
    'image/gif',
    'image/bmp',
    'image/tiff',
    'image/webp'
}

# Supported document formats
SUPPORTED_DOCUMENT_TYPES = {
    'application/pdf'
}

# All supported file types
SUPPORTED_FILE_TYPES = SUPPORTED_IMAGE_TYPES | SUPPORTED_DOCUMENT_TYPES


def validate_file_type(file: UploadFile) -> None:
    """
    Validate uploaded file type.
    
    Args:
        file: Uploaded file to validate
        
    Raises:
        HTTPException: If file type is not supported
    """
    # Get content type (either from file or guess from filename)
    content_type = file.content_type
    if not content_type:
        # Try to guess content type from filename
        content_type, _ = mimetypes.guess_type(file.filename or '')
        if not content_type:
            raise HTTPException(
                status_code=400,
                detail="Could not determine file type. Please upload a supported image or PDF file."
            )

    if content_type not in SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. "
                   f"Supported types: {', '.join(sorted(SUPPORTED_FILE_TYPES))}"
        )


def validate_file_size(file_content: bytes, max_size_mb: int, filename: Optional[str] = None) -> None:
    """
    Validate file size from content.

    Args:
        file_content: File content as bytes
        max_size_mb: Maximum file size in MB
        filename: Optional filename for error message

    Raises:
        HTTPException: If file is too large
    """
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size_mb:.1f}MB. Maximum size: {max_size_mb}MB"
        )


def validate_image_content(file_content: bytes) -> None:
    """
    Validate image content and format.
    
    Args:
        file_content: Image file content as bytes
        
    Raises:
        HTTPException: If image is invalid or corrupted
    """
    try:
        # Try to open and validate the image
        image = Image.open(io.BytesIO(file_content))
        image.verify()  # Verify image integrity
        
        # Check image dimensions (reasonable limits)
        if hasattr(image, 'size'):
            width, height = image.size
            if width < 50 or height < 50:
                raise HTTPException(
                    status_code=400,
                    detail="Image too small. Minimum size: 50x50 pixels"
                )
            if width > 10000 or height > 10000:
                raise HTTPException(
                    status_code=400,
                    detail="Image too large. Maximum size: 10000x10000 pixels"
                )
                
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file: {str(e)}"
        )


def optimize_image_for_ocr(file_content: bytes, max_size: tuple = (2048, 2048)) -> bytes:
    """
    Optimize image for OCR processing.
    
    Args:
        file_content: Original image content
        max_size: Maximum dimensions (width, height)
        
    Returns:
        bytes: Optimized image content
    """
    try:
        # Open image
        image = Image.open(io.BytesIO(file_content))
        
        # Convert to RGB if necessary (for JPEG compatibility)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save optimized image
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=85, optimize=True)
        return output.getvalue()
        
    except Exception as e:
        # Return original if optimization fails
        return file_content


def extract_filename_info(filename: Optional[str]) -> dict:
    """
    Extract information from filename.
    
    Args:
        filename: Original filename
        
    Returns:
        dict: Extracted filename information
    """
    if not filename:
        return {
            'name': 'unknown',
            'extension': '',
            'is_image': False,
            'is_pdf': False
        }
    
    # Split filename and extension
    name_parts = filename.rsplit('.', 1)
    name = name_parts[0] if len(name_parts) > 1 else filename
    extension = name_parts[1].lower() if len(name_parts) > 1 else ''
    
    # Determine file type
    is_image = extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
    is_pdf = extension == 'pdf'
    
    return {
        'name': name,
        'extension': extension,
        'is_image': is_image,
        'is_pdf': is_pdf,
        'full_name': filename
    }


def sanitize_text(text: Optional[str]) -> Optional[str]:
    """
    Sanitize extracted text for processing.
    
    Args:
        text: Raw text to sanitize
        
    Returns:
        Optional[str]: Sanitized text or None
    """
    if not text:
        return None
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove control characters but keep newlines and tabs
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # Return None if empty after sanitization
    return sanitized.strip() if sanitized.strip() else None


def format_currency(amount: Optional[float], currency: Optional[str] = None) -> str:
    """
    Format currency amount for display.
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        str: Formatted currency string
    """
    if amount is None:
        return "N/A"
    
    # Format with 2 decimal places
    formatted_amount = f"{amount:,.2f}"
    
    # Add currency symbol if provided
    if currency:
        currency_symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'TRY': '₺'
        }
        symbol = currency_symbols.get(currency, currency)
        return f"{symbol}{formatted_amount}"
    
    return formatted_amount


def calculate_confidence_level(confidence_score: float) -> str:
    """
    Convert confidence score to level description.
    
    Args:
        confidence_score: Confidence score (0.0 to 1.0)
        
    Returns:
        str: Confidence level description
    """
    if confidence_score >= 0.9:
        return "Very High"
    elif confidence_score >= 0.8:
        return "High"
    elif confidence_score >= 0.7:
        return "Medium"
    elif confidence_score >= 0.5:
        return "Low"
    else:
        return "Very Low"


def generate_processing_id() -> str:
    """
    Generate unique processing ID for tracking.
    
    Returns:
        str: Unique processing ID
    """
    import uuid
    import time
    
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    
    return f"inv_{timestamp}_{unique_id}"


def estimate_processing_time(file_size_bytes: int, file_type: str) -> int:
    """
    Estimate processing time based on file characteristics.
    
    Args:
        file_size_bytes: File size in bytes
        file_type: File MIME type
        
    Returns:
        int: Estimated processing time in seconds
    """
    # Base processing time
    base_time = 5  # seconds
    
    # Size factor (larger files take longer)
    size_mb = file_size_bytes / (1024 * 1024)
    size_factor = min(size_mb * 0.5, 10)  # Cap at 10 seconds for size
    
    # Type factor (PDFs typically take longer)
    type_factor = 3 if file_type == 'application/pdf' else 1
    
    estimated_time = int(base_time + size_factor + type_factor)
    
    return min(estimated_time, 60)  # Cap at 60 seconds


def validate_processing_result(result: dict) -> List[str]:
    """
    Validate processing result for completeness.
    
    Args:
        result: Processing result dictionary
        
    Returns:
        List[str]: List of validation warnings
    """
    warnings = []
    
    # Check for critical missing data
    if not result.get('vendor', {}).get('name'):
        warnings.append("Vendor name is missing")
    
    if not result.get('invoice_number'):
        warnings.append("Invoice number is missing")
    
    if not result.get('total_amount'):
        warnings.append("Total amount is missing")
    
    if not result.get('invoice_date'):
        warnings.append("Invoice date is missing")
    
    # Check line items
    line_items = result.get('line_items', [])
    if not line_items:
        warnings.append("No line items found")
    else:
        for i, item in enumerate(line_items):
            if not item.get('description'):
                warnings.append(f"Line item {i+1} missing description")
    
    return warnings
