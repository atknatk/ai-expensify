"""
OCR service for extracting text and data from invoice images.
"""

import base64
import json
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from openai import OpenAI

from app.config import settings
from app.models.invoice import InvoiceData, ProcessingPhase
from app.models.cost_tracking import CostTracker, ServiceProvider
from app.services.textract_service import TextractService
from app.prompts.extraction import EXTRACTION_PROMPT


logger = logging.getLogger(__name__)


class OCRService:
    """OCR service using OpenAI GPT-4 Vision with Textract fallback."""
    
    def __init__(self):
        """Initialize OCR service."""
        if settings.openai_api_key:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.openai_client = None
        self.textract_service = TextractService()
        self.config = settings
        self.cost_tracker = CostTracker()
        
    async def extract_data(self, file_content: bytes, filename: Optional[str] = None, use_fallback: bool = False) -> InvoiceData:
        """
        Extract invoice data from image using OCR.
        
        Args:
            file_content: Image file content as bytes
            filename: Original filename
            
        Returns:
            InvoiceData: Extracted invoice data
            
        Raises:
            Exception: If both OCR methods fail
        """
        start_time = time.time()
        
        try:
            # Use fallback if explicitly requested
            if use_fallback:
                logger.info("Using AWS Textract fallback as requested")
                result = await self._extract_with_textract(file_content, filename)
                processing_time = int((time.time() - start_time) * 1000)
                result.processing_phases.append(
                    ProcessingPhase(
                        phase_name="OCR_Textract_Fallback",
                        status="success",
                        duration_ms=processing_time,
                        confidence=0.7,
                        details={"method": "textract_fallback"}
                    )
                )
                return result

            # Try GPT-4 Vision first if API key is available
            if self.openai_client:
                logger.info("Attempting OCR with GPT-4 Vision")
                result = await self._extract_with_gpt4_vision(file_content, filename)
            else:
                raise Exception("OpenAI API key not configured")
            
            processing_time = int((time.time() - start_time) * 1000)
            result.processing_phases.append(
                ProcessingPhase(
                    phase_name="OCR_GPT4_Vision",
                    status="success",
                    duration_ms=processing_time,
                    confidence=0.9,
                    details={"method": "gpt4_vision"}
                )
            )
            
            return result
            
        except Exception as gpt4_error:
            logger.warning(f"GPT-4 Vision OCR failed: {str(gpt4_error)}")
            
            try:
                # Fallback to Textract
                logger.info("Falling back to AWS Textract")
                result = await self._extract_with_textract(file_content, filename)
                
                processing_time = int((time.time() - start_time) * 1000)
                result.processing_phases.append(
                    ProcessingPhase(
                        phase_name="OCR_Textract_Fallback",
                        status="success",
                        duration_ms=processing_time,
                        confidence=0.7,
                        details={"method": "textract", "fallback_reason": str(gpt4_error)}
                    )
                )
                
                return result
                
            except Exception as textract_error:
                logger.error(f"Textract OCR also failed: {str(textract_error)}")
                
                # Both methods failed
                processing_time = int((time.time() - start_time) * 1000)
                result = InvoiceData()
                result.processing_phases.append(
                    ProcessingPhase(
                        phase_name="OCR_Failed",
                        status="error",
                        duration_ms=processing_time,
                        confidence=0.0,
                        errors=[
                            f"GPT-4 Vision error: {str(gpt4_error)}",
                            f"Textract error: {str(textract_error)}"
                        ]
                    )
                )
                
                raise Exception(f"All OCR methods failed. GPT-4: {gpt4_error}, Textract: {textract_error}")
    
    async def _extract_with_gpt4_vision(self, file_content: bytes, filename: Optional[str] = None) -> InvoiceData:
        """
        Extract data using OpenAI GPT-4 Vision.
        
        Args:
            file_content: Image file content
            filename: Original filename
            
        Returns:
            InvoiceData: Extracted data
        """
        # Encode image to base64
        base64_image = base64.b64encode(file_content).decode('utf-8')
        
        try:
            start_time = time.time()
            response = self.openai_client.chat.completions.create(
                model=self.config.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": EXTRACTION_PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            duration_ms = int((time.time() - start_time) * 1000)

            # Parse the response
            extracted_text = response.choices[0].message.content

            # Track cost for GPT-4 Vision
            self.cost_tracker.track_usage(
                provider=ServiceProvider.OPENAI,
                model_name=self.config.vision_model,
                input_text="[IMAGE_INPUT]",  # Image input, placeholder for token estimation
                output_text=extracted_text,
                duration_ms=duration_ms
            )
            
            # Convert to structured data using LLM
            structured_data = await self._parse_extracted_text(extracted_text)
            
            # Set metadata
            structured_data.extracted_text = extracted_text
            structured_data.original_filename = filename
            
            return structured_data
            
        except Exception as e:
            logger.error(f"GPT-4 Vision extraction failed: {str(e)}")
            raise
    
    async def _extract_with_textract(self, file_content: bytes, filename: Optional[str] = None) -> InvoiceData:
        """
        Extract data using AWS Textract as fallback.
        
        Args:
            file_content: Image file content
            filename: Original filename
            
        Returns:
            InvoiceData: Extracted data
        """
        try:
            # Use Textract service
            extracted_text = await self.textract_service.extract_text(file_content)
            
            # Convert to structured data using LLM
            structured_data = await self._parse_extracted_text(extracted_text)
            
            # Set metadata
            structured_data.extracted_text = extracted_text
            structured_data.original_filename = filename
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Textract extraction failed: {str(e)}")
            raise
    
    async def _parse_extracted_text(self, extracted_text: str) -> InvoiceData:
        """
        Parse extracted text into structured InvoiceData.

        Args:
            extracted_text: JSON string from GPT-4 Vision

        Returns:
            InvoiceData: Structured invoice data
        """
        try:
            # Check if extracted_text is JSON (from GPT-4 Vision) or raw text (from Textract)
            extracted_text = extracted_text.strip()

            # Try to parse as JSON first (from GPT-4 Vision)
            if extracted_text.startswith('{') and extracted_text.endswith('}'):
                try:
                    parsed_data = json.loads(extracted_text)

                    # Convert to InvoiceData format
                    currency = parsed_data.get("currency")
                    # Convert currency codes for enum compatibility
                    currency_mapping = {
                        "TL": "TRY",
                        "₹": "INR",
                        "Rs": "INR",
                        "Rupees": "INR",
                        "$": "USD",
                        "€": "EUR",
                        "£": "GBP"
                    }
                    if currency in currency_mapping:
                        currency = currency_mapping[currency]

                    invoice_data = {
                        "invoice_number": parsed_data.get("invoice_number"),
                        "invoice_date": parsed_data.get("invoice_date"),
                        "vendor": {
                            "name": parsed_data.get("vendor_name"),
                            "address": parsed_data.get("vendor_address"),
                            "phone": parsed_data.get("vendor_phone"),
                            "email": parsed_data.get("vendor_email"),
                            "tax_id": parsed_data.get("vendor_tax_number"),
                        },
                        "currency": currency,
                        "subtotal": parsed_data.get("subtotal"),
                        "tax_info": {
                            "tax_rate": float(parsed_data.get("tax_rate", 0)) / 100 if parsed_data.get("tax_rate") and float(parsed_data.get("tax_rate", 0)) > 1 else parsed_data.get("tax_rate"),
                            "tax_amount": parsed_data.get("tax_amount"),
                        } if parsed_data.get("tax_rate") or parsed_data.get("tax_amount") else None,
                        "total_amount": parsed_data.get("total_amount"),
                        "line_items": [
                            {
                                "description": item.get("description"),
                                "quantity": item.get("quantity"),
                                "unit_price": item.get("unit_price"),
                                "total_price": item.get("line_total", item.get("total_price")),
                            }
                            for item in parsed_data.get("line_items", [])
                        ]
                    }

                    return InvoiceData.model_validate(invoice_data)

                except json.JSONDecodeError:
                    logger.warning("Text looks like JSON but failed to parse, treating as raw text")

            # If not JSON or JSON parsing failed, treat as raw text and use LLM to parse
            logger.warning("Extracted text is not JSON, using LLM to parse")

            # Use cheaper model to structure the extracted text
            start_time = time.time()
            response = self.openai_client.chat.completions.create(
                model=self.config.text_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at parsing invoice text into structured JSON data.
                        Parse the following invoice text and return ONLY a JSON object with the invoice information.
                        Use null for missing values. Ensure all numeric values are properly formatted.

                        Required JSON format:
                        {
                          "invoice_number": "string or null",
                          "invoice_date": "YYYY-MM-DD or null",
                          "vendor_name": "string or null",
                          "vendor_address": "string or null",
                          "vendor_tax_number": "string or null",
                          "currency": "USD/EUR/TRY/INR/GBP or null",
                          "subtotal": number or null,
                          "tax_rate": number or null,
                          "tax_amount": number or null,
                          "total_amount": number or null,
                          "line_items": [
                            {
                              "description": "string",
                              "quantity": number,
                              "unit_price": number,
                              "line_total": number
                            }
                          ]
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Parse this invoice text:\n\n{extracted_text}"
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            duration_ms = int((time.time() - start_time) * 1000)

            # Track cost for text parsing
            self.cost_tracker.track_usage(
                provider=ServiceProvider.OPENAI,
                model_name=self.config.text_model,
                input_text=f"Parse this invoice text:\n\n{extracted_text}",
                output_text=response.choices[0].message.content,
                duration_ms=duration_ms
            )

            # Parse JSON response
            llm_response = response.choices[0].message.content.strip()
            logger.info(f"LLM Raw Response: {llm_response[:500]}...")

            # Clean up the response (remove markdown code blocks if present)
            if llm_response.startswith('```json'):
                llm_response = llm_response[7:]
            if llm_response.endswith('```'):
                llm_response = llm_response[:-3]
            llm_response = llm_response.strip()

            # Find JSON content between braces
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                llm_response = llm_response[start_idx:end_idx+1]

            logger.info(f"Cleaned JSON: {llm_response[:500]}...")
            parsed_data = json.loads(llm_response)
            logger.info(f"Parsed data keys: {list(parsed_data.keys())}")

            # Convert to InvoiceData format with proper structure
            currency = parsed_data.get("currency")
            # Convert currency codes for enum compatibility
            currency_mapping = {
                "TL": "TRY",
                "₹": "INR",
                "Rs": "INR",
                "Rupees": "INR",
                "$": "USD",
                "€": "EUR",
                "Euro": "EUR",
                "£": "GBP"
            }
            if currency in currency_mapping:
                currency = currency_mapping[currency]

            invoice_data = {
                "invoice_number": parsed_data.get("invoice_number"),
                "invoice_date": parsed_data.get("invoice_date"),
                "vendor": {
                    "name": parsed_data.get("vendor_name"),
                    "address": parsed_data.get("vendor_address"),
                    "tax_id": parsed_data.get("vendor_tax_number"),
                },
                "currency": currency,
                "subtotal": parsed_data.get("subtotal"),
                "tax_info": {
                    "tax_rate": float(parsed_data.get("tax_rate", 0)) / 100 if parsed_data.get("tax_rate") and float(parsed_data.get("tax_rate", 0)) > 1 else parsed_data.get("tax_rate"),
                    "tax_amount": parsed_data.get("tax_amount"),
                } if parsed_data.get("tax_rate") or parsed_data.get("tax_amount") else None,
                "total_amount": parsed_data.get("total_amount"),
                "line_items": [
                    {
                        "description": item.get("description"),
                        "quantity": item.get("quantity"),
                        "unit_price": item.get("unit_price"),
                        "total_price": item.get("line_total", item.get("total_price")),
                    }
                    for item in parsed_data.get("line_items", [])
                ],
                "extracted_text": extracted_text,
                "processing_timestamp": datetime.now(timezone.utc)
            }

            return InvoiceData.model_validate(invoice_data)

        except Exception as e:
            logger.error(f"Text parsing failed: {str(e)}")
            logger.error(f"Extracted text was: {extracted_text[:500]}...")
            # Return empty InvoiceData if parsing fails
            return InvoiceData(extracted_text=extracted_text)
