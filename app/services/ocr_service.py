"""
OCR service for extracting text and data from invoice images.
"""

import base64
import json
import logging
import time
from typing import Optional, Tuple
from datetime import datetime, timezone

import httpx
from openai import OpenAI

from app.config import settings
from app.models.invoice import InvoiceData, ProcessingPhase
from app.models.cost_tracking import CostTracker, ServiceProvider
from app.services.textract_service import TextractService
from app.prompts.extraction import EXTRACTION_PROMPT


logger = logging.getLogger(__name__)


class OCRService:
    """3-Tier OCR service: Ollama LLaVA -> OpenAI GPT-4 Vision -> AWS Textract."""

    def __init__(self):
        """Initialize OCR service with 3-tier fallback system."""
        # Tier 2: OpenAI GPT-4 Vision
        if settings.openai_api_key:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.openai_client = None

        # Tier 3: AWS Textract
        self.textract_service = TextractService()

        # Configuration and tracking
        self.config = settings
        self.cost_tracker = CostTracker()

        # Tier 1: Ollama LLaVA client
        self.ollama_client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout
        )
        
    async def extract_data(self, file_content: bytes, filename: Optional[str] = None, use_fallback: bool = False) -> InvoiceData:
        """
        Extract invoice data using 3-tier fallback system:
        Tier 1: Ollama LLaVA 7B (Primary - Free)
        Tier 2: OpenAI GPT-4 Vision (Secondary - Low cost)
        Tier 3: AWS Textract (Last resort - High accuracy)

        Args:
            file_content: Image file content as bytes
            filename: Original filename
            use_fallback: Force skip to Tier 3 (Textract)

        Returns:
            InvoiceData: Extracted invoice data with tier info

        Raises:
            Exception: If all tiers fail
        """
        start_time = time.time()

        # Check if this is a high-value invoice that should go directly to Textract
        if use_fallback:
            logger.info("Using AWS Textract (Tier 3) as explicitly requested")
            return await self._extract_tier3_textract(file_content, filename, start_time, "explicit_request")

        # TIER 1: Try Ollama LLaVA first (Primary - Free)
        if settings.ollama_enabled:
            try:
                logger.info("🚀 TIER 1: Attempting OCR with Ollama LLaVA 7B")
                result, confidence = await self._extract_with_ollama_llava(file_content, filename)

                if confidence >= settings.ollama_confidence_threshold:
                    logger.info(f"✅ TIER 1 SUCCESS: Ollama confidence {confidence:.2f} >= {settings.ollama_confidence_threshold}")
                    processing_time = int((time.time() - start_time) * 1000)
                    result.processing_phases.append(
                        ProcessingPhase(
                            phase_name="OCR_Ollama_LLaVA",
                            status="success",
                            duration_ms=processing_time,
                            confidence=confidence,
                            details={
                                "tier": 1,
                                "method": "ollama_llava",
                                "model": settings.ollama_model,
                                "cost_estimate": 0.0
                            }
                        )
                    )
                    return result
                else:
                    logger.warning(f"⚠️ TIER 1 LOW CONFIDENCE: Ollama confidence {confidence:.2f} < {settings.ollama_confidence_threshold}, falling back to Tier 2")

            except Exception as ollama_error:
                logger.warning(f"❌ TIER 1 FAILED: Ollama LLaVA error: {str(ollama_error)}")
        else:
            logger.info("⏭️ TIER 1 DISABLED: Ollama is disabled, skipping to Tier 2")

        # TIER 2: Try OpenAI GPT-4 Vision (Secondary - Low cost)
        if self.openai_client:
            try:
                logger.info("🔄 TIER 2: Attempting OCR with OpenAI GPT-4 Vision")
                result, confidence = await self._extract_with_gpt4_vision(file_content, filename)

                if confidence >= settings.openai_confidence_threshold:
                    logger.info(f"✅ TIER 2 SUCCESS: OpenAI confidence {confidence:.2f} >= {settings.openai_confidence_threshold}")
                    processing_time = int((time.time() - start_time) * 1000)
                    result.processing_phases.append(
                        ProcessingPhase(
                            phase_name="OCR_OpenAI_GPT4_Vision",
                            status="success",
                            duration_ms=processing_time,
                            confidence=confidence,
                            details={
                                "tier": 2,
                                "method": "openai_gpt4_vision",
                                "model": settings.vision_model,
                                "cost_estimate": 0.002,
                                "fallback_reason": "ollama_low_confidence" if settings.ollama_enabled else "ollama_disabled"
                            }
                        )
                    )
                    return result
                else:
                    logger.warning(f"⚠️ TIER 2 LOW CONFIDENCE: OpenAI confidence {confidence:.2f} < {settings.openai_confidence_threshold}, falling back to Tier 3")

            except Exception as openai_error:
                logger.warning(f"❌ TIER 2 FAILED: OpenAI GPT-4 Vision error: {str(openai_error)}")
        else:
            logger.warning("⏭️ TIER 2 UNAVAILABLE: OpenAI API key not configured, skipping to Tier 3")

        # TIER 3: AWS Textract (Last resort - High accuracy)
        logger.info("🔧 TIER 3: Using AWS Textract as last resort")
        return await self._extract_tier3_textract(file_content, filename, start_time, "all_tiers_failed")

    async def _extract_with_ollama_llava(self, file_content: bytes, filename: Optional[str] = None) -> Tuple[InvoiceData, float]:
        """
        Extract data using Ollama LLaVA 7B model.

        Args:
            file_content: Image file content
            filename: Original filename

        Returns:
            Tuple[InvoiceData, confidence_score]: Extracted data and confidence
        """
        # Encode image to base64
        base64_image = base64.b64encode(file_content).decode('utf-8')

        try:
            start_time = time.time()

            # Prepare Ollama API request
            payload = {
                "model": settings.ollama_model,
                "prompt": EXTRACTION_PROMPT,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 2000
                }
            }

            # Make request to Ollama
            response = await self.ollama_client.post("/api/generate", json=payload)
            response.raise_for_status()

            duration_ms = int((time.time() - start_time) * 1000)
            result_data = response.json()

            # Extract response text
            extracted_text = result_data.get("response", "").strip()
            if not extracted_text:
                raise Exception("Empty response from Ollama LLaVA")

            logger.info(f"Ollama LLaVA response (first 200 chars): {extracted_text[:200]}...")

            # Track cost (free for self-hosted)
            self.cost_tracker.track_usage(
                provider=ServiceProvider.OPENAI,  # Using OpenAI enum for now
                model_name=f"ollama_{settings.ollama_model}",
                input_text="[IMAGE_INPUT]",
                output_text=extracted_text,
                duration_ms=duration_ms
            )

            # Convert to structured data using LLM parsing
            structured_data = await self._parse_extracted_text(extracted_text)

            # Set metadata
            structured_data.extracted_text = extracted_text
            structured_data.original_filename = filename

            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(structured_data)

            # Log successful extraction
            logger.info(f"Ollama extracted: invoice_number={structured_data.invoice_number}, total_amount={structured_data.total_amount}")

            logger.info(f"Ollama LLaVA extraction completed with confidence: {confidence:.2f}")
            return structured_data, confidence

        except Exception as e:
            logger.error(f"Ollama LLaVA extraction failed: {str(e)}")
            raise

    def _calculate_confidence(self, invoice_data: InvoiceData) -> float:
        """
        Calculate confidence score based on data completeness and quality.

        Args:
            invoice_data: Extracted invoice data

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        score = 0.0
        max_score = 0.0

        # Essential fields (higher weight)
        essential_fields = [
            ("total_amount", 0.25),
            ("invoice_number", 0.20),
            ("invoice_date", 0.15),
            ("vendor.name", 0.15)
        ]

        # Optional fields (lower weight)
        optional_fields = [
            ("currency", 0.10),
            ("subtotal", 0.05),
            ("tax_info", 0.05),
            ("line_items", 0.05)
        ]

        all_fields = essential_fields + optional_fields

        for field_path, weight in all_fields:
            max_score += weight

            # Navigate nested fields
            value = invoice_data
            for field in field_path.split('.'):
                if hasattr(value, field):
                    value = getattr(value, field)
                else:
                    value = None
                    break

            # Check if field has meaningful value
            if value is not None:
                if isinstance(value, str) and value.strip():
                    score += weight
                elif isinstance(value, (int, float)) and value > 0:
                    score += weight
                elif isinstance(value, list) and len(value) > 0:
                    score += weight
                elif hasattr(value, '__dict__'):  # Object with attributes
                    if any(getattr(value, attr, None) for attr in dir(value) if not attr.startswith('_')):
                        score += weight

        confidence = score / max_score if max_score > 0 else 0.0
        return min(1.0, max(0.0, confidence))  # Clamp between 0 and 1
    
    async def _extract_with_gpt4_vision(self, file_content: bytes, filename: Optional[str] = None) -> Tuple[InvoiceData, float]:
        """
        Extract data using OpenAI GPT-4 Vision.

        Args:
            file_content: Image file content
            filename: Original filename

        Returns:
            Tuple[InvoiceData, confidence_score]: Extracted data and confidence
        """
        # Encode image to base64
        base64_image = base64.b64encode(file_content).decode('utf-8')
        
        try:
            if not self.openai_client:
                raise Exception("OpenAI client not initialized")

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
            if not extracted_text:
                raise Exception("Empty response from OpenAI GPT-4 Vision")

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

            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(structured_data)

            return structured_data, confidence
            
        except Exception as e:
            logger.error(f"GPT-4 Vision extraction failed: {str(e)}")
            raise

    async def _extract_tier3_textract(self, file_content: bytes, filename: Optional[str], start_time: float, fallback_reason: str) -> InvoiceData:
        """
        Extract data using AWS Textract (Tier 3 - Last resort).

        Args:
            file_content: Image file content
            filename: Original filename
            start_time: Processing start time
            fallback_reason: Reason for using Tier 3

        Returns:
            InvoiceData: Extracted data with tier info
        """
        try:
            logger.info("🔧 TIER 3: Extracting with AWS Textract")

            # Use Textract service
            extracted_text = await self.textract_service.extract_text(file_content)

            # Convert to structured data using LLM
            structured_data = await self._parse_extracted_text(extracted_text)

            # Set metadata
            structured_data.extracted_text = extracted_text
            structured_data.original_filename = filename

            # Textract is considered high confidence (0.85)
            confidence = 0.85

            processing_time = int((time.time() - start_time) * 1000)
            structured_data.processing_phases.append(
                ProcessingPhase(
                    phase_name="OCR_AWS_Textract",
                    status="success",
                    duration_ms=processing_time,
                    confidence=confidence,
                    details={
                        "tier": 3,
                        "method": "aws_textract",
                        "cost_estimate": 0.05,
                        "fallback_reason": fallback_reason
                    }
                )
            )

            logger.info(f"✅ TIER 3 SUCCESS: AWS Textract completed with confidence {confidence:.2f}")
            return structured_data

        except Exception as textract_error:
            logger.error(f"❌ TIER 3 FAILED: AWS Textract error: {str(textract_error)}")

            # All tiers failed - return empty result with error info
            processing_time = int((time.time() - start_time) * 1000)
            result = InvoiceData()
            result.processing_phases.append(
                ProcessingPhase(
                    phase_name="OCR_All_Tiers_Failed",
                    status="error",
                    duration_ms=processing_time,
                    confidence=0.0,
                    errors=[
                        f"All 3 tiers failed. Final error: {str(textract_error)}",
                        f"Fallback reason: {fallback_reason}"
                    ]
                )
            )

            raise Exception(f"All 3 OCR tiers failed. Final error: {textract_error}")
    
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
            # Check if extracted_text is JSON (from GPT-4 Vision/Ollama) or raw text (from Textract)
            extracted_text = extracted_text.strip()

            # Clean up markdown code blocks first (from Ollama responses)
            if extracted_text.startswith('```json'):
                extracted_text = extracted_text[7:]
            if extracted_text.endswith('```'):
                extracted_text = extracted_text[:-3]
            extracted_text = extracted_text.strip()

            # Log text parsing attempt
            logger.debug(f"Parsing extracted text ({len(extracted_text)} chars)")

            # Try to parse as JSON first (from GPT-4 Vision/Ollama)
            # For Ollama, we need to extract JSON from mixed content
            json_text = extracted_text
            if extracted_text.startswith('{'):
                # Find the end of JSON by counting braces
                brace_count = 0
                json_end = -1
                for i, char in enumerate(extracted_text):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break

                if json_end > 0:
                    json_text = extracted_text[:json_end]
                    logger.debug(f"Extracted JSON portion from mixed content")

            if json_text.startswith('{') and json_text.endswith('}'):
                try:
                    parsed_data = json.loads(json_text)
                    logger.debug(f"Successfully parsed JSON with {len(parsed_data)} fields")

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

                    result = InvoiceData.model_validate(invoice_data)
                    logger.debug(f"Successfully validated InvoiceData: {result.invoice_number}")
                    return result

                except json.JSONDecodeError:
                    logger.warning("Text looks like JSON but failed to parse, treating as raw text")

            # If not JSON or JSON parsing failed, treat as raw text and use LLM to parse
            logger.warning("Extracted text is not JSON, using LLM to parse")

            # Use cheaper model to structure the extracted text
            if not self.openai_client:
                raise Exception("OpenAI client not initialized")

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

            # Parse JSON response
            llm_response = response.choices[0].message.content
            if not llm_response:
                raise Exception("Empty response from OpenAI text model")

            llm_response = llm_response.strip()

            # Track cost for text parsing
            self.cost_tracker.track_usage(
                provider=ServiceProvider.OPENAI,
                model_name=self.config.text_model,
                input_text=f"Parse this invoice text:\n\n{extracted_text}",
                output_text=llm_response,
                duration_ms=duration_ms
            )
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
