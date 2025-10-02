"""
LLM service for advanced invoice processing using OpenAI GPT-4.
"""

import json
import logging
import time
from typing import Dict, Any

from openai import OpenAI

from app.config import settings
from app.models.invoice import ProcessedInvoice, ProcessingPhase
from app.models.response import InvoiceAnalysisResponse
from app.models.cost_tracking import CostTracker, ServiceProvider, ModelUsage, estimate_tokens, calculate_cost
from app.prompts.summary import SUMMARY_PROMPT


logger = logging.getLogger(__name__)


class LLMService:
    """LLM service for advanced invoice processing."""
    
    def __init__(self):
        """Initialize LLM service."""
        if settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.client = None

        # Model selection based on cost optimization
        self.text_model = settings.text_model if settings.use_cheap_models else "gpt-4"
        self.vision_model = settings.vision_model if settings.use_cheap_models else "gpt-4o"

        # Cost tracking
        self.cost_tracker = CostTracker()
    
    async def generate_summary(self, processed_invoice: ProcessedInvoice) -> InvoiceAnalysisResponse:
        """
        Generate final summary and analysis response.
        
        Args:
            processed_invoice: Processed invoice data
            
        Returns:
            InvoiceAnalysisResponse: Complete analysis response
        """
        start_time = time.time()
        
        try:
            # Generate intelligent summary using GPT-4 if available
            if self.client:
                summary_data = await self._generate_intelligent_summary(processed_invoice)
            else:
                summary_data = self._generate_basic_summary(processed_invoice)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(processed_invoice)
            processed_invoice.overall_confidence = overall_confidence
            
            # Generate processing recommendations
            recommendations = await self._generate_recommendations(processed_invoice)
            
            # Create final response
            response = InvoiceAnalysisResponse(
                invoice=processed_invoice,
                summary=summary_data,
                recommendations=recommendations
            )
            
            # Add processing phase
            processing_time = int((time.time() - start_time) * 1000)
            processed_invoice.processing_phases.append(
                ProcessingPhase(
                    phase_name="Summary_Generation",
                    status="success",
                    duration_ms=processing_time,
                    confidence=overall_confidence,
                    details={"summary_generated": True}
                )
            )
            
            logger.info("Summary generation completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            
            # Add error phase
            processing_time = int((time.time() - start_time) * 1000)
            processed_invoice.processing_phases.append(
                ProcessingPhase(
                    phase_name="Summary_Generation",
                    status="error",
                    duration_ms=processing_time,
                    confidence=0.0,
                    errors=[str(e)]
                )
            )
            
            # Return basic response even if summary generation fails
            return InvoiceAnalysisResponse(
                invoice=processed_invoice,
                summary={"error": "Summary generation failed"},
                recommendations=["Manual review recommended due to processing errors"]
            )
    
    async def _generate_intelligent_summary(self, processed_invoice: ProcessedInvoice) -> Dict[str, Any]:
        """
        Generate intelligent summary using GPT-4.
        
        Args:
            processed_invoice: Processed invoice data
            
        Returns:
            Dict[str, Any]: Generated summary data
        """
        try:
            # Prepare invoice data for analysis
            invoice_json = processed_invoice.model_dump_json(indent=2)
            input_text = f"{SUMMARY_PROMPT}\n\nFatura verisi:\n{invoice_json}"

            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Sen bir fatura analiz uzmanısın. Sadece JSON formatında cevap ver, başka hiçbir şey yazma."
                    },
                    {
                        "role": "user",
                        "content": input_text
                    }
                ],
                max_tokens=800,
                temperature=0.1
            )
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Parse the response
            summary_text = response.choices[0].message.content.strip()
            logger.info(f"Summary response: {summary_text[:200]}...")

            # Track cost
            usage = self.cost_tracker.track_usage(
                provider=ServiceProvider.OPENAI,
                model_name=self.text_model,
                input_text=input_text,
                output_text=summary_text,
                duration_ms=duration_ms
            )

            # Clean up the response (remove markdown code blocks if present)
            if summary_text.startswith('```json'):
                summary_text = summary_text[7:]
            if summary_text.endswith('```'):
                summary_text = summary_text[:-3]
            summary_text = summary_text.strip()

            # Find JSON content between braces
            start_idx = summary_text.find('{')
            end_idx = summary_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                summary_text = summary_text[start_idx:end_idx+1]

            # Try to parse as JSON, fallback to text summary
            try:
                summary_data = json.loads(summary_text)
                return summary_data
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse summary as JSON: {e}")
                logger.warning(f"Raw response: {summary_text}")

                # Return structured fallback
                return {
                    "invoice_summary": {
                        "vendor": processed_invoice.invoice_data.vendor.name if processed_invoice.invoice_data.vendor else "Unknown",
                        "date": str(processed_invoice.invoice_data.invoice_date) if processed_invoice.invoice_data.invoice_date else "Unknown",
                        "total": float(processed_invoice.invoice_data.total_amount) if processed_invoice.invoice_data.total_amount else 0.0,
                        "currency": processed_invoice.invoice_data.currency or "Unknown",
                        "category": "mixed",
                        "status": "needs_review"
                    },
                    "key_insights": [
                        "Fatura başarıyla işlendi",
                        "Manuel inceleme önerilir"
                    ],
                    "action_required": {
                        "immediate": [],
                        "review": ["Fatura detaylarını kontrol et"],
                        "none": []
                    },
                    "processing_decision": {
                        "method": "standard",
                        "confidence": 0.7,
                        "reason": "Normal işlem akışı"
                    }
                }

        except Exception as e:
            logger.error(f"Intelligent summary generation failed: {str(e)}")
            return {
                "error": "Summary generation failed",
                "fallback_summary": self._generate_basic_summary(processed_invoice)
            }
    
    def _generate_basic_summary(self, processed_invoice: ProcessedInvoice) -> Dict[str, Any]:
        """
        Generate basic summary without LLM.
        
        Args:
            processed_invoice: Processed invoice data
            
        Returns:
            Dict[str, Any]: Basic summary data
        """
        invoice_data = processed_invoice.invoice_data
        
        return {
            "vendor_name": invoice_data.vendor.name,
            "invoice_number": invoice_data.invoice_number,
            "total_amount": float(invoice_data.total_amount) if invoice_data.total_amount else None,
            "currency": invoice_data.currency,
            "line_items_count": len(invoice_data.line_items),
            "processing_phases": len(processed_invoice.processing_phases),
            "validation_issues": len(processed_invoice.validation_issues),
            "overall_confidence": processed_invoice.overall_confidence
        }
    
    def _calculate_overall_confidence(self, processed_invoice: ProcessedInvoice) -> float:
        """
        Calculate overall confidence score based on processing phases.
        
        Args:
            processed_invoice: Processed invoice data
            
        Returns:
            float: Overall confidence score (0.0 to 1.0)
        """
        if not processed_invoice.processing_phases:
            return 0.0
        
        # Calculate weighted average of phase confidences
        total_weight = 0
        weighted_confidence = 0
        
        phase_weights = {
            "OCR_GPT4_Vision": 0.4,
            "OCR_Textract_Fallback": 0.3,
            "Categorization": 0.2,
            "Validation": 0.3,
            "Summary_Generation": 0.1
        }
        
        for phase in processed_invoice.processing_phases:
            if phase.confidence is not None:
                weight = phase_weights.get(phase.phase_name, 0.1)
                weighted_confidence += phase.confidence * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        base_confidence = weighted_confidence / total_weight
        
        # Apply penalties for validation issues
        error_penalty = len([issue for issue in processed_invoice.validation_issues if issue.severity == "error"]) * 0.1
        warning_penalty = len([issue for issue in processed_invoice.validation_issues if issue.severity == "warning"]) * 0.05
        
        final_confidence = max(0.0, base_confidence - error_penalty - warning_penalty)
        
        return min(1.0, final_confidence)

    async def generate_text(self, prompt: str, use_vision: bool = False) -> str:
        """
        Generate text using OpenAI API with cost optimization.

        Args:
            prompt: Input prompt
            use_vision: Whether to use vision model

        Returns:
            str: Generated text
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized")

        try:
            model = self.vision_model if use_vision else self.text_model

            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000 if use_vision else 1500,  # Reduce tokens for cost
                temperature=0.1
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    async def _generate_recommendations(self, processed_invoice: ProcessedInvoice) -> list[str]:
        """
        Generate processing recommendations using GPT-4.
        
        Args:
            processed_invoice: Processed invoice data
            
        Returns:
            list[str]: List of recommendations
        """
        try:
            # Prepare context for recommendations
            context = {
                "confidence_level": processed_invoice.confidence_level,
                "overall_confidence": processed_invoice.overall_confidence,
                "validation_issues": len(processed_invoice.validation_issues),
                "error_issues": len([issue for issue in processed_invoice.validation_issues if issue.severity == "error"]),
                "warning_issues": len([issue for issue in processed_invoice.validation_issues if issue.severity == "warning"]),
                "missing_data": self._identify_missing_data(processed_invoice.invoice_data),
                "processing_errors": [phase.phase_name for phase in processed_invoice.processing_phases if phase.errors]
            }
            
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert invoice processing analyst. Generate specific, actionable recommendations based on the invoice processing results. Focus on data quality, accuracy, and next steps. SADECE JSON array döndür, başka hiçbir şey yazma. Örnek: ["Recommendation 1", "Recommendation 2"]"""
                    },
                    {
                        "role": "user",
                        "content": f"Generate recommendations based on this processing context:\n\n{json.dumps(context, indent=2)}"
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse recommendations
            recommendations_text = response.choices[0].message.content.strip()

            # Clean up the response (remove markdown code blocks if present)
            if recommendations_text.startswith('```json'):
                recommendations_text = recommendations_text[7:]
            if recommendations_text.endswith('```'):
                recommendations_text = recommendations_text[:-3]
            recommendations_text = recommendations_text.strip()

            # Find JSON array content between brackets
            start_idx = recommendations_text.find('[')
            end_idx = recommendations_text.rfind(']')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                recommendations_text = recommendations_text[start_idx:end_idx+1]

            try:
                recommendations = json.loads(recommendations_text)
                if isinstance(recommendations, list):
                    return recommendations
                else:
                    return [str(recommendations)]
            except json.JSONDecodeError:
                # Fallback: split by lines and clean up
                lines = [line.strip() for line in recommendations_text.split('\n') if line.strip()]
                return lines[:10]  # Limit to 10 recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return self._generate_basic_recommendations(processed_invoice)
    
    def _generate_basic_recommendations(self, processed_invoice: ProcessedInvoice) -> list[str]:
        """
        Generate basic recommendations without LLM.
        
        Args:
            processed_invoice: Processed invoice data
            
        Returns:
            list[str]: Basic recommendations
        """
        recommendations = []
        
        # Confidence-based recommendations
        if processed_invoice.overall_confidence < 0.5:
            recommendations.append("Manual review strongly recommended due to low confidence score")
        elif processed_invoice.overall_confidence < 0.7:
            recommendations.append("Consider manual verification of key data points")
        
        # Validation-based recommendations
        error_count = len([issue for issue in processed_invoice.validation_issues if issue.severity == "error"])
        if error_count > 0:
            recommendations.append(f"Address {error_count} validation errors before processing")
        
        # Data completeness recommendations
        missing_data = self._identify_missing_data(processed_invoice.invoice_data)
        if missing_data:
            recommendations.append(f"Complete missing data: {', '.join(missing_data)}")
        
        return recommendations
    
    def _identify_missing_data(self, invoice_data) -> list[str]:
        """
        Identify missing critical data fields.
        
        Args:
            invoice_data: Invoice data to analyze
            
        Returns:
            list[str]: List of missing data fields
        """
        missing = []
        
        if not invoice_data.vendor.name:
            missing.append("vendor name")
        if not invoice_data.invoice_number:
            missing.append("invoice number")
        if not invoice_data.invoice_date:
            missing.append("invoice date")
        if not invoice_data.total_amount:
            missing.append("total amount")
        if not invoice_data.line_items:
            missing.append("line items")
        
        return missing
