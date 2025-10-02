"""
Categorization service for expense classification.
"""

import json
import logging
import time
from typing import Dict, List, Tuple

from openai import OpenAI

from app.config import settings
from app.models.invoice import InvoiceData, ExpenseCategory, ProcessingPhase
from app.prompts.categorization import CATEGORIZATION_PROMPT


logger = logging.getLogger(__name__)


class CategorizationService:
    """Service for categorizing invoice expenses using AI."""
    
    def __init__(self):
        """Initialize categorization service."""
        if settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.client = None
        
        # Category keywords for fallback classification
        self.category_keywords = {
            ExpenseCategory.OFFICE_SUPPLIES: [
                'paper', 'pen', 'pencil', 'stapler', 'folder', 'binder', 'supplies',
                'stationery', 'notebook', 'printer', 'ink', 'toner', 'office'
            ],
            ExpenseCategory.TRAVEL: [
                'flight', 'hotel', 'taxi', 'uber', 'lyft', 'rental', 'gas', 'fuel',
                'airline', 'accommodation', 'lodging', 'mileage', 'parking', 'toll'
            ],
            ExpenseCategory.MEALS: [
                'restaurant', 'food', 'lunch', 'dinner', 'breakfast', 'catering',
                'meal', 'coffee', 'cafe', 'dining', 'delivery', 'takeout'
            ],
            ExpenseCategory.UTILITIES: [
                'electricity', 'gas', 'water', 'internet', 'phone', 'mobile',
                'utility', 'power', 'energy', 'telecom', 'broadband'
            ],
            ExpenseCategory.RENT: [
                'rent', 'lease', 'rental', 'property', 'building', 'space',
                'office space', 'warehouse', 'facility'
            ],
            ExpenseCategory.MARKETING: [
                'advertising', 'marketing', 'promotion', 'campaign', 'social media',
                'google ads', 'facebook', 'linkedin', 'billboard', 'flyer'
            ],
            ExpenseCategory.PROFESSIONAL_SERVICES: [
                'legal', 'accounting', 'consulting', 'lawyer', 'attorney', 'cpa',
                'audit', 'tax preparation', 'advisory', 'professional'
            ],
            ExpenseCategory.EQUIPMENT: [
                'computer', 'laptop', 'monitor', 'keyboard', 'mouse', 'desk',
                'chair', 'furniture', 'machinery', 'tools', 'hardware'
            ],
            ExpenseCategory.SOFTWARE: [
                'software', 'license', 'subscription', 'saas', 'app', 'platform',
                'microsoft', 'adobe', 'google', 'aws', 'cloud', 'hosting'
            ],
            ExpenseCategory.INSURANCE: [
                'insurance', 'premium', 'coverage', 'policy', 'liability',
                'health insurance', 'auto insurance', 'property insurance'
            ],
            ExpenseCategory.TAXES: [
                'tax', 'irs', 'federal', 'state', 'local', 'property tax',
                'sales tax', 'payroll tax', 'income tax'
            ]
        }
    
    async def categorize_expenses(self, invoice_data: InvoiceData) -> InvoiceData:
        """
        Categorize invoice expenses using AI and fallback methods.
        
        Args:
            invoice_data: Invoice data with line items to categorize
            
        Returns:
            InvoiceData: Invoice data with categorized line items
        """
        start_time = time.time()
        
        try:
            logger.info("Starting expense categorization")
            
            if not invoice_data.line_items:
                logger.warning("No line items to categorize")
                return invoice_data
            
            # Try AI categorization first if client is available
            if self.client:
                try:
                    categorized_data = await self._categorize_with_ai(invoice_data)
                    method_used = "ai_categorization"
                    confidence = 0.85

                except Exception as ai_error:
                    logger.warning(f"AI categorization failed: {str(ai_error)}, using fallback")
                    categorized_data = self._categorize_with_keywords(invoice_data)
                    method_used = "keyword_fallback"
                    confidence = 0.6
            else:
                logger.info("OpenAI client not available, using keyword categorization")
                categorized_data = self._categorize_with_keywords(invoice_data)
                method_used = "keyword_fallback"
                confidence = 0.6
            
            # Add processing phase
            processing_time = int((time.time() - start_time) * 1000)
            if not hasattr(categorized_data, 'processing_phases'):
                categorized_data.processing_phases = []
            
            categorized_data.processing_phases.append(
                ProcessingPhase(
                    phase_name="Categorization",
                    status="success",
                    duration_ms=processing_time,
                    confidence=confidence,
                    details={
                        "method_used": method_used,
                        "items_categorized": len([item for item in categorized_data.line_items if item.category]),
                        "total_items": len(categorized_data.line_items)
                    }
                )
            )
            
            logger.info(f"Categorization completed using {method_used}")
            return categorized_data
            
        except Exception as e:
            logger.error(f"Categorization failed: {str(e)}")
            
            # Add error phase
            processing_time = int((time.time() - start_time) * 1000)
            if not hasattr(invoice_data, 'processing_phases'):
                invoice_data.processing_phases = []
            
            invoice_data.processing_phases.append(
                ProcessingPhase(
                    phase_name="Categorization",
                    status="error",
                    duration_ms=processing_time,
                    confidence=0.0,
                    errors=[str(e)]
                )
            )
            
            return invoice_data
    
    async def _categorize_with_ai(self, invoice_data: InvoiceData) -> InvoiceData:
        """
        Categorize expenses using OpenAI GPT-4.
        
        Args:
            invoice_data: Invoice data to categorize
            
        Returns:
            InvoiceData: Categorized invoice data
        """
        # Prepare line items for categorization
        items_text = []
        for i, item in enumerate(invoice_data.line_items):
            items_text.append(f"{i+1}. {item.description}")
        
        items_string = "\n".join(items_text)
        
        # Get vendor context for better categorization
        vendor_context = ""
        if invoice_data.vendor.name:
            vendor_context = f"Vendor: {invoice_data.vendor.name}\n"
        
        # Prepare invoice data for categorization
        invoice_data_dict = {
            "invoice_number": invoice_data.invoice_number,
            "vendor_name": invoice_data.vendor.name if invoice_data.vendor else None,
            "total_amount": invoice_data.total_amount,
            "currency": invoice_data.currency,
            "line_items": [
                {
                    "description": item.description,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "total_price": item.total_price
                }
                for item in invoice_data.line_items
            ]
        }

        # Prepare categorization prompt with Decimal handling
        def decimal_serializer(obj):
            if hasattr(obj, '__float__'):
                return float(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        full_prompt = CATEGORIZATION_PROMPT.format(
            invoice_data=json.dumps(invoice_data_dict, indent=2, ensure_ascii=False, default=decimal_serializer)
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert expense categorization system. Categorize each line item and provide confidence scores."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            max_tokens=1500,
            temperature=0.2
        )
        
        # Parse the response
        categorization_result = response.choices[0].message.content
        
        # Apply categorizations to line items
        categorized_data = self._apply_ai_categorizations(invoice_data, categorization_result)
        
        return categorized_data
    
    def _apply_ai_categorizations(self, invoice_data: InvoiceData, categorization_result: str) -> InvoiceData:
        """
        Apply AI categorization results to line items.
        
        Args:
            invoice_data: Original invoice data
            categorization_result: AI categorization response
            
        Returns:
            InvoiceData: Updated invoice data with categories
        """
        try:
            # Try to parse as JSON first
            try:
                categories_data = json.loads(categorization_result)
            except json.JSONDecodeError:
                # Fallback: parse text response
                categories_data = self._parse_text_categorization(categorization_result)

            # Apply overall categorization to all line items
            if isinstance(categories_data, dict):
                primary_category = categories_data.get('primary_category', 'other')
                confidence_score = categories_data.get('confidence_score', 0.5)

                # Apply to all line items
                for item in invoice_data.line_items:
                    try:
                        item.category = ExpenseCategory(primary_category.lower())
                        item.category_confidence = confidence_score
                    except ValueError:
                        # Try to find closest match
                        item.category = self._find_closest_category(primary_category)
                        item.category_confidence = confidence_score * 0.8  # Reduce confidence for fuzzy match
            
            return invoice_data
            
        except Exception as e:
            logger.error(f"Failed to apply AI categorizations: {str(e)}")
            # Fallback to keyword categorization
            return self._categorize_with_keywords(invoice_data)
    
    def _parse_text_categorization(self, text_result: str) -> List[Dict]:
        """
        Parse text-based categorization result.
        
        Args:
            text_result: Text response from AI
            
        Returns:
            List[Dict]: Parsed categorization data
        """
        categories = []
        lines = text_result.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    category_part = parts[1].strip()
                    
                    # Extract category and confidence if present
                    if '(' in category_part and ')' in category_part:
                        category = category_part.split('(')[0].strip()
                        confidence_str = category_part.split('(')[1].split(')')[0]
                        try:
                            confidence = float(confidence_str.replace('%', '')) / 100
                        except:
                            confidence = 0.7
                    else:
                        category = category_part
                        confidence = 0.7
                    
                    categories.append({
                        'category': category,
                        'confidence': confidence
                    })
        
        return categories
    
    def _categorize_with_keywords(self, invoice_data: InvoiceData) -> InvoiceData:
        """
        Categorize expenses using keyword matching as fallback.
        
        Args:
            invoice_data: Invoice data to categorize
            
        Returns:
            InvoiceData: Categorized invoice data
        """
        for item in invoice_data.line_items:
            if not item.category:
                category, confidence = self._classify_by_keywords(item.description)
                item.category = category
                item.category_confidence = confidence
        
        return invoice_data
    
    def _classify_by_keywords(self, description: str) -> Tuple[ExpenseCategory, float]:
        """
        Classify item by keyword matching.
        
        Args:
            description: Item description
            
        Returns:
            Tuple[ExpenseCategory, float]: Category and confidence score
        """
        description_lower = description.lower()
        
        # Score each category based on keyword matches
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in description_lower:
                    # Longer keywords get higher scores
                    score += len(keyword.split())
            
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            # Return category with highest score
            best_category = max(category_scores, key=category_scores.get)
            # Confidence based on score strength
            max_score = category_scores[best_category]
            confidence = min(0.8, 0.4 + (max_score * 0.1))
            return best_category, confidence
        
        # Default category if no matches
        return ExpenseCategory.OTHER, 0.3
    
    def _find_closest_category(self, category_name: str) -> ExpenseCategory:
        """
        Find closest matching category for fuzzy matches.
        
        Args:
            category_name: Category name to match
            
        Returns:
            ExpenseCategory: Closest matching category
        """
        category_name_lower = category_name.lower()
        
        # Direct mapping for common variations
        category_mappings = {
            'office': ExpenseCategory.OFFICE_SUPPLIES,
            'supplies': ExpenseCategory.OFFICE_SUPPLIES,
            'travel': ExpenseCategory.TRAVEL,
            'transportation': ExpenseCategory.TRAVEL,
            'food': ExpenseCategory.MEALS,
            'dining': ExpenseCategory.MEALS,
            'restaurant': ExpenseCategory.MEALS,
            'utilities': ExpenseCategory.UTILITIES,
            'utility': ExpenseCategory.UTILITIES,
            'rent': ExpenseCategory.RENT,
            'rental': ExpenseCategory.RENT,
            'marketing': ExpenseCategory.MARKETING,
            'advertising': ExpenseCategory.MARKETING,
            'legal': ExpenseCategory.PROFESSIONAL_SERVICES,
            'consulting': ExpenseCategory.PROFESSIONAL_SERVICES,
            'equipment': ExpenseCategory.EQUIPMENT,
            'hardware': ExpenseCategory.EQUIPMENT,
            'software': ExpenseCategory.SOFTWARE,
            'subscription': ExpenseCategory.SOFTWARE,
            'insurance': ExpenseCategory.INSURANCE,
            'tax': ExpenseCategory.TAXES,
            'taxes': ExpenseCategory.TAXES
        }
        
        # Check for direct mappings
        for key, category in category_mappings.items():
            if key in category_name_lower:
                return category
        
        # Default to OTHER if no match found
        return ExpenseCategory.OTHER
