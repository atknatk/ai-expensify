"""
AWS Textract service for OCR fallback.
"""

import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from app.config import settings


logger = logging.getLogger(__name__)


class TextractService:
    """AWS Textract service for document text extraction."""
    
    def __init__(self):
        """Initialize Textract service."""
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize AWS Textract client if credentials are available."""
        try:
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                self.client = boto3.client(
                    'textract',
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                    region_name=settings.aws_region
                )
                logger.info("AWS Textract client initialized successfully")
            else:
                logger.warning("AWS credentials not provided, Textract will not be available")
        except Exception as e:
            logger.error(f"Failed to initialize Textract client: {str(e)}")
            self.client = None
    
    async def extract_text(self, file_content: bytes) -> str:
        """
        Extract text from document using AWS Textract.
        
        Args:
            file_content: Document file content as bytes
            
        Returns:
            str: Extracted text
            
        Raises:
            Exception: If Textract is not available or extraction fails
        """
        if not self.client:
            raise Exception("AWS Textract client not available. Check AWS credentials.")
        
        try:
            # Call Textract detect_document_text
            response = self.client.detect_document_text(
                Document={'Bytes': file_content}
            )
            
            # Extract text from response
            extracted_text = self._parse_textract_response(response)
            
            logger.info("Text extraction completed successfully with Textract")
            return extracted_text
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Textract API error [{error_code}]: {error_message}")
            raise Exception(f"Textract extraction failed: {error_message}")
        
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise Exception("AWS credentials not configured for Textract")
        
        except Exception as e:
            logger.error(f"Unexpected error during Textract extraction: {str(e)}")
            raise Exception(f"Textract extraction failed: {str(e)}")
    
    def _parse_textract_response(self, response: dict) -> str:
        """
        Parse Textract response and extract text.
        
        Args:
            response: Textract API response
            
        Returns:
            str: Extracted and formatted text
        """
        text_blocks = []
        
        # Extract text from blocks
        for block in response.get('Blocks', []):
            if block['BlockType'] == 'LINE':
                text_blocks.append(block.get('Text', ''))
        
        # Join text blocks with newlines
        extracted_text = '\n'.join(text_blocks)
        
        return extracted_text
    
    def is_available(self) -> bool:
        """
        Check if Textract service is available.
        
        Returns:
            bool: True if Textract client is initialized and ready
        """
        return self.client is not None
    
    async def analyze_expense(self, file_content: bytes) -> dict:
        """
        Analyze expense document using Textract's AnalyzeExpense API.
        
        Args:
            file_content: Document file content as bytes
            
        Returns:
            dict: Structured expense data
            
        Raises:
            Exception: If Textract is not available or analysis fails
        """
        if not self.client:
            raise Exception("AWS Textract client not available. Check AWS credentials.")
        
        try:
            # Call Textract analyze_expense
            response = self.client.analyze_expense(
                Document={'Bytes': file_content}
            )
            
            # Parse expense analysis response
            expense_data = self._parse_expense_response(response)
            
            logger.info("Expense analysis completed successfully with Textract")
            return expense_data
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Textract AnalyzeExpense error [{error_code}]: {error_message}")
            raise Exception(f"Textract expense analysis failed: {error_message}")
        
        except Exception as e:
            logger.error(f"Unexpected error during Textract expense analysis: {str(e)}")
            raise Exception(f"Textract expense analysis failed: {str(e)}")
    
    def _parse_expense_response(self, response: dict) -> dict:
        """
        Parse Textract AnalyzeExpense response.
        
        Args:
            response: Textract AnalyzeExpense API response
            
        Returns:
            dict: Parsed expense data
        """
        expense_data = {
            'vendor_name': None,
            'invoice_date': None,
            'total_amount': None,
            'line_items': []
        }
        
        # Parse expense documents
        for expense_doc in response.get('ExpenseDocuments', []):
            # Parse summary fields
            for summary_field in expense_doc.get('SummaryFields', []):
                field_type = summary_field.get('Type', {}).get('Text', '')
                field_value = summary_field.get('ValueDetection', {}).get('Text', '')
                
                if field_type == 'VENDOR_NAME':
                    expense_data['vendor_name'] = field_value
                elif field_type == 'INVOICE_RECEIPT_DATE':
                    expense_data['invoice_date'] = field_value
                elif field_type == 'TOTAL':
                    expense_data['total_amount'] = field_value
            
            # Parse line items
            for line_item_group in expense_doc.get('LineItemGroups', []):
                for line_item in line_item_group.get('LineItems', []):
                    item_data = {}
                    
                    for field in line_item.get('LineItemExpenseFields', []):
                        field_type = field.get('Type', {}).get('Text', '')
                        field_value = field.get('ValueDetection', {}).get('Text', '')
                        
                        if field_type == 'ITEM':
                            item_data['description'] = field_value
                        elif field_type == 'PRICE':
                            item_data['total_price'] = field_value
                        elif field_type == 'QUANTITY':
                            item_data['quantity'] = field_value
                    
                    if item_data:
                        expense_data['line_items'].append(item_data)
        
        return expense_data
