# Invoice Analyzer API

AI-powered invoice analysis API that extracts, categorizes, and validates invoice data using OCR and LLM technologies.

## Features

- **Multi-phase Analysis**: 4-phase processing pipeline for accurate invoice data extraction
- **OCR Integration**: Primary OpenAI GPT-4 Vision with AWS Textract fallback
- **Smart Categorization**: Automatic expense categorization with confidence scoring
- **Data Validation**: Multi-layer validation for accuracy and completeness
- **RESTful API**: FastAPI-based REST endpoints with automatic documentation

## Architecture

### Processing Pipeline

1. **Phase 1 - Data Extraction**: OCR processing to extract raw invoice data
2. **Phase 2 - Categorization**: AI-powered expense categorization
3. **Phase 3 - Validation**: Data accuracy and completeness validation
4. **Phase 4 - Summary**: Final processing and confidence scoring

### Technology Stack

- **Framework**: FastAPI
- **OCR**: OpenAI GPT-4 Vision (primary), AWS Textract (fallback)
- **Language Model**: OpenAI GPT-4
- **Validation**: Pydantic models
- **Image Processing**: Pillow

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd invoice-analyzer
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the application:
```bash
poetry run uvicorn app.main:app --reload
```

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for GPT-4 Vision and text processing
- `AWS_ACCESS_KEY_ID`: AWS access key for Textract (optional)
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for Textract (optional)
- `AWS_REGION`: AWS region for Textract (default: us-east-1)

## API Endpoints

- `POST /analyze-invoice`: Upload and analyze invoice image
- `GET /health`: Health check endpoint
- `GET /docs`: Interactive API documentation

## Usage

```python
import httpx

# Upload invoice for analysis
with open("invoice.jpg", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/analyze-invoice",
        files={"file": f}
    )
    
result = response.json()
print(result)
```

## Development

Run tests:
```bash
poetry run pytest
```

Format code:
```bash
poetry run black .
poetry run isort .
```

Type checking:
```bash
poetry run mypy app/
```
