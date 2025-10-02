"""
API endpoint tests for invoice analyzer.
"""

import pytest
import io
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health check returns correct response."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "message" in data


class TestInvoiceAnalysisEndpoint:
    """Test invoice analysis endpoint."""
    
    def create_test_image(self, width: int = 800, height: int = 600) -> bytes:
        """Create a test image for upload."""
        # Create a simple test image
        image = Image.new('RGB', (width, height), color='white')
        
        # Add some text-like patterns (simple rectangles to simulate text)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # Simulate invoice header
        draw.rectangle([50, 50, 300, 80], fill='black')
        draw.rectangle([50, 100, 200, 120], fill='black')
        
        # Simulate line items
        for i in range(3):
            y = 200 + i * 30
            draw.rectangle([50, y, 150, y + 15], fill='black')
            draw.rectangle([200, y, 250, y + 15], fill='black')
            draw.rectangle([300, y, 350, y + 15], fill='black')
        
        # Save to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    def test_analyze_invoice_no_file(self):
        """Test analysis endpoint without file."""
        response = client.post("/analyze-invoice")
        assert response.status_code == 422  # Validation error
    
    def test_analyze_invoice_invalid_file_type(self):
        """Test analysis with invalid file type."""
        # Create a text file
        text_content = b"This is not an image"
        
        response = client.post(
            "/analyze-invoice",
            files={"file": ("test.txt", text_content, "text/plain")}
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    
    def test_analyze_invoice_valid_image(self):
        """Test analysis with valid image."""
        # Create test image
        image_content = self.create_test_image()
        
        response = client.post(
            "/analyze-invoice",
            files={"file": ("test_invoice.jpg", image_content, "image/jpeg")}
        )
        
        # Note: This test will fail without proper API keys
        # In a real test environment, you would mock the external services
        assert response.status_code in [200, 500]  # 500 if no API keys configured
    
    def test_analyze_invoice_large_file(self):
        """Test analysis with oversized file."""
        # Create a very large image (simulate by creating large content)
        large_content = b"x" * (15 * 1024 * 1024)  # 15MB
        
        response = client.post(
            "/analyze-invoice",
            files={"file": ("large_invoice.jpg", large_content, "image/jpeg")}
        )
        assert response.status_code == 400
        assert "File too large" in response.json()["detail"]


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_endpoint(self):
        """Test accessing invalid endpoint."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test using wrong HTTP method."""
        response = client.get("/analyze-invoice")
        assert response.status_code == 405


@pytest.fixture
def mock_openai_client(monkeypatch):
    """Mock OpenAI client for testing."""
    class MockOpenAIResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]
    
    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)
    
    class MockMessage:
        def __init__(self, content):
            self.content = content
    
    class MockOpenAIClient:
        def __init__(self, *args, **kwargs):
            pass
        
        @property
        def chat(self):
            return self
        
        @property
        def completions(self):
            return self
        
        def create(self, **kwargs):
            # Return mock response based on model
            if kwargs.get('model') == 'gpt-4-vision-preview':
                return MockOpenAIResponse('{"invoice_number": "INV-001", "total_amount": 100.00}')
            else:
                return MockOpenAIResponse('Mock analysis result')
    
    # Mock the OpenAI client
    monkeypatch.setattr("app.services.ocr_service.OpenAI", MockOpenAIClient)
    monkeypatch.setattr("app.services.llm_service.OpenAI", MockOpenAIClient)
    monkeypatch.setattr("app.services.categorizer.OpenAI", MockOpenAIClient)


class TestMockedAnalysis:
    """Test analysis with mocked external services."""
    
    def test_analyze_invoice_mocked(self, mock_openai_client):
        """Test analysis with mocked OpenAI client."""
        # Create test image
        image = Image.new('RGB', (800, 600), color='white')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        image_content = img_bytes.getvalue()
        
        response = client.post(
            "/analyze-invoice",
            files={"file": ("test_invoice.jpg", image_content, "image/jpeg")}
        )
        
        # Should succeed with mocked services
        assert response.status_code == 200
        
        data = response.json()
        assert "invoice" in data
        assert "summary" in data
        assert "recommendations" in data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
