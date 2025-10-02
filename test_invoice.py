#!/usr/bin/env python3
"""
Test script to create a sample invoice image and test the API.
"""

from PIL import Image, ImageDraw, ImageFont
import requests
import json

def create_test_invoice():
    """Create a simple test invoice image."""
    # Create a white image
    width, height = 800, 1000
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw invoice content
    y = 50
    
    # Header
    draw.text((50, y), "FATURA / INVOICE", fill='black', font=font_large)
    y += 60
    
    # Company info
    draw.text((50, y), "ABC Teknoloji Ltd. Şti.", fill='black', font=font_medium)
    y += 25
    draw.text((50, y), "Atatürk Cad. No: 123 Kadıköy/İstanbul", fill='black', font=font_small)
    y += 20
    draw.text((50, y), "Vergi No: 1234567890", fill='black', font=font_small)
    y += 40
    
    # Invoice details
    draw.text((50, y), "Fatura No: 2024-001", fill='black', font=font_medium)
    draw.text((400, y), "Tarih: 2024-10-02", fill='black', font=font_medium)
    y += 40
    
    # Customer info
    draw.text((50, y), "Müşteri:", fill='black', font=font_medium)
    y += 25
    draw.text((50, y), "XYZ Şirketi", fill='black', font=font_small)
    y += 20
    draw.text((50, y), "Bağdat Cad. No: 456 Üsküdar/İstanbul", fill='black', font=font_small)
    y += 20
    draw.text((50, y), "Vergi No: 0987654321", fill='black', font=font_small)
    y += 60
    
    # Table header
    draw.rectangle([50, y, 750, y+30], outline='black', width=2)
    draw.text((60, y+5), "Açıklama", fill='black', font=font_small)
    draw.text((300, y+5), "Miktar", fill='black', font=font_small)
    draw.text((400, y+5), "Birim Fiyat", fill='black', font=font_small)
    draw.text((550, y+5), "Toplam", fill='black', font=font_small)
    y += 30
    
    # Line items
    items = [
        ("Web Tasarım Hizmeti", "1", "5000.00", "5000.00"),
        ("Hosting (12 ay)", "1", "1200.00", "1200.00"),
        ("Domain (.com)", "1", "150.00", "150.00")
    ]
    
    for item in items:
        draw.rectangle([50, y, 750, y+25], outline='black', width=1)
        draw.text((60, y+5), item[0], fill='black', font=font_small)
        draw.text((300, y+5), item[1], fill='black', font=font_small)
        draw.text((400, y+5), item[2] + " TL", fill='black', font=font_small)
        draw.text((550, y+5), item[3] + " TL", fill='black', font=font_small)
        y += 25
    
    y += 40
    
    # Totals
    draw.text((400, y), "Ara Toplam:", fill='black', font=font_medium)
    draw.text((550, y), "6350.00 TL", fill='black', font=font_medium)
    y += 25
    
    draw.text((400, y), "KDV (%18):", fill='black', font=font_medium)
    draw.text((550, y), "1143.00 TL", fill='black', font=font_medium)
    y += 25
    
    draw.text((400, y), "GENEL TOPLAM:", fill='black', font=font_large)
    draw.text((550, y), "7493.00 TL", fill='black', font=font_large)
    
    # Save the image
    image.save('test_invoice.png')
    print("Test invoice image created: test_invoice.png")
    return 'test_invoice.png'

def test_api(image_path):
    """Test the invoice analysis API."""
    url = "http://localhost:8001/analyze-invoice?use_fallback=true"  # Test fallback
    
    with open(image_path, 'rb') as f:
        files = {'file': ('test_invoice.png', f, 'image/png')}
        
        print("Sending request to API...")
        response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("API Response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Error Response:")
            print(response.text)

if __name__ == "__main__":
    # Create test invoice
    image_path = create_test_invoice()
    
    # Test the API
    test_api(image_path)
