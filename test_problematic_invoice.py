#!/usr/bin/env python3
"""
Test script for problematic invoice that should trigger fallback.
Creates an invoice with mismatched line items vs subtotal.
"""

import requests
from PIL import Image, ImageDraw, ImageFont
import json

def create_problematic_invoice():
    """Create a test invoice image with calculation errors."""
    # Create image
    img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    y = 50
    
    # Header
    draw.text((50, y), "INVOICE", font=font_large, fill='black')
    y += 40
    
    # Vendor info
    draw.text((50, y), "France Sant", font=font_medium, fill='black')
    y += 30
    
    # Invoice details
    draw.text((50, y), "Invoice #: 88", font=font_small, fill='black')
    y += 25
    draw.text((50, y), "Date: 2025-04-20", font=font_small, fill='black')
    y += 25
    draw.text((50, y), "Currency: EUR", font=font_small, fill='black')
    y += 40
    
    # Customer
    draw.text((50, y), "Bill To: Jivewise", font=font_medium, fill='black')
    y += 40
    
    # Line items header
    draw.text((50, y), "Description", font=font_small, fill='black')
    draw.text((300, y), "Qty", font=font_small, fill='black')
    draw.text((400, y), "Price", font=font_small, fill='black')
    draw.text((500, y), "Total", font=font_small, fill='black')
    y += 30
    
    # Line items (these will create calculation mismatch)
    line_items = [
        ("15 divers", "15", "300.00", "4500.00"),
        ("10 divers", "10", "300.00", "3000.00"),
        ("13 divers", "13", "300.00", "3900.00"),
        ("10 divers", "10", "300.00", "3000.00"),
        ("11 divers", "11", "300.00", "3300.00"),
        ("12 divers", "12", "300.00", "3600.00"),
        ("5 divers", "5", "200.00", "1000.00"),
    ]
    
    for desc, qty, price, total in line_items:
        draw.text((50, y), desc, font=font_small, fill='black')
        draw.text((300, y), qty, font=font_small, fill='black')
        draw.text((400, y), price, font=font_small, fill='black')
        draw.text((500, y), total, font=font_small, fill='black')
        y += 25
    
    y += 20
    
    # Totals (WRONG - this will cause validation error)
    draw.text((400, y), "Subtotal:", font=font_small, fill='black')
    draw.text((500, y), "2000.00 EUR", font=font_small, fill='black')  # Should be 22300!
    y += 25
    
    draw.text((400, y), "TOTAL:", font=font_medium, fill='black')
    draw.text((500, y), "2000.00 EUR", font=font_medium, fill='black')
    
    # Save image
    img.save('problematic_invoice.png')
    print("Problematic invoice image created: problematic_invoice.png")
    return 'problematic_invoice.png'

def test_api():
    """Test the API with problematic invoice."""
    filename = create_problematic_invoice()
    
    print("Sending request to API...")
    url = "http://localhost:8003/analyze-invoice"
    
    with open(filename, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("API Response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Check if fallback was used
        if result.get('invoice', {}).get('processing_metadata', {}).get('fallback_used'):
            print("\n✅ FALLBACK WAS USED!")
            print(f"Reason: {result['invoice']['processing_metadata'].get('fallback_reason', 'unknown')}")
        else:
            print("\n❌ FALLBACK WAS NOT USED")
            
        # Check processing decision
        summary = result.get('summary', {})
        if summary.get('processing_decision', {}).get('method') == 'fallback_required':
            print(f"✅ Summary recommends fallback: {summary['processing_decision'].get('reason')}")
        
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_api()
