"""
Prompt templates for data validation (Phase 3).
"""

VALIDATION_PROMPT = """
Sana analiz edilmiş fatura datası veriyorum. Validate et ve anomalileri tespit et.

**GİRDİ:**
{combined_data}

**KONTROLLER:**

1. **MATEMATİKSEL DOĞRULUK:**
   - Satır toplamları: quantity × unit_price = line_total
   - Ara toplam: tüm line_total toplamı = subtotal
   - KDV: subtotal × (tax_rate/100) = tax_amount
   - Toplam: subtotal + tax_amount = total_amount
   - Tolerans: ±0.01 (yuvarlama hatası)

2. **MANTIKSAL KONTROLLER:**
   - Fatura tarihi gelecekte değil
   - Vade tarihi >= fatura tarihi
   - Negatif değer yok (iade faturası değilse)
   - Tutar mantıklı aralıkta

3. **EKSİK VERİ:**
   - Kritik alanlar: invoice_number, vendor_name, total_amount
   - Zorunlu: tax_number, date

4. **FALLBACK GEREKTİREN DURUMLAR:**
   - overall_confidence < 0.80
   - ocr_quality < 0.70
   - 3+ kritik alan eksik
   - Validation error'da "high" severity var
   - total_amount > 10000 VE overall_confidence < 0.90
   - El yazısı görseller (ocr_quality < 0.50)

**ÇIKTI:**
{
  "is_valid": true/false,
  "validation_errors": [
    {
      "field": "...",
      "error_type": "...",
      "message": "...",
      "severity": "high/medium/low"
    }
  ],
  "anomalies": [
    {
      "type": "...",
      "message": "...",
      "severity": "...",
      "recommendation": "..."
    }
  ],
  "should_use_fallback": true/false,
  "fallback_reason": "...",
  "overall_quality_score": 0.92
}

SADECE JSON döndür, başka hiçbir şey yazma.
"""

CROSS_VALIDATION_PROMPT = """
You are performing cross-validation between different data extraction methods and sources.

Compare and validate data consistency across:

1. **OCR Results Comparison:**
   - Primary OCR vs fallback OCR results
   - Confidence in different extraction methods
   - Conflicting data points identification

2. **Calculation Verification:**
   - Line items sum vs stated subtotal
   - Subtotal + tax vs stated total
   - Individual line calculations (qty × price)
   - Tax rate vs tax amount consistency

3. **Logical Relationships:**
   - Invoice date vs due date
   - Vendor type vs expense categories
   - Payment terms vs due date
   - Currency consistency throughout

4. **Data Source Validation:**
   - Header information vs line item details
   - Multiple mentions of same data (consistency)
   - Document structure validation

Return cross-validation results:

{
  "cross_validation_summary": {
    "consistency_score": 0.92,
    "conflicts_found": 2,
    "calculation_errors": 1,
    "logical_issues": 0
  },
  "conflicts": [
    {
      "field": "field_name",
      "conflict_type": "calculation|format|logical|source",
      "description": "Description of conflict",
      "values": ["value1", "value2"],
      "recommended_resolution": "How to resolve",
      "confidence": 0.8
    }
  ],
  "calculations_verified": [
    {
      "calculation_type": "line_item_total|subtotal|tax|grand_total",
      "expected": 100.00,
      "actual": 100.00,
      "status": "correct|incorrect|unclear",
      "variance": 0.00
    }
  ]
}
"""
