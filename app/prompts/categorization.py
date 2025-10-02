"""
Prompt templates for expense categorization (Phase 2).
"""

CATEGORIZATION_PROMPT = """
Sana bir fatura analiz sonucu veriyorum. Bu faturayı kategorize et.

**GİRDİ:**
{invoice_data}

**GÖREV:**
Aşağıdaki kategorilendirmeyi yap:

**1. ANA KATEGORİ (primary_category):**
- food_beverage: Gıda ve içecek
- electronics: Elektronik
- clothing_textile: Giyim ve tekstil
- automotive: Otomotiv
- construction_hardware: İnşaat ve hırdavat
- office_supplies: Ofis malzemeleri
- healthcare_pharma: Sağlık ve ilaç
- software_services: Yazılım ve hizmetler
- energy_utilities: Enerji ve faturalar (elektrik, su, doğalgaz)
- hospitality_travel: Konaklama ve seyahat
- other: Diğer

**2. ALT KATEGORİLER (sub_categories):**
Ürün kalemlerine göre spesifik alt kategoriler belirle

**3. HARCAMA TİPİ (expense_type):**
- business_expense: İşletme gideri
- personal_expense: Kişisel gider
- mixed: Karışık

**4. VERGİ KATEGORİSİ (tax_category):**
- deductible: Tamamen indirilebilir
- non_deductible: İndirilemez
- partially_deductible: Kısmen indirilebilir

**5. ÖNCELİK (priority):**
- high: Yüksek (tutar >5000 TRY VEYA vade <7 gün)
- normal: Normal
- low: Düşük

**ÇIKTI:**
{
  "primary_category": "...",
  "sub_categories": ["..."],
  "expense_type": "...",
  "tax_category": "...",
  "priority": "...",
  "reasoning": "Kısa açıklama",
  "confidence_score": 0.95
}

Sadece JSON çıktı ver, ekstra açıklama yapma. JSON başında ve sonunda ```json işaretleri kullanma.
"""

CATEGORY_VALIDATION_PROMPT = """
You are validating expense categorizations for accuracy and consistency.

Review the categorized line items and identify any potential issues:

1. **Categorization Accuracy:**
   - Are items placed in the most appropriate categories?
   - Are there any obvious miscategorizations?
   - Do confidence scores match the categorization quality?

2. **Consistency Checks:**
   - Are similar items categorized consistently?
   - Are there any conflicting categorizations?
   - Do vendor-specific patterns make sense?

3. **Confidence Assessment:**
   - Are confidence scores realistic?
   - Should any low-confidence items be flagged for review?
   - Are high-confidence scores justified?

4. **Recommendations:**
   - Suggest any recategorizations needed
   - Identify items requiring manual review
   - Note any patterns or insights

Return validation results in JSON format:

{
  "validation_summary": {
    "total_items": number,
    "high_confidence_items": number,
    "low_confidence_items": number,
    "potential_issues": number
  },
  "issues_found": [
    {
      "item_number": number,
      "current_category": "category",
      "issue_type": "miscategorization|low_confidence|inconsistency",
      "description": "description of issue",
      "suggested_category": "category or null",
      "suggested_confidence": number
    }
  ],
  "recommendations": [
    "recommendation text"
  ]
}
"""
