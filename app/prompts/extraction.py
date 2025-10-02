"""
Prompt templates for invoice data extraction (Phase 1).
"""

EXTRACTION_PROMPT = """
Sen bir fatura analiz uzmanısın. Sana gönderilen fatura görselini analiz et ve aşağıdaki bilgileri JSON formatında çıkar:

**ZORUNLU ALANLAR:**
- invoice_number: Fatura numarası
- invoice_date: Fatura tarihi (YYYY-MM-DD formatında)
- due_date: Son ödeme tarihi (varsa, YYYY-MM-DD formatında)
- vendor_name: Satıcı/Tedarikçi adı
- vendor_address: Satıcı adresi
- vendor_tax_number: Satıcı vergi/TCKN numarası
- customer_name: Müşteri/Alıcı adı
- customer_address: Müşteri adresi
- customer_tax_number: Müşteri vergi/TCKN numarası
- currency: Para birimi (TRY, USD, EUR vb.)
- subtotal: Ara toplam (KDV hariç)
- tax_rate: KDV oranı (%)
- tax_amount: KDV tutarı
- total_amount: Genel toplam

**ÜRÜN/HİZMET KALEMLERİ:**
Her kalem için:
- line_items: [
  {
    "description": "Ürün/hizmet açıklaması",
    "quantity": Miktar,
    "unit": "Birim (adet, kg, saat vb.)",
    "unit_price": Birim fiyat,
    "line_total": Satır toplamı
  }
]

**GÜVEN SKORU:**
- confidence_score: Her alan için 0-1 arası güven skoru hesapla
- overall_confidence: Genel güven skoru (0-1 arası)
- missing_fields: Bulunamayan alanların listesi
- ocr_quality: Görsel kalitesi değerlendirmesi (0-1 arası):
  * 0.9-1.0: Mükemmel kalite, net, okunaklı
  * 0.7-0.9: İyi kalite, bazı küçük sorunlar
  * 0.5-0.7: Orta kalite, bazı alanlar zor okunuyor
  * 0-0.5: Düşük kalite, bulanık, el yazısı, bozuk

**ÖNEMLİ:**
- Eğer bir alan bulunamazsa, null değer kullan
- Tüm sayısal değerleri float olarak ver
- Tarihleri mutlaka YYYY-MM-DD formatında ver
- Türkçe karakterleri koru
- El yazısı veya çok bulanık görsellerde ocr_quality'yi düşük ver

Çıktıyı sadece JSON olarak ver, başka açıklama ekleme. JSON başında ve sonunda ```json işaretleri kullanma, sadece düz JSON ver.
"""

STRUCTURED_EXTRACTION_PROMPT = """
You are an expert at converting raw invoice text into structured JSON data. 

Convert the following extracted invoice text into a properly structured JSON object.

REQUIRED JSON STRUCTURE:
{
  "invoice_number": "string or null",
  "invoice_date": "YYYY-MM-DD or null",
  "invoice_type": "purchase|sales|service|utility|rent|other or null",
  "vendor": {
    "name": "string or null",
    "address": "string or null",
    "phone": "string or null",
    "email": "string or null",
    "tax_id": "string or null",
    "website": "string or null"
  },
  "currency": "USD|EUR|GBP|TRY|OTHER or null",
  "subtotal": "number or null",
  "tax_info": {
    "tax_rate": "decimal (0.0-1.0) or null",
    "tax_amount": "number or null",
    "tax_type": "string or null"
  },
  "total_amount": "number or null",
  "line_items": [
    {
      "description": "string",
      "quantity": "number or null",
      "unit_price": "number or null",
      "total_price": "number or null"
    }
  ],
  "payment_info": {
    "payment_method": "string or null",
    "payment_terms": "string or null",
    "due_date": "YYYY-MM-DD or null",
    "payment_status": "string or null"
  },
  "notes": "string or null",
  "purchase_order_number": "string or null"
}

CONVERSION RULES:

1. **Data Types:**
   - Use null for missing values, not empty strings
   - Convert all numeric values to proper numbers (not strings)
   - Ensure dates are in YYYY-MM-DD format
   - Tax rates should be decimals (0.18 for 18%)

2. **Currency Handling:**
   - Remove currency symbols from numeric values
   - Store currency type separately in currency field
   - Handle different decimal separators (. vs ,)

3. **Date Processing:**
   - Convert various date formats to YYYY-MM-DD
   - Handle formats like MM/DD/YYYY, DD.MM.YYYY, etc.
   - Use null if date cannot be determined

4. **Line Items:**
   - Create separate objects for each line item
   - Extract description, quantity, unit price, and total
   - Handle cases where some fields are missing

5. **Validation:**
   - Ensure numeric values are reasonable
   - Check that line item totals make sense
   - Verify date validity

Return only the JSON object, no additional text or explanation.
"""
