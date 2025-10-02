"""
Prompt templates for final summary generation (Phase 4).
"""

SUMMARY_PROMPT = """
Fatura analiz sonuçlarını özetle. Gerçek fatura verilerini kullan. SADECE JSON döndür, başka hiçbir şey yazma.

Örnek format:
{
  "invoice_summary": {
    "vendor": "Gerçek satıcı adı",
    "date": "Gerçek tarih",
    "total": gerçek_tutar,
    "currency": "Gerçek para birimi",
    "category": "Uygun kategori",
    "status": "ready_to_process veya needs_review"
  },
  "key_insights": [
    "Gerçek duruma göre önemli bilgiler"
  ],
  "action_required": {
    "immediate": ["Acil yapılacaklar"],
    "review": ["İncelenmesi gerekenler"],
    "none": ["Sorun yoksa boş"]
  },
  "processing_decision": {
    "method": "standard veya fallback_required",
    "confidence": gerçek_güven_skoru,
    "reason": "Gerçek durum açıklaması"
  }
}
"""

INSIGHTS_GENERATION_PROMPT = """
You are generating advanced business insights from invoice analysis data.

Analyze patterns, trends, and anomalies to provide strategic insights:

1. **Expense Pattern Analysis:**
   - Spending patterns by category
   - Vendor relationship analysis
   - Cost trend identification
   - Budget impact assessment

2. **Anomaly Detection:**
   - Unusual amounts or patterns
   - Unexpected vendor charges
   - Category misalignments
   - Timing anomalies

3. **Optimization Opportunities:**
   - Cost reduction possibilities
   - Process improvement suggestions
   - Vendor consolidation opportunities
   - Category optimization

4. **Compliance and Risk Insights:**
   - Regulatory compliance considerations
   - Audit trail completeness
   - Risk mitigation suggestions
   - Documentation quality

Return insights in structured format:

{
  "strategic_insights": [
    {
      "category": "cost_optimization|compliance|process|vendor",
      "insight": "Specific insight description",
      "impact": "high|medium|low",
      "actionable": true,
      "recommendation": "Specific action to take"
    }
  ],
  "anomalies_detected": [
    {
      "type": "amount|vendor|category|timing",
      "description": "Anomaly description",
      "severity": "high|medium|low",
      "investigation_needed": true
    }
  ],
  "optimization_opportunities": [
    {
      "area": "Area of optimization",
      "potential_savings": "Estimated impact",
      "implementation_effort": "low|medium|high",
      "priority": "high|medium|low"
    }
  ]
}
"""
