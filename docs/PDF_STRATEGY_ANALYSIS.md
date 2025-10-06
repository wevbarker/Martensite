# PDF Ingestion Strategy Analysis for Martensite
## Comprehensive Review of LLM Service PDF Capabilities (2025)

### Executive Summary

Based on current research, **all major LLM providers now support direct PDF upload via API**, but with significantly different capabilities and constraints. This document provides a detailed comparison and optimal strategy for the Martensite adversarial hardening system.

---

## Service-by-Service Analysis

### 1. OpenAI API
**Status**: ✅ **Full PDF Support** (March 2025)

**Capabilities**:
- Direct PDF upload via API without preprocessing
- Supports both text and visual content extraction
- Available on: GPT-4o, GPT-4o-mini, o1-preview, o1-mini

**Limits**:
- **File Size**: 32MB per request across all file inputs
- **Pages**: 100 pages maximum per request
- **General Upload**: 512MB per file (for GPT builders)
- **Token Limit**: 2M tokens per file for text documents

**Integration Method**:
```python
# Direct API upload with vision models
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "Review this application..."},
            {"type": "document", "document_url": {"url": "data:application/pdf;base64,..."}}
        ]}
    ]
)
```

---

### 2. Anthropic Claude API
**Status**: ✅ **Full PDF Support** (Files API)

**Capabilities**:
- Visual and text analysis via page rasterization
- Each page converted to high-resolution image
- Files API with document registration and reuse
- Available on: Claude-3.5-Sonnet, Claude-3-Opus

**Limits**:
- **File Size**: 32MB maximum
- **Pages**: 100 pages maximum per document
- **Cache**: 5-minute lifetime for repeated queries
- **Beta Header Required**: `anthropic-beta: files-api-2025-04-14`

**Integration Method**:
```python
# Two-step process: upload then reference
file = anthropic.files.create(
    file=open("application.pdf", "rb"),
    purpose="document"
)

response = anthropic.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Review this application..."},
            {"type": "document", "source": {"type": "file", "file_id": file.id}}
        ]
    }]
)
```

---

### 3. Google Gemini API
**Status**: ✅ **Excellent PDF Support** (Files API)

**Capabilities**:
- Massive context windows (1M+ tokens)
- Can handle up to 1000 pages per request
- Files API for large document management
- Available on: Gemini-1.5-Pro, Gemini-1.5-Flash, Gemini-2.0-Flash

**Limits**:
- **File Size**: 2GB for Gemini-2.0 models
- **Pages**: 1000 pages per request
- **Direct Upload**: 15MB without Files API
- **Context**: 1M tokens (≈1500 pages of text)

**Integration Method**:
```python
# Upload via Files API
pdf_file = genai.upload_file(
    path="application.pdf",
    display_name="Grant Application"
)

# Reference in prompt
model = genai.GenerativeModel("gemini-1.5-pro")
response = model.generate_content([
    "Review this application...",
    pdf_file
])
```

---

### 4. xAI Grok API
**Status**: ⚠️ **Limited PDF Support**

**Capabilities**:
- Basic PDF upload and analysis
- Multimodal text and image processing
- 128K token context window
- Available on: Grok-4

**Limits**:
- **File Size**: 5MB (planning to increase)
- **Context**: 128,000 tokens
- **Response**: 16,000 tokens maximum

**Integration Method**:
```python
# Direct upload (limited by 5MB)
response = xai.chat.completions.create(
    model="grok-4",
    messages=[{
        "role": "user", 
        "content": [
            {"type": "text", "text": "Review this application..."},
            {"type": "document", "document": pdf_data}
        ]
    }]
)
```

---

## MCP vs Direct API Comparison

### MCP Services Available
Current MCP integrations provide:
- **mcp__gemini__ask_gemini**: Text-only, no file upload
- **mcp__openai__ask_gpt**: Text-only, no file upload  
- **mcp__xai__ask_grok**: Text-only, no file upload

### Key Limitation
**MCP services are text-only** and do not support file uploads or PDF processing. For Martensite's PDF adversarial hardening, **direct API access is mandatory**.

---

## Optimal Strategy for Martensite

### Recommended Architecture

```python
class PDFProcessor:
    def __init__(self):
        self.providers = {
            'openai': self._setup_openai(),
            'anthropic': self._setup_anthropic(), 
            'google': self._setup_google(),
            'xai': self._setup_xai()
        }
    
    async def process_pdf(self, pdf_path: str, models: List[str]):
        # Check file constraints
        file_size = os.path.getsize(pdf_path)
        page_count = self._count_pdf_pages(pdf_path)
        
        # Route to appropriate providers based on constraints
        results = []
        for model in models:
            if self._can_handle(model, file_size, page_count):
                result = await self._process_with_provider(model, pdf_path)
                results.append(result)
        
        return results
```

### Provider Selection Logic

1. **Large Files (>32MB)**: Google Gemini only (2GB limit)
2. **Long Documents (>100 pages)**: Google Gemini only (1000 page limit)
3. **Standard Documents (<32MB, <100 pages)**: All providers
4. **Small Files (<5MB)**: All providers including xAI Grok

### Cost-Optimized Tiers

**Tier 1: Minimal ($0.08)**
- Models: GPT-4o-mini, Claude-3.5-Haiku, Gemini-1.5-Flash
- Method: Direct API upload, 2 calls per model

**Tier 2: Balanced ($0.51)**  
- Models: GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro
- Method: Direct API upload, 3 calls per model

**Tier 3: Premium ($3.11)**
- Models: o1-preview, Claude-3-Opus, Gemini-1.5-Pro, Grok-4
- Method: Direct API upload, 5 calls per model

---

## Implementation Recommendations

### 1. File Validation
```python
def validate_pdf(pdf_path: str) -> Dict[str, Any]:
    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    page_count = count_pdf_pages(pdf_path)
    
    return {
        'size_mb': size_mb,
        'page_count': page_count,
        'openai_compatible': size_mb <= 32 and page_count <= 100,
        'anthropic_compatible': size_mb <= 32 and page_count <= 100,
        'google_compatible': size_mb <= 2048 and page_count <= 1000,
        'xai_compatible': size_mb <= 5
    }
```

### 2. Error Handling
- Implement automatic fallback to text extraction for oversized files
- Retry logic with exponential backoff
- Graceful degradation when providers are unavailable

### 3. Caching Strategy
- Cache uploaded files (especially for Anthropic's 5-minute cache)
- Store file metadata to avoid re-uploading
- Implement local file hashing for duplicate detection

---

## Next Steps for Martensite

1. **Update `application_reviewer.py`** to support direct PDF uploads
2. **Implement provider-specific upload logic** based on file constraints
3. **Add file validation and routing** for optimal provider selection
4. **Create configuration templates** for different document types
5. **Test with actual grant application PDFs** to validate performance

---

*This analysis provides the foundation for implementing robust PDF processing in the Martensite adversarial hardening suite, ensuring maximum compatibility and cost-effectiveness across all major LLM providers.*