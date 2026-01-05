"""
RunPod Serverless Worker - INGEST Endpoint
===========================================

Heavy-lifting worker for document ingestion:
- late_chunk: Span-based pooling for paragraphs
- late_chunk_batch: Batch processing multiple paragraphs
- embed_batch: Batch embedding for documents
- embed_tokens: Token-level embedding with offset mapping

Model: BAAI/bge-m3 (1024 dimensions)

IMPORTANT: Logic here MUST match late_embedding_service.py LOCAL implementation exactly!
"""
import runpod
import torch
import numpy as np
import re
from typing import List, Dict, Any, Tuple

# =============================================================================
# GLOBAL MODEL (loaded once on cold start)
# =============================================================================
MODEL = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Starting INGEST worker on {DEVICE}...")


def load_model():
    """Load embedding model (called once on cold start)."""
    global MODEL, TOKENIZER
    
    if MODEL is None:
        from sentence_transformers import SentenceTransformer
        
        print("🔄 Loading BAAI/bge-m3...")
        MODEL = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
        
        if DEVICE == "cuda":
            MODEL.half()
        
        TOKENIZER = MODEL.tokenizer
        print("✅ Model loaded!")
    
    return MODEL, TOKENIZER


# =============================================================================
# SPAN-BASED POOLING HELPERS (EXACT COPY FROM late_embedding_service.py)
# =============================================================================

def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, strip."""
    return re.sub(r'\s+', ' ', text).strip()


def _find_sentence_char_spans_with_indices(
    paragraph_text: str, 
    sentence_texts: List[str]
) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Find character positions of each sentence in paragraph.
    HANDLES:
    - Duplicate sentences (occurrence tracking)
    - Whitespace mismatches (normalized fuzzy search)
    
    Returns:
        List of (original_index, (char_start, char_end)) - preserves original order info
    """
    results = []  # (original_idx, (char_start, char_end))
    
    # Track how many times we've seen each sentence text
    # Key: normalized sentence text, Value: next search_start position
    seen_cursors: Dict[str, int] = {}
    
    # Pre-normalize paragraph for fallback fuzzy search
    para_normalized = _normalize_whitespace(paragraph_text)
    
    for orig_idx, sent_text in enumerate(sentence_texts):
        if not sent_text.strip():
            results.append((orig_idx, (0, 0)))
            continue
        
        sent_key = sent_text.strip()
        sent_normalized = _normalize_whitespace(sent_text)
        
        # Get the cursor for this sentence (where to start searching)
        search_start = seen_cursors.get(sent_normalized, 0)
        
        # =====================================================================
        # SEARCH STRATEGY (in order of preference):
        # 1. Exact match
        # 2. Stripped match  
        # 3. Normalized whitespace match (fuzzy)
        # =====================================================================
        
        idx = paragraph_text.find(sent_text, search_start)
        found_len = len(sent_text)
        
        if idx == -1:
            # Try stripped version
            idx = paragraph_text.find(sent_key, search_start)
            found_len = len(sent_key)
            
        if idx == -1:
            # Try from beginning
            idx = paragraph_text.find(sent_key, 0)
            found_len = len(sent_key)
        
        if idx == -1:
            # =====================================================================
            # FUZZY SEARCH: Normalize whitespace and find approximate position
            # =====================================================================
            norm_idx = para_normalized.find(sent_normalized, 0)
            
            if norm_idx != -1:
                # Found in normalized text - estimate position in original
                # Use ratio-based approximation
                ratio = norm_idx / max(len(para_normalized), 1)
                approx_start = int(ratio * len(paragraph_text))
                
                # Search nearby in original text (within 50 chars)
                search_window_start = max(0, approx_start - 50)
                search_window_end = min(len(paragraph_text), approx_start + len(sent_normalized) + 50)
                
                # Try to find first few words in the window
                first_words = sent_normalized.split()[:3]
                first_words_pattern = ' '.join(first_words)
                
                window_text = paragraph_text[search_window_start:search_window_end]
                local_idx = window_text.find(first_words_pattern)
                
                if local_idx != -1:
                    idx = search_window_start + local_idx
                    # Estimate end based on sentence length ratio
                    found_len = int(len(sent_text) * 1.2)  # Allow some buffer
                else:
                    # Use normalized position approximation
                    idx = approx_start
                    found_len = len(sent_normalized)
            else:
                # Really not found - use fallback
                print(f"⚠️ Sentence not found: '{sent_text[:40]}...'")
                results.append((orig_idx, (0, len(sent_text))))
                continue
        
        char_start = idx
        char_end = min(idx + found_len, len(paragraph_text))
        results.append((orig_idx, (char_start, char_end)))
        
        # Update cursor for this sentence text to AFTER current match
        seen_cursors[sent_normalized] = char_end
    
    return results


def find_sentence_char_spans(paragraph_text: str, sentence_texts: List[str]) -> List[Tuple[int, int]]:
    """
    Find character start/end positions of each sentence in the paragraph.
    SORTS by actual position to handle out-of-order DB results!
    
    Returns:
        List of (char_start, char_end) tuples, ordered by position in paragraph
    """
    # Get spans with original indices
    indexed_spans = _find_sentence_char_spans_with_indices(paragraph_text, sentence_texts)
    
    # Sort by character position (not original order!)
    indexed_spans.sort(key=lambda x: x[1][0])
    
    # Extract just the spans (now correctly ordered)
    return [span for _, span in indexed_spans]


def map_char_spans_to_token_spans(
    char_spans: List[Tuple[int, int]],
    token_offsets: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Map character spans to token spans using offset mapping.
    
    Args:
        char_spans: List of (char_start, char_end) for each sentence
        token_offsets: List of (char_start, char_end) for each token
    
    Returns:
        List of (token_start, token_end) for each sentence
    """
    token_spans = []
    
    for char_start, char_end in char_spans:
        if char_start == char_end:
            # Zero-width span
            token_spans.append((0, 0))
            continue
        
        # Find first token that overlaps with char_start
        token_start = None
        token_end = None
        
        for tok_idx, (tok_char_start, tok_char_end) in enumerate(token_offsets):
            # Skip special tokens (they have (0,0) offset)
            if tok_char_start == 0 and tok_char_end == 0 and tok_idx > 0:
                continue
            
            # Check if this token overlaps with our sentence
            if tok_char_end > char_start and tok_char_start < char_end:
                if token_start is None:
                    token_start = tok_idx
                token_end = tok_idx + 1  # Exclusive end
        
        if token_start is None:
            # No tokens found for this sentence
            token_spans.append((0, 0))
        else:
            token_spans.append((token_start, token_end))
    
    return token_spans


# =============================================================================
# HANDLER FUNCTIONS
# =============================================================================

def embed_batch(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch embedding for documents.
    
    NOTE: BGE-M3 does NOT need query/passage prefix (unlike E5 models).
    Same behavior as LOCAL to ensure consistency.
    
    Input:
        texts: List[str]
        is_query: bool (ignored for BGE-M3, kept for API compatibility)
    
    Output:
        embeddings: List[List[float]]
    """
    model, _ = load_model()
    
    texts = input_data.get("texts", [])
    # is_query param ignored for BGE-M3 (no prefix needed)
    
    if not texts:
        return {"error": "No texts provided", "embeddings": []}
    
    with torch.inference_mode():
        # BGE-M3 does NOT need prefix (same as LOCAL behavior)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        return {"embeddings": embeddings.tolist()}


def embed_tokens(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Token-level embedding with offset mapping (for late chunking).
    
    Input:
        text: str
    
    Output:
        token_embeddings: List[List[float]]
        offset_mapping: List[List[int]]
        pooled_embedding: List[float]
    """
    model, tokenizer = load_model()
    
    text = input_data.get("text", "")
    if not text:
        return {"error": "No text provided"}
    
    with torch.inference_mode():
        # Tokenize with offset mapping
        tokenized = tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        )
        offset_mapping = tokenized["offset_mapping"][0].tolist()
        
        # Get token embeddings
        token_embeddings = model.encode(
            text,
            output_value="token_embeddings",
            convert_to_numpy=False,
            show_progress_bar=False
        )
        
        if torch.is_tensor(token_embeddings):
            token_embeddings = token_embeddings.cpu().numpy()
        
        if len(token_embeddings.shape) == 3:
            token_embeddings = token_embeddings[0]
        
        # Also get pooled embedding
        pooled = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return {
            "token_embeddings": token_embeddings.tolist(),
            "offset_mapping": offset_mapping,
            "pooled_embedding": pooled.tolist()
        }


def late_chunk(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Late chunk: Span-based pooling for a single paragraph.
    
    EXACT SAME LOGIC AS late_embedding_service.py _late_chunk_local()
    
    Input:
        paragraph_text: str
        sentence_texts: List[str]
        prefix_len: int (optional, for section context offset)
    
    Output:
        vectors: List[List[float]]
    """
    model, tokenizer = load_model()
    
    paragraph_text = input_data.get("paragraph_text", "")
    sentence_texts = input_data.get("sentence_texts", [])
    prefix_len = input_data.get("prefix_len", 0)
    
    if not paragraph_text or not sentence_texts:
        return {"vectors": []}
    
    with torch.inference_mode():
        # Get token embeddings + offset mapping
        tokenized = tokenizer(
            paragraph_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        )
        offset_mapping = tokenized["offset_mapping"][0].tolist()
        
        token_embeddings = model.encode(
            paragraph_text,
            output_value="token_embeddings",
            convert_to_numpy=False,
            show_progress_bar=False
        )
        
        if torch.is_tensor(token_embeddings):
            token_embeddings = token_embeddings.cpu().numpy()
        
        if len(token_embeddings.shape) == 3:
            token_embeddings = token_embeddings[0]
        
        num_tokens = len(token_embeddings)
        
        # Get pooled embedding for fallback
        pooled_embedding = model.encode(
            paragraph_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        if num_tokens == 0:
            # Fallback: return pooled for all sentences
            return {"vectors": [pooled_embedding.tolist()] * len(sentence_texts)}
        
        # Find char spans (accounting for prefix)
        # NOTE: Search in paragraph AFTER prefix (offset by prefix_len)
        para_without_prefix = paragraph_text[prefix_len:] if prefix_len > 0 else paragraph_text
        char_spans = find_sentence_char_spans(para_without_prefix, sentence_texts)
        
        # Adjust char spans by prefix length
        if prefix_len > 0:
            char_spans = [(s + prefix_len, e + prefix_len) for s, e in char_spans]
        
        # Map to token spans
        token_spans = map_char_spans_to_token_spans(char_spans, offset_mapping)
        
        # Pool tokens for each sentence (EXACT SAME LOGIC AS LOCAL)
        vectors = []
        fallback_vec = pooled_embedding.tolist()
        
        for i, (token_start, token_end) in enumerate(token_spans):
            if token_start >= token_end or token_start >= num_tokens:
                # No valid tokens - use fallback (same as local)
                if vectors:
                    vectors.append(vectors[-1])  # Use previous vector
                else:
                    vectors.append(fallback_vec)  # Use pooled
                continue
            
            # Clamp to valid range
            token_end = min(token_end, num_tokens)
            
            # Extract and mean pool ONLY this sentence's tokens
            chunk_tokens = token_embeddings[token_start:token_end]
            chunk_vec = np.mean(chunk_tokens, axis=0)
            
            # Normalize
            norm = np.linalg.norm(chunk_vec)
            if norm > 0:
                chunk_vec = chunk_vec / norm
            
            vectors.append(chunk_vec.tolist())
        
        return {"vectors": vectors}


def late_chunk_batch(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch late chunk: Process multiple paragraphs.
    
    Input:
        paragraphs: List[{
            "paragraph_text": str,
            "sentence_texts": List[str],
            "section_context": str (optional)
        }]
    
    Output:
        results: List[{"vectors": List[List[float]]}]
    """
    paragraphs = input_data.get("paragraphs", [])
    
    if not paragraphs:
        return {"results": []}
    
    results = []
    for para in paragraphs:
        para_text = para.get("paragraph_text", "")
        sentence_texts = para.get("sentence_texts", [])
        section_context = para.get("section_context", "")
        
        # Add section context prefix (SAME AS LOCAL)
        prefix = ""
        if section_context and section_context.strip():
            prefix = f"{section_context}: "
            para_text = prefix + para_text
        
        # Process this paragraph
        result = late_chunk({
            "paragraph_text": para_text,
            "sentence_texts": sentence_texts,
            "prefix_len": len(prefix)
        })
        
        results.append(result)
    
    return {"results": results}


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler for INGEST endpoint.
    
    Actions:
    - embed_batch: Batch document embedding
    - embed_tokens: Token-level embedding
    - late_chunk: Single paragraph late chunking
    - late_chunk_batch: Batch late chunking
    - health_check: Health check
    """
    try:
        input_data = event.get("input", {})
        action = input_data.get("action", "")
        
        if action == "embed_batch":
            return embed_batch(input_data)
        elif action == "embed_tokens":
            return embed_tokens(input_data)
        elif action == "late_chunk":
            return late_chunk(input_data)
        elif action == "late_chunk_batch":
            return late_chunk_batch(input_data)
        elif action == "health_check":
            return {"status": "ok", "endpoint": "ingest"}
        else:
            return {"error": f"Unknown action: {action}"}
    
    except Exception as e:
        return {"error": str(e)}


# Start RunPod serverless
if __name__ == "__main__":
    print("🎉 INGEST worker ready!")
    runpod.serverless.start({"handler": handler})
