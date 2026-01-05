# RUNPOD-INGEST: PyTorch CUDA-based Embedding Worker
# ==================================================
# Heavy lifting for document ingestion:
# - late_chunk, late_chunk_batch
# - embed_batch, embed_tokens

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model at build time (faster cold start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Copy handler
COPY handler.py .

# RunPod serverless entry point
CMD ["python", "-u", "handler.py"]
