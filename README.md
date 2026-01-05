# RUNPOD-INGEST: PyTorch CUDA Embedding Worker

Heavy-lifting worker for document ingestion using PyTorch + CUDA.

## Features

- `late_chunk` - Span-based pooling for single paragraph
- `late_chunk_batch` - Batch processing multiple paragraphs
- `embed_batch` - Batch document embedding
- `embed_tokens` - Token-level embedding with offset mapping

## Model

- **Model**: BAAI/bge-m3
- **Dimensions**: 1024
- **Runtime**: PyTorch + CUDA (GPU)

## Setup GitHub Actions

### 1. Create Docker Hub Access Token

1. Go to [Docker Hub](https://hub.docker.com/) → Account Settings → Security
2. Create new Access Token with Read/Write permissions
3. Copy the token

### 2. Add GitHub Secrets

Go to your repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret Name | Value |
|-------------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Your Docker Hub access token |

### 3. Push to Main

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

GitHub Actions will automatically build and push the Docker image.

## Create RunPod Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Name**: `ingest-embedding`
   - **Docker Image**: `yourusername/runpod-ingest:latest`
   - **GPU**: RTX 3090 or better (24GB VRAM recommended)
   - **Min Workers**: 0 (scale to zero when idle)
   - **Max Workers**: 2-5 (based on traffic)
   - **Idle Timeout**: 5 seconds
   - **Execution Timeout**: 300 seconds

4. Copy the Endpoint ID for your app config

## API Actions

### `late_chunk`
```json
{
  "input": {
    "action": "late_chunk",
    "paragraph_text": "Full paragraph text...",
    "sentence_texts": ["Sentence 1.", "Sentence 2."],
    "prefix_len": 0
  }
}
```

### `late_chunk_batch`
```json
{
  "input": {
    "action": "late_chunk_batch",
    "paragraphs": [
      {
        "paragraph_text": "...",
        "sentence_texts": ["..."],
        "section_context": "Section Title"
      }
    ]
  }
}
```

### `embed_batch`
```json
{
  "input": {
    "action": "embed_batch",
    "texts": ["Text 1", "Text 2"],
    "is_query": false
  }
}
```

### `health_check`
```json
{
  "input": {
    "action": "health_check"
  }
}
```

## Local Testing

```bash
# Build
docker build -t runpod-ingest .

# Run (requires NVIDIA GPU)
docker run --gpus all -p 8000:8000 runpod-ingest
```

## Environment Variables

Set these in your main app:

```bash
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_INGEST_ENDPOINT_ID=your_endpoint_id
BACKEND_EMBED_INGEST=runpod
BACKEND_LATE_EMBED=runpod
```
