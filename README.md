# Reranker Service for Open WebUI

## Overview

This repository packages a lightweight, GPU-aware reranking microservice that you can deploy next to Open WebUI on bare Docker or in a k3s cluster. The service exposes a single `/rerank` endpoint that mirrors what Open WebUI's **Reranking Engine** option expects: provide an API base URL plus the name of the reranker model you want to use, and the service will score and sort documents accordingly.

## Architecture

- **Framework**: FastAPI + Uvicorn for a small, async HTTP surface that matches Open WebUI's request cadence.
- **Model backend**: Hugging Face Transformers cross-encoders (e.g., `BAAI/bge-reranker-v2-m3`, `jinaai/jina-reranker-v2-base`), executed with PyTorch and automatically pinned to CUDA if available. Precision defaults to `float16` on GPUs and `float32` on CPU.
- **Model manager**: Lazy, thread-safe cache so multiple Open WebUI tenants can hot-swap models without restarting the pod. Only models declared in `ALLOWED_MODELS` (via env or config) can be loaded.
- **Request schema**: Accepts Open WebUI's `query`, `documents`, optional `model`, `top_n`, and `return_documents` flags. Documents can be simple strings or `{ "id": str, "text": str, ... }` dictionaries; the service normalises them internally.
- **Response schema**: Returns ordered hits, each containing the document payload, score, and indices, so Open WebUI can feed the reranked list back into hybrid search.

## API Contract (v1)

`POST /rerank`

```jsonc
{
  "query": "What are vector databases?",
  "documents": [
    "Chroma is an open-source embedding database...",
    { "id": "doc-2", "text": "Redis can act as a vector database", "meta": {"source": "blog"} }
  ],
  "model": "BAAI/bge-reranker-v2-m3",   // optional; defaults to DEFAULT_MODEL
  "top_n": 5,                             // optional; defaults to length of documents
  "return_documents": true                // optional; include the raw document payloads
}
```

Response

```jsonc
{
  "model": "BAAI/bge-reranker-v2-m3",
  "latency_ms": 38.7,
  "results": [
    { "index": 0, "score": 0.92, "document": "Chroma is an open-source embedding database..." },
    { "index": 1, "score": 0.74, "document": { "id": "doc-2", "text": "Redis can act as a vector database", "meta": {"source": "blog"} } }
  ]
}
```

`GET /healthz` responds with build info; `GET /models` lists the cached/allowed model names.

---

## Configuration

The service reads environment variables with the `RERANKER_` prefix. Key toggles:

| Variable | Description | Default |
| --- | --- | --- |
| `RERANKER_ALLOWED_MODELS` | Comma-separated list of Hugging Face model IDs permitted for loading. | `BAAI/bge-reranker-v2-m3,...` |
| `RERANKER_DEFAULT_MODEL` | Model to use when a client does not pass `model`. Must be in the allowed list. | `BAAI/bge-reranker-v2-m3` |
| `RERANKER_MAX_DOCUMENTS` | Hard limit per request to prevent OOM. | `512` |
| `RERANKER_MAX_TOP_N` | Clips `top_n` requests. | `100` |
| `RERANKER_BATCH_SIZE` | Query-document pairs processed per forward pass. Tune for your GPU memory. | `32` |
| `RERANKER_MAX_SEQUENCE_LENGTH` | Tokenizer `max_length`. Lower values save VRAM. | `512` |
| `RERANKER_WARM_MODELS` | Comma-separated models to preload on boot. | unset |
| `RERANKER_DEVICE` / `RERANKER_TORCH_DTYPE` | Force device (e.g., `cpu`) or dtype (`float32`, `bfloat16`). Leave unset for auto-detect. | auto |

`TRANSFORMERS_CACHE` defaults to `/models` inside the container; mount persistent storage if you want to avoid re-downloading weights when pods restart.

## Local run (bare metal)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# For CUDA hosts, pull GPU wheels:
pip install torch==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Sanity check:

```bash
curl -X POST http://localhost:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{
        "query": "best vector store",
        "documents": ["Chroma is an embedding database", "S3 is object storage"],
        "top_n": 1
      }'
```

## Container build & GPU run

```bash
docker build -t ghcr.io/you/reranker:latest .
# NVIDIA Container Toolkit required for --gpus flag
docker run --rm --gpus all -p 8080:8080 \
  -e RERANKER_ALLOWED_MODELS="BAAI/bge-reranker-v2-m3,jinaai/jina-reranker-v2-base" \
  -e RERANKER_WARM_MODELS="BAAI/bge-reranker-v2-m3" \
  ghcr.io/you/reranker:latest
```

## Deploy to k3s / containerd

1. Ensure the NVIDIA device plugin and runtime class are installed on the GPU nodes (`RuntimeClass` often named `nvidia`).
2. Push the built image to a registry the cluster can reach (e.g., GHCR, Docker Hub).
3. Apply a manifest similar to the example below (adjust namespace, storage, and tolerations as needed):

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: openwebui
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reranker
  namespace: openwebui
spec:
  replicas: 1
  selector:
    matchLabels:
      app: reranker
  template:
    metadata:
      labels:
        app: reranker
    spec:
      runtimeClassName: nvidia   # matches your GPU RuntimeClass
      containers:
        - name: reranker
          image: ghcr.io/you/reranker:latest
          imagePullPolicy: IfNotPresent
          resources:
            limits:
              nvidia.com/gpu: 1
              cpu: "2"
              memory: 6Gi
          ports:
            - containerPort: 8080
          env:
            - name: RERANKER_ALLOWED_MODELS
              value: "BAAI/bge-reranker-v2-m3,jinaai/jina-reranker-v2-base"
            - name: RERANKER_DEFAULT_MODEL
              value: "BAAI/bge-reranker-v2-m3"
            - name: RERANKER_WARM_MODELS
              value: "BAAI/bge-reranker-v2-m3"
          volumeMounts:
            - name: model-cache
              mountPath: /models
      volumes:
        - name: model-cache
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: reranker
  namespace: openwebui
spec:
  selector:
    app: reranker
  ports:
    - port: 8080
      targetPort: 8080
```

If you rely on containerd without the NVIDIA runtime, set `spec.template.spec.nodeSelector` or tolerations for GPU nodes and swap `runtimeClassName` with the appropriate configuration.

## Wiring into Open WebUI

Inside Open WebUI's **Settings → RAG → Hybrid Search** section (screenshot in the prompt):

1. Toggle **Hybrid Search** on.
2. Set **Reranking Engine → API Base URL** to `http://reranker.openwebui.svc.cluster.local:8080/rerank` (k3s) or `http://localhost:8080/rerank` (Docker).
3. Leave **API Key** blank unless you front the service with an auth proxy.
4. Enter one of the allowed model names, e.g., `BAAI/bge-reranker-v2-m3`, into **Reranking Model**. The service enforces this value against `RERANKER_ALLOWED_MODELS`, so switching models is instantaneous.
5. Keep **Top K**/`Top K Reranker` settings aligned with the `RERANKER_MAX_TOP_N` limit.

You can verify connectivity from Open WebUI by watching the reranker pod logs; every hybrid search call will POST to `/rerank` with the query and retrieved candidates.

## Operational tips

- Pin the Hugging Face cache (`/models`) to a persistent volume (PVC, hostPath, or RWX storage) so pod restarts avoid cold downloads.
- Use `RERANKER_WARM_MODELS` to pre-load heavy rerankers during low-traffic windows; otherwise, the first Open WebUI request pays the download/initialization cost.
- For multi-model setups, set `resources.requests.memory` so the pod isn't rescheduled mid-inference.
- Expose metrics by scraping latency from the JSON response or add Prometheus middlewares as needed.

