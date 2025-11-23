# Clara Chatbot – Healthcare Compliance RAG Assistant

Clara is a Retrieval-Augmented Generation (RAG) chatbot focused on healthcare accreditation and regulatory content, such as:

- DNV Healthcare Standards (NIAHO, CAH, Stroke, Ortho & Spine, etc.)
- CMS Conditions of Participation (42 CFR 482)
- NFPA Life Safety Code (NFPA-101)
- Other related regulatory documents

It combines AWS Bedrock, OpenSearch, and document-aware metadata to answer compliance questions with citations back to the source.

---

## Features

- RAG-based Q&A over DNV, CMS, NFPA and related standards  
- Hybrid search (semantic + keyword) on OpenSearch (or S3 vector store fallback if configured)  
- Program-aware metadata (program, edition, effective_date, domain_code, chapter_code, etc.)  
- Conversation history stored in Valkey/Redis for contextual chat  
- Chunking and ingestion pipeline via AWS Lambda for PDFs  
- FastAPI backend with built-in interactive documentation at `/docs`

---

## High-Level Architecture

Logical view:

```text
User (UI / client)
        │
        ▼
FastAPI app (Clara backend)
  - /chat, /health, /search, etc.
        │
        ├──► AWS Bedrock (LLM for answer generation)
        │
        ├──► AWS Bedrock (Embeddings) ──► OpenSearch Index
        │
        └──► Valkey/Redis (conversation state / caching)

Offline / ingestion path (AWS):
S3 (PDFs) ─► Lambda (chunking + metadata) ─► Bedrock embeddings ─► OpenSearch index
```

This repository focuses primarily on the FastAPI backend and the supporting scripts.  
AWS infrastructure (Lambdas, OpenSearch domain, S3 buckets, etc.) is assumed to already exist.

---

## Tech Stack

- Language: Python 3.11  
- Web Framework: FastAPI + Uvicorn  
- LLM & Embeddings: AWS Bedrock (e.g., Claude for chat, Titan for embeddings)  
- Vector / Search: Amazon OpenSearch Service (hybrid search)  
- State / Cache: Valkey or Redis  
- Message Queue (optional): Amazon SQS  
- Storage: Amazon S3 for source PDFs and/or vector store  
- Infrastructure (outside this repo): AWS Lambda, IAM, CloudWatch, etc.

---

## Prerequisites

Before running the app locally, you will need:

1. Python 3.11 installed  
2. Git installed  
3. AWS credentials configured on your machine with permissions for:
   - `bedrock:InvokeModel`
   - `es:ESHttp*` (or `aoss:*` depending on your OpenSearch flavor)
4. An existing OpenSearch index with the Clara embeddings and metadata loaded  
5. (Recommended) A Valkey/Redis instance available (local Docker or remote)

---

## Installation & Local Development

Run these steps from the repository root (where `requirements.txt` and `main.py` live).

### 1. Clone the repository

```bash
git clone <YOUR_REPO_URL> clara-chatbot
cd clara-chatbot
```

### 2. Create a virtual environment (Python 3.11)

On Windows (PowerShell):

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

You can either:

- Use a `.env` file, or  
- Export variables directly in your shell / PowerShell profile

Common variables (adjust to your setup):

| Variable                          | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `AWS_REGION`                      | AWS region (e.g. `us-east-1`)                                              |
| `BEDROCK_EMBEDDING_MODEL_ID`     | Bedrock embedding model ID (e.g. `amazon.titan-embed-text-v1`)             |
| `BEDROCK_CHAT_MODEL_ID`          | Bedrock chat model ID (e.g. `anthropic.claude-3-...`)                      |
| `OPENSEARCH_ENDPOINT`            | OpenSearch HTTPS endpoint (no trailing `/`)                                |
| `OPENSEARCH_INDEX_NAME`          | Index name used for Clara documents                                        |
| `S3_SOURCE_BUCKET`               | (If used) Bucket containing source PDFs / docs                             |
| `S3_VECTOR_BUCKET`               | (If using S3 vector store fallback)                                        |
| `VALKEY_HOST` / `REDIS_HOST`     | Hostname for Valkey/Redis                                                  |
| `VALKEY_PORT` / `REDIS_PORT`     | Port (usually `6379`)                                                      |
| `SQS_QUEUE_URL`                  | (Optional) SQS queue URL for async tasks                                   |
| `LOG_LEVEL`                      | Logging level (e.g. `INFO`, `DEBUG`)                                       |
| `PROGRAM`                        | Default program filter (e.g. `NIAHO`, `CAH`, `NFPA101`, etc.)              |
| `EDITION`                        | Default edition/year (e.g. `2024`)                                         |
| `EFFECTIVE_DATE`                 | Default effective date for standards                                       |

If the app uses a `.env` loader (e.g. `python-dotenv`), create a `.env` file in the repo root and add these keys there.

### 5. Run the API locally

With your virtual environment activated:

```powershell
uvicorn main:app --reload --port 8000
```

You should see logs indicating the app has started, for example:

- `Uvicorn running on http://127.0.0.1:8000`
- Clara-specific logs such as `Initializing Redis connection pool`, etc.

---

## Using the API

Once the server is running:

- Open interactive documentation:  
  - Swagger UI: <http://127.0.0.1:8000/docs>  
  - ReDoc: <http://127.0.0.1:8000/redoc>

Typical endpoints (may vary slightly by implementation):

- `GET /health` — health check  
- `POST /chat` or `POST /api/chat` — main chat endpoint (prompt + optional filters)  
- `POST /search` or `POST /api/search` — search-only endpoint (no generation)  

Refer to `/docs` for the exact route names and request/response schema.

---

## Ingestion & Indexing (High-Level Overview)

These steps are AWS-side and may not be fully runnable from this repository alone. They are included to explain how new content is ingested into Clara.

1. Upload PDFs (DNV, CMS, NFPA, etc.) into the configured S3 bucket.  
2. Lambda: Chunk and extract metadata  
   - Strips headers/footers  
   - Splits into chunks  
   - Attaches metadata such as program, edition, effective_date, domain_code, chapter_code, etc.  
3. Lambda: Embed and index  
   - Calls the Bedrock embedding model  
   - Writes vectors and metadata into the OpenSearch index (or S3 vector store, if used)  

New documents or editions trigger re-ingestion and update the index used by the chat API.

---

## Testing & Development

If tests are configured in `tests/`:

```bash
pytest
```

If the project uses formatters or linters such as `black` or `ruff`, you can add the corresponding commands here once they are set up.

---

## Troubleshooting

### 1. OpenSearch 403 / `security_exception`

If you see an error similar to:

> `no permissions for [indices:admin/mappings/get]`

then your IAM role or user does not have the necessary OpenSearch permissions.  
Update your policy to allow at least:

- `es:ESHttpGet`
- `es:ESHttpPost`
- `es:ESHttpPut`
- `indices:admin/mappings/get` (or the equivalent for your domain)

### 2. AWS credentials parse errors

If you see:

> `Unable to parse config file: C:\Users\<user>\.aws\credentials`

then your AWS credentials file has invalid formatting (extra spaces, missing `=` or section headers).  
Fix `~/.aws/credentials` (or `C:\Users\<user>\.aws\credentials` on Windows) so it follows this structure:

```ini
[default]
aws_access_key_id=YOUR_KEY
aws_secret_access_key=YOUR_SECRET
```

### 3. Redis/Valkey connection issues

If startup logs show errors initializing Redis/Valkey:

- Verify `VALKEY_HOST` / `VALKEY_PORT` (or `REDIS_HOST` / `REDIS_PORT`)  
- Ensure the container or service is running and reachable  
- Check any required password or authentication environment variables

---

## Project Status & Roadmap

You can adapt or update this section over time. Baseline items:

- Completed: Core RAG chat with Bedrock and OpenSearch  
- Completed: Program-aware metadata filtering for standards  
- In progress: Improved reranking, better hybrid search, evaluation (nDCG, accuracy)  
- Planned: UI front-end, multi-tenant routing, cost-optimized indexing strategies

---

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Make your changes (add tests where appropriate)  
4. Open a pull request with a clear description of the change and how to test it

---

## Contact

For questions about Clara’s architecture, deployment, or extensions, please use the following:

**Maintainer:** Praneet Kulkarni
**Email:** praneet.kulkarni97@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/praneet-kulkarni

---

Note: After integrating this README into your repository, you can refine section titles, environment variable descriptions, or endpoint details to match any future changes in your implementation.
