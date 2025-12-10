# Indian Law RAG Pipeline (Pinecone Version)

> A crisis-focused legal Q&A system for Indian law using Pinecone vector database and Chimera LLM.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Pinecone](https://img.shields.io/badge/Pinecone-vector%20DB-success)](https://www.pinecone.io/)

## What This Is

This is the **Pinecone-powered variant** of the Indian Law RAG pipeline. Unlike the lightweight NumPy version designed for Railway's free tier, this implementation uses Pinecone's managed vector database for semantic search over Indian legal statutes.

It answers questions about Indian law by retrieving relevant sections from a comprehensive corpus and generating practical, crisis-focused guidance for people in urgent legal situations.

**Key Difference**: This version connects to a live Pinecone index at runtime instead of loading precomputed embeddings from local files. This makes it more flexible for updates but requires a Pinecone account.

---

## Why This Version Exists

### The Hosting Problem

When I first built this system, I ran into Railway's deployment limitations:
- Docker images over 4GB fail to build
- PyTorch + sentence-transformers + FastAPI = 5-6GB easily
- Free-tier hosting can't handle that footprint

So I created **two versions**:

#### Version 1: NumPy (Lightweight)
- Precomputes embeddings offline
- Ships `embeddings.npy` + `docs.json` files (~60MB)
- Runs on Railway's free tier
- Can't update the corpus without regenerating embeddings
- **Best for**: Demos, static deployments, resource-constrained hosting

#### Version 2: Pinecone (This Version)
- Connects to Pinecone's managed vector database
- Uses `sentence-transformers` at runtime for query encoding
- Requires more resources but allows real-time corpus updates
- **Best for**: Development, production with proper infrastructure, frequently updated legal corpus

This README covers **Version 2**—the one that talks directly to Pinecone.

---

## Architecture Overview

### High-Level Flow

```
┌─────────────┐
│   User      │
│  Question   │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────────────┐
│  1. LLM Router (Chimera)                    │
│     • Classifies question into topics       │
│     • Maps topics to relevant Acts          │
│     • Builds metadata filter                │
└──────┬──────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────┐
│  2. Query Encoding (E5-Large-v2)            │
│     • Encode question with SentenceTransf.  │
│     • Generate 1024-dim embedding vector    │
└──────┬──────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────┐
│  3. Pinecone Vector Search                  │
│     • Query index with embedding + filter   │
│     • Retrieve top-k matching chunks        │
│     • Return with metadata + scores         │
└──────┬──────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────┐
│  4. RAG Prompt Construction                 │
│     • Format retrieved sections             │
│     • Add crisis-format instructions        │
│     • Include original question             │
└──────┬──────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────┐
│  5. Generation (Chimera via OpenRouter)     │
│     • Generate crisis-style guidance        │
│     • Follow strict formatting rules        │
└──────┬──────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────┐
│  Response: answer + sources + metadata      │
└─────────────────────────────────────────────┘
```

### Detailed Pipeline Flow

```
USER QUERY: "wife filed 498A case, what should I do?"
│
├─► STEP 1: TOPIC CLASSIFICATION
│   ├─ Input: Raw question text
│   ├─ Process: Call Chimera with routing prompt
│   │   System: "You are an Indian law routing assistant..."
│   │   Output format: JSON with topics, acts, notes
│   ├─ LLM Response: {
│   │     "topics": ["family_dv", "criminal_ipc"],
│   │     "acts": ["IPC Section 498A", "Domestic Violence Act", "CrPC"],
│   │     "notes": "Query about false dowry harassment case"
│   │   }
│   └─ Fallback: If JSON parse fails, use keyword matching
│       Time: ~2.5s
│
├─► STEP 2: BUILD METADATA FILTER
│   ├─ Input: Topics from router
│   ├─ Process: Map topics → Pinecone source names
│   │   "family_dv" → {"source": {"$in": ["dv_act_2005", "constitution_of_india_2024"]}}
│   │   "criminal_ipc" → {"source": {"$in": ["bns_etc_2019", "evidence_act_1872", "crpc_1973"]}}
│   ├─ Combine: {"$or": [source_clause_1, source_clause_2, qa_explanation_clause]}
│   └─ Output: Pinecone metadata filter
│       Time: <1ms
│
├─► STEP 3: ENCODE QUERY
│   ├─ Model: intfloat/e5-large-v2 (SentenceTransformer)
│   ├─ Input: "query: wife filed 498A case, what should I do?"
│   │   (Note: "query: " prefix is required by E5 model)
│   ├─ Process: Forward pass through transformer
│   ├─ Output: 1024-dimensional embedding vector
│   │   [0.023, -0.145, 0.267, ..., 0.089]
│   └─ Time: ~200ms (CPU) / ~50ms (GPU)
│
├─► STEP 4: PINECONE VECTOR SEARCH
│   ├─ Index: indian-law-docs (hosted on Pinecone)
│   ├─ Query params:
│   │   - vector: [1024-dim embedding]
│   │   - top_k: 12
│   │   - filter: {metadata filter from step 2}
│   │   - include_metadata: true
│   ├─ Process: Pinecone's managed similarity search
│   │   • Compares against ~2,800 indexed chunks
│   │   • Applies metadata filter
│   │   • Returns top matches by cosine similarity
│   ├─ Response: [
│   │     {
│   │       "id": "chunk_1247",
│   │       "score": 0.89,
│   │       "metadata": {
│   │         "source": "bns_etc_2019",
│   │         "kind": "section",
│   │         "text": "Section 498A: Whoever, being the husband..."
│   │       }
│   │     },
│   │     ... 11 more results
│   │   ]
│   └─ Time: ~150ms (network + search)
│
├─► STEP 5: CONTEXT PREPARATION
│   ├─ Input: 12 Pinecone matches
│   ├─ Format each chunk:
│   │   [Source: bns_etc_2019 | Kind: section | Score: 0.890]
│   │   Section 498A: Whoever, being the husband or relative of the
│   │   husband of a woman, subjects such woman to cruelty shall be...
│   │   [text truncated to 1500 chars per chunk]
│   ├─ Concatenate with separators: "---"
│   └─ Total context: ~5,000 tokens
│       Time: ~5ms
│
├─► STEP 6: RAG PROMPT CONSTRUCTION
│   ├─ Template: Crisis-focused format with strict instructions
│   ├─ Components:
│   │   1. System role: "You are a senior Indian lawyer with 25 years experience..."
│   │   2. Format example: Shows exact structure expected
│   │   3. Legal context: [All 12 retrieved chunks]
│   │   4. User question: Original query
│   │   5. Instructions: "FOLLOW FORMAT EXACTLY. SAY EXACT NAMES/PLACES."
│   ├─ Final prompt length: ~7,000 tokens
│   └─ Time: <1ms
│
├─► STEP 7: LLM GENERATION
│   ├─ API: OpenRouter → tngtech/tng-r1t-chimera:free
│   ├─ System prompt: "NEVER show reasoning, thinking... Answer in CRISIS format"
│   ├─ Parameters:
│   │   - max_tokens: 3000
│   │   - temperature: 0.05 (very focused, minimal creativity)
│   ├─ Process: Chimera generates response following crisis template
│   ├─ Response: ~2,000 tokens of crisis-formatted guidance
│   │   "CRISIS SITUATION - FALSE 498A CASE
│   │    
│   │    You're facing a criminal charge under Section 498A...
│   │    
│   │    SURVIVAL STEPS (DO THESE TODAY):
│   │    • Go to [nearest district court] immediately..."
│   └─ Time: ~15-18s
│
├─► STEP 8: POST-PROCESSING
│   ├─ Clean response:
│   │   - Remove any leaked "LEGAL CONTEXT" sections
│   │   - Strip thinking/reasoning artifacts
│   ├─ Extract top 5 sources for citation
│   ├─ Validate structure (ensure crisis format present)
│   └─ Time: <10ms
│
└─► STEP 9: API RESPONSE
    ├─ Format:
    │   {
    │     "success": true,
    │     "answer": "CRISIS SITUATION - FALSE 498A CASE\n\n...",
    │     "sources": [
    │       {"source": "bns_etc_2019", "kind": "section", "score": 0.89},
    │       {"source": "crpc_1973", "kind": "section", "score": 0.84},
    │       {"source": "dv_act_2005", "kind": "section", "score": 0.81},
    │       ...
    │     ],
    │     "total_context": 12
    │   }
    └─ Total latency: ~18-21 seconds
        ├─ Classification: 2.5s
        ├─ Encoding: 0.2s
        ├─ Pinecone search: 0.15s
        ├─ LLM generation: 15.5s
        └─ Processing: 0.05s

BOTTLENECK: LLM generation (85% of total time)
```

### Data Flow Architecture

```
                    OFFLINE (Index Setup)
┌────────────────────────────────────────────────────────┐
│                                                          │
│  Legal PDFs → Parse & Chunk → Generate Embeddings      │
│  (IPC, CrPC,    (sections,      (E5-large-v2 model)    │
│   CPC, etc.)    articles)                               │
│                                                          │
│         ↓                                                │
│                                                          │
│   Upload to Pinecone Index                              │
│   ├─ Index name: indian-law-docs                       │
│   ├─ Dimension: 1024                                    │
│   ├─ Metric: cosine                                     │
│   ├─ Total vectors: ~2,800                             │
│   └─ Metadata: source, kind, text, topic               │
│                                                          │
└────────────────────────────────────────────────────────┘
                            ↓
                    (Hosted on Pinecone)
                            ↓
                    ONLINE (Runtime)
┌────────────────────────────────────────────────────────┐
│                                                          │
│  App Startup:                                           │
│    • Initialize Pinecone client                         │
│    • Connect to index: indian-law-docs                 │
│    • Load SentenceTransformer model (~420MB)           │
│    • Start FastAPI server                               │
│                                                          │
│  Per Request:                                           │
│    • Encode query with SentenceTransformer             │
│    • Query Pinecone over network                        │
│    • Build prompt with results                          │
│    • Call OpenRouter API                                │
│    • Return formatted response                          │
│                                                          │
└────────────────────────────────────────────────────────┘
```

### System Components

```
┌─────────────────────────────────────────────────────────┐
│              FastAPI Application (~2GB)                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  SentenceTransformer Model (E5-Large-v2)                │
│  • Model weights: ~420MB                                 │
│  • Tokenizer: ~2MB                                       │
│  • Runtime VRAM: ~1.5GB (CPU mode)                      │
│                                                           │
│  Pinecone Client                                         │
│  • API connection (network-based)                        │
│  • No local vector storage                               │
│  • Minimal memory footprint (~50MB)                      │
│                                                           │
│  FastAPI + Dependencies                                  │
│  • FastAPI framework: ~20MB                              │
│  • Uvicorn server: ~15MB                                 │
│  • Requests, Pydantic: ~30MB                             │
│                                                           │
│  PyTorch (CPU)                                           │
│  • Core libraries: ~800MB                                │
│  • Required for SentenceTransformers                     │
│                                                           │
└─────────────────────────────────────────────────────────┘

TOTAL FOOTPRINT: ~2.3GB (vs. ~450MB for NumPy version)
```

---

## Why Pinecone Instead of Local Embeddings?

### Pinecone Advantages

**1. Real-Time Updates**  
Adding new legal documents or judgments doesn't require rebuilding the entire application. Just upsert vectors to the Pinecone index and they're immediately searchable.

**2. Better Search Quality**  
Pinecone uses optimized HNSW (Hierarchical Navigable Small World) indexing, which can be more accurate than brute-force NumPy cosine similarity for larger corpora.

**3. Scalability**  
As the legal corpus grows (adding case law, state acts, ordinances), Pinecone handles millions of vectors without degrading performance. NumPy similarity search becomes slow beyond ~50,000 documents.

**4. Metadata Filtering**  
Pinecone's native metadata filtering is efficient. The router LLM classifies queries into topics, and we filter Pinecone results by source/kind without scanning the entire corpus.

**5. Development Flexibility**  
During development, you can experiment with different chunking strategies, embedding models, or corpus additions without regenerating local files.

### Pinecone Tradeoffs

**1. External Dependency**  
Requires internet connectivity and Pinecone account. The NumPy version is fully self-contained.

**2. Latency**  
Network round-trip to Pinecone adds ~100-150ms per query. For local embeddings, similarity search is <50ms.

**3. Cost**  
Pinecone's free tier supports 1 index with 100K vectors. Beyond that, paid plans start at $70/month. NumPy version costs nothing.

**4. Larger Docker Image**  
Shipping `sentence-transformers` + PyTorch adds ~1.5GB to the container, which blocks deployment on ultra-low-resource platforms.

### Why I Chose This Architecture Initially

When I started building this system, I wanted:
- The ability to rapidly iterate on the legal corpus
- Easy updates when new acts are passed
- Professional-grade vector search capabilities
- Room to scale to 10K+ documents

Pinecone was the obvious choice. The NumPy version came later as a workaround for Railway's deployment constraints.

---

## Features

### Two-Stage Routing System

Unlike naive RAG systems that dump the entire query into vector search, this uses a **classification-then-filter** approach:

1. **LLM Router**: Chimera analyzes the question and outputs topic tags (family law, criminal law, consumer protection, etc.)
2. **Metadata Filter**: Those tags map to specific Indian Acts in Pinecone metadata
3. **Filtered Search**: Only queries vectors from relevant legal domains

**Why this matters**: A question about "harassment at workplace" shouldn't retrieve sections from the Motor Vehicles Act. The router ensures retrieval precision.

### Crisis-Formatted Responses

Every answer follows a strict template:
- **Immediate action steps**: What to do in the next 24 hours
- **Exact locations**: Police stations, courts, legal aid offices
- **Document checklist**: Evidence to gather
- **Contact information**: Helpline numbers, lawyer referral services
- **Legal basis**: Relevant sections cited from retrieved context

This isn't academic legal analysis. It's survival instructions for people in urgent situations.

### Source Attribution

The API returns the top 5 Pinecone matches with:
- Source identifier (e.g., `dv_act_2005`, `crpc_1973`)
- Chunk type (section, article, rule, qa_explanation)
- Similarity score (0.0 to 1.0)

Users can verify claims against actual statutory text.

---

## API Reference

### Endpoints

#### Health Check

**Endpoint:** `GET /health`

Returns system status and configuration.

**Response:**
```json
{
  "status": "healthy",
  "model": "tngtech/tng-r1t-chimera:free",
  "pinecone_index": "indian-law-docs"
}
```

#### Ask a Legal Question

**Endpoint:** `POST /ask`

Submit a legal query and receive crisis-formatted guidance.

**Request:**
```json
{
  "question": "My landlord is refusing to return my security deposit after I moved out"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "CRISIS SITUATION - SECURITY DEPOSIT DISPUTE\n\nYou're dealing with unlawful retention of your deposit, which is both a civil breach and consumer violation.\n\nSURVIVAL STEPS (DO THESE TODAY):\n• Send legal notice to landlord via registered post demanding return within 15 days...",
  "sources": [
    {
      "source": "consumer_protection_act_2019",
      "kind": "section",
      "score": 0.87
    },
    {
      "source": "registration_act_1908",
      "kind": "section",
      "score": 0.82
    }
  ],
  "total_context": 12
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Pinecone query timeout",
  "answer": "Service temporarily unavailable"
}
```

#### Root Endpoint

**Endpoint:** `GET /`

Returns basic usage information.

**Response:**
```json
{
  "message": "Indian Law RAG API is running!",
  "usage": "POST /ask with {'question': 'your legal query'}",
  "example": "curl -X POST /ask -d '{\"question\": \"wife filed 498A\"}'"
}
```

---

## Local Development Setup

### Prerequisites

- Python 3.11 or higher
- Pinecone account with an index set up
- OpenRouter API key
- 4GB+ available RAM (for PyTorch + SentenceTransformers)

### Installation

Clone the repository:

```bash
git clone https://github.com/theunknownodysseus/indian-law-rag-pipeline.git
cd indian-law-rag-pipeline
```

Create virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pinecone==3.2.2
sentence-transformers==3.1.1
requests==2.32.3
pydantic==2.9.2
python-multipart==0.0.9
torch==2.4.1+cpu
numpy==1.26.4
transformers==4.45.1
loguru==0.7.2
```

### Configuration

Set environment variables:

```bash
export PINECONE_API_KEY=your_pinecone_api_key
export OPENROUTER_API_KEY=your_openrouter_api_key
```

Or create a `.env` file:

```
PINECONE_API_KEY=pcsk_xxxxx
OPENROUTER_API_KEY=sk-or-xxxxx
```

### Creating the Pinecone Index

If you don't have the index yet:

```python
import pinecone

pinecone.init(api_key="your_key", environment="us-west1-gcp-free")

# Create index with E5-large-v2 dimensions
pinecone.create_index(
    name="indian-law-docs",
    dimension=1024,
    metric="cosine"
)

# Upload your legal corpus embeddings
# (See offline pipeline section below)
```

### Running the Server

Start the development server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Access API documentation at `http://localhost:8000/docs`

Test with curl:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what are my rights under consumer protection act?"}'
```

---

## Deployment

### Why This Version is Harder to Deploy

The ~2GB Docker image (PyTorch + SentenceTransformers) exceeds free-tier limits on most platforms:

| Platform | Free Tier Limit | Will This Deploy? |
|----------|----------------|-------------------|
| Railway | 1GB image | ❌ No |
| Render | 512MB RAM | ❌ No |
| Fly.io | 256MB RAM | ❌ No |
| Heroku | 512MB RAM | ❌ No |
| AWS Lambda | 10GB image | ✅ Yes (but cold start ~30s) |
| Google Cloud Run | 4GB image | ✅ Yes |
| Azure Container Instances | 4GB image | ✅ Yes |

### Recommended Deployment: Google Cloud Run

Cloud Run supports large containers and has a generous free tier (2M requests/month).

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
```

**Deploy:**

```bash
gcloud run deploy indian-law-rag \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --timeout 60s
```

Set environment variables in Cloud Run console:
- `PINECONE_API_KEY`
- `OPENROUTER_API_KEY`

### Alternative: Run on a VPS

Cheapest option with full control:

1. Get a $5/month DigitalOcean droplet (2GB RAM)
2. Install Docker
3. Clone repo and run:

```bash
docker build -t indian-law-rag .
docker run -d \
  -p 8000:8000 \
  -e PINECONE_API_KEY=your_key \
  -e OPENROUTER_API_KEY=your_key \
  indian-law-rag
```

4. Use nginx as reverse proxy with SSL

---

## The Offline Pipeline (Creating the Index)

This section explains how the Pinecone index gets populated initially.

### Step 1: Parse Legal Documents

Extract text from PDFs of Indian statutes:

```python
import pymupdf  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Process all acts
acts = [
    "IPC_1860.pdf",
    "CrPC_1973.pdf",
    "Evidence_Act_1872.pdf",
    # ... etc
]

corpus = []
for act_file in acts:
    text = extract_text_from_pdf(act_file)
    corpus.append({"source": act_file, "text": text})
```

### Step 2: Chunk into Semantic Units

Split by sections/articles while preserving legal structure:

```python
import re

def chunk_legal_text(text, source_name):
    chunks = []
    
    # Match "Section 123:" or "Article 45:" patterns
    sections = re.split(r'(Section \d+|Article \d+)', text)
    
    for i in range(1, len(sections), 2):
        section_num = sections[i]
        section_text = sections[i+1] if i+1 < len(sections) else ""
        
        chunks.append({
            "id": f"{source_name}_{section_num.replace(' ', '_')}",
            "text": section_text[:2000],  # Truncate long sections
            "source": source_name,
            "kind": "section",
            "topic": infer_topic(section_text)  # Custom function
        })
    
    return chunks
```

### Step 3: Generate Embeddings

Use the same E5-large-v2 model as runtime:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2")

def embed_chunks(chunks):
    for chunk in chunks:
        text = f"passage: {chunk['text']}"  # E5 requires prefix
        embedding = model.encode(text).tolist()
        chunk["embedding"] = embedding
    return chunks

embedded_corpus = embed_chunks(all_chunks)
```

### Step 4: Upload to Pinecone

```python
import pinecone

pinecone.init(api_key="your_key", environment="us-west1-gcp-free")
index = pinecone.Index("indian-law-docs")

# Batch upload
batch_size = 100
for i in range(0, len(embedded_corpus), batch_size):
    batch = embedded_corpus[i:i+batch_size]
    
    vectors = [
        (
            chunk["id"],
            chunk["embedding"],
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "kind": chunk["kind"],
                "topic": chunk["topic"]
            }
        )
        for chunk in batch
    ]
    
    index.upsert(vectors=vectors)

print(f"Uploaded {len(embedded_corpus)} vectors to Pinecone")
```

This process runs once when setting up the system or whenever the legal corpus is updated.

---

## Technical Decisions Explained

### Why E5-Large-v2 for Embeddings?

Several embedding models were tested:

| Model | Dimension | Performance | Speed | Choice |
|-------|-----------|-------------|-------|--------|
| `all-MiniLM-L6-v2` | 384 | Decent | Fast | ❌ Too weak for legal text |
| `all-mpnet-base-v2` | 768 | Good | Medium | ❌ Insufficient domain knowledge |
| `intfloat/e5-large-v2` | 1024 | Excellent | Medium | ✅ Best balance |
| `sentence-t5-xxl` | 768 | Excellent | Slow | ❌ Too slow for runtime |

E5-large-v2 was trained on massive text pairs including legal/technical domains. It understands legal terminology better than general-purpose models.

### Why Not Fine-Tune the Embedding Model?

Fine-tuning on Indian legal text would improve retrieval accuracy by 10-15%. However:
- Requires curated training pairs (legal query → relevant section)
- Training takes ~8 GPU-hours
- Model weights grow from 420MB to ~1.5GB after LoRA adapters
- Deployment becomes even more resource-intensive

For this project, pretrained E5 was "good enough." A production system would fine-tune.

### The LLM Router: Why Bother?

Early versions used pure vector search without classification. Problems:

1. **Topic Confusion**: "How to register a company?" would retrieve marriage registration procedures
2. **Wasted Context**: Including irrelevant Acts in the prompt confused the LLM
3. **Slow Retrieval**: Searching all 2,800 vectors when only 200 are relevant

The router adds 2 seconds of latency but improves answer quality significantly. The tradeoff is worth it.

### Temperature 0.05: Why So Low?

Legal advice should be consistent and precise. High temperature (0.7+) causes:
- Inconsistent formatting (sometimes follows crisis template, sometimes doesn't)
- Made-up case citations
- Vague instructions ("consult a lawyer near you" instead of specific helplines)

Temperature 0.05 keeps responses deterministic while allowing minimal variation for natural language.

---

## Limitations and Known Issues

### Current Constraints

**Latency**: End-to-end response time is 18-21 seconds:
- 2.5s: LLM routing
- 0.2s: Query encoding
- 0.15s: Pinecone search
- 15.5s: Answer generation
- 0.05s: Post-processing

The LLM generation is the bottleneck. Switching to Llama 3.1 8B (self-hosted) could reduce this to 5-7 seconds.

**No Case Law**: The index only contains statutory text. Supreme Court and High Court judgments would add crucial precedent and interpretation.

**English Only**: Many Indian laws have official Hindi versions that aren't included.

**Static Prompts**: The crisis format is hardcoded. Users can't request different response styles (academic, brief, etc.).

**No Multi-Turn Conversations**: Each query is independent. Follow-up questions like "What's the punishment for that?" require re-stating context.

### Accuracy Considerations

This system provides legal information, not legal advice:

**It can:**
- Surface relevant statutory provisions
- Explain general legal procedures
- Point to resources (helplines, legal aid)

**It cannot:**
- Replace a qualified lawyer
- Provide case-specific legal strategy
- Guarantee accuracy for edge cases
- Represent you in court

Always consult professional legal counsel for serious matters.

---

## Contributing

This is a research project and proof-of-concept. If you want to improve it:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Make your changes with clear commit messages
4. Test locally with `pytest` (tests are in `/tests` if they exist)
5. Submit a pull request with a description of what you've added

Particularly valuable contributions:
- Adding more legal corpora (state laws, case law)
- Improving retrieval accuracy
- Reducing response latency
- Adding multi-language support

---

## License

This project's code is available under the MIT License. However:

- **Legal corpus**: Indian statutes are public domain as government works
- **OpenRouter/Chimera**: Subject to OpenRouter's terms of service
- **Dependencies**: Each library has its own license (see `requirements.txt`)

Review all upstream licenses before commercial use.

---

## Acknowledgments

- Legal corpus sourced from [IndianKanoon](https://indiankanoon.org/) and [Ministry of Law and Justice](https://legislative.gov.in/)
- Built with [FastAPI](https://fastapi.tiangolo.com/) by Sebastián Ramírez
- LLM access via [OpenRouter](https://openrouter.ai/)
- Inspired by legal aid initiatives and the need for accessible legal information

---

## Contact

For questions, suggestions, or issues:

- **GitHub Issues**: [Open an issue](https://github.com/theunknownodysseus/indian-law-rag-pipeline/issues)
- **Repository**: [github.com/theunknownodysseus/indian-law-rag-pipeline](https://github.com/theunknownodysseus/indian-law-rag-pipeline)

Built by [Varun](https://github.com/theunknownodysseus)
---
