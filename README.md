# Is Retrieval Enough?

A controlled evaluation of retrieval-based evidence selection for long-document summarisation.

This project compares four summarisation pipelines to investigate how retrieval strategy affects **factual grounding, completeness and overall summary quality** when only a limited amount of source context can be provided to a language model.

The experiment uses the **GovReport** long-document summarisation dataset and evaluates the methods across source-context budgets of **4,000, 6,000, 8,000, 10,000 and 12,000 tokens**.

## Retrieval Methods

The four evidence-selection strategies are:

1. **No-RAG baseline**  
   Uses the beginning of the source document up to the configured token budget.

2. **Basic RAG**  
   Splits the full document into overlapping chunks, embeds the chunks and a shared retrieval query, and ranks chunks using cosine similarity.

3. **Improved RAG**  
   Extends Basic RAG with document-specific keyword reranking and near-duplicate removal.

4. **Context-aware RAG**  
   Uses the Improved RAG ranking and adds neighbouring chunks around selected evidence to preserve local context, while remaining within the same source-token budget.

## Experimental Configuration

- Dataset: GovReport
- Documents per budget: 40
- Methods per document: 4
- Source-context budgets: 4,000 / 6,000 / 8,000 / 10,000 / 12,000 tokens
- Chunk size: 180 words
- Chunk overlap: 40 words
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Generation model: `gpt-4o-mini`
- Temperature: `0.2`
- Maximum output: 220 tokens
- Improved-RAG keyword bonus: `0.02 × number of matched keywords`
- Maximum extracted keywords: 30
- Near-duplicate threshold: word-set overlap greater than `0.75`
- Context-aware window: one preceding and one following chunk

## Evaluation Metrics

### Lexical similarity
- ROUGE-1 precision
- ROUGE-1 recall
- ROUGE-1 F1

### Factual grounding
- Hallucination rate
- Factual precision

### Completeness
- Importance-weighted key-fact recall

### Combined factual quality
- Factual F1

Factual F1 is the harmonic mean of factual precision and importance-weighted key-fact recall, reflecting the balance between grounding and completeness.

## Project Structure

```text
.
├── run_experiments.py        # Runs all four summarisation pipelines
├── pipelines.py              # No-RAG, Basic RAG, Improved RAG and Context-aware RAG
├── retrieval.py              # Dense retrieval and similarity ranking
├── embeddings.py             # Sentence-transformer embeddings
├── chunking.py               # Overlapping word-based chunking
├── prompts.py                # Generation prompts and shared retrieval query
├── generator.py              # OpenAI API generation wrapper
├── data_loader.py            # Loads qualifying GovReport documents
├── evaluation.py             # ROUGE-1 calculations
├── factual_evaluation.py     # Claim-level factual and key-fact evaluation
├── aggregate_winner_table.py # Aggregates metrics and method comparisons
├── config.py                 # Experimental configuration
├── token_utils.py            # Token counting and token-budget utilities
└── results/                  # Generated summaries, metrics and retrieval logs
```

## Installation

Python 3.10 or later is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install openai pandas numpy tqdm python-dotenv datasets huggingface_hub fsspec pyarrow sentence-transformers tiktoken
```

## OpenAI API Key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EVAL_MODEL=gpt-4o-mini
```

Do **not** commit your `.env` file or API key to GitHub.

## Running the Experiment

The source-context budget is controlled by `max_source_tokens` in `config.py`.

For the 4,000-token condition:

```python
max_source_tokens: int = 4_000
```

Run all four methods on 40 documents:

```bash
python run_experiments.py --limit 40
```

The script produces:

```text
results/
├── summaries.csv
├── metrics.csv
└── retrieval_logs.csv
```

- `summaries.csv` stores generated summaries, prompts and source-budget information.
- `metrics.csv` stores ROUGE-1 scores and supporting document-level statistics.
- `retrieval_logs.csv` stores retrieved chunks and retrieval scores.

For other budget conditions, update `max_source_tokens` in `config.py` to 6,000, 8,000, 10,000 or 12,000 before rerunning. Archive results in separate budget-specific folders to avoid overwriting earlier runs.

## Running the Factual Evaluation

Example for the 4,000-token condition:

```bash
python factual_evaluation.py --input results/summaries.csv --budget 4000 --evidence-column article --output-dir results/4k_factual_eval
```

This creates:

```text
factual_key_facts.csv
factual_claims.csv
factual_metrics.csv
```

The evaluator extracts weighted key facts from the reference summaries and evaluates generated-summary claims against source evidence. Claims are labelled as:

- supported
- unsupported
- contradicted
- unclear

The resulting files allow factual metrics to be traced back to individual documents, claims and key-fact decisions.

## Retrieval Logic

### Basic RAG

Documents are split into overlapping chunks. The shared retrieval query and all chunks are embedded into the same vector space using L2-normalised sentence embeddings.

Because the vectors are normalised, the dot product is equivalent to cosine similarity:

```text
score = chunk_embedding · query_embedding
```

Chunks are ranked by this score and selected until the configured source-context budget is reached.

### Improved RAG

Improved RAG adds a keyword-overlap bonus:

```text
combined_score = cosine_similarity + (0.02 × keyword_matches)
```

Up to 30 document-specific keywords are extracted from the opening 1,000 characters of the source document.

Near-duplicate chunks are removed when their word-set overlap ratio exceeds `0.75`.

### Context-aware RAG

Context-aware RAG begins with the Improved RAG ranking. For each high-ranking chunk it attempts to include:

```text
previous chunk
selected chunk
next chunk
```

The selected evidence remains within the same source-token budget and is restored to original document order before generation.

## Shared Retrieval Query

All three RAG pipelines use the same retrieval query so that differences between methods arise from retrieval strategy rather than query wording.

The query prioritises evidence relating to:

- main findings
- recommendations
- central issues
- important entities
- dates
- numerical claims
- causes
- outcomes
- conclusions
- risks, concerns or limitations

## Research Aim

The project asks whether retrieval-based evidence selection is sufficient for producing long-document summaries that are both **well grounded** and **complete** under constrained source-context budgets.

The experiment distinguishes semantic relevance from summary importance: a passage can be relevant to a retrieval query without necessarily containing the most salient findings, recommendations or conclusions needed for a complete document-level summary.

## Reproducibility

The workflow retains:

- generated summaries
- prompts
- retrieval logs
- retrieval scores
- ROUGE metrics
- extracted key facts
- claim-level factual judgements
- key-fact coverage decisions
- document-level factual metrics

These outputs make it possible to trace aggregate results back to individual documents and model outputs.

## Author

Michael Majekodunmi

MSc dissertation project on retrieval-based evidence selection, factual grounding and completeness in long-document summarisation.
