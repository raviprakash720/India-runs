"""
domain_finetune.py - Domain-Specific Fine-Tuning for Embedding Model
=====================================================================
Redrob Data & AI Challenge - Semantic Alignment via Fine-Tuning

Since this is an OFFLINE challenge, we cannot run fine-tuning at
submission time.  This module instead:

  1. Generates synthetic (query, positive, negative) training triplets
     derived from the JD for a 'Senior AI Engineer – Founding Team'
     role (embeddings, retrieval, vector DBs, ranking, RAG, LLM FT).
  2. Provides `finetune_model()` to train offline once before submission.
  3. Provides `load_model()` to load the fine-tuned model if available,
     falling back to the base model transparently.

Usage:
    # One-time offline training (before submission)
    python domain_finetune.py

    # In rank.py / precompute.py
    from domain_finetune import load_model
    model = load_model("cache/finetuned_minilm")
"""

import os
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "cache", "finetuned_minilm")
BASE_MODEL_NAME = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# TRAINING TRIPLETS
# ──────────────────────────────────────────────
# Each entry is (query, positive, negative).
# Positives are realistic skill/experience descriptions a strong
# candidate would have.  Negatives are plausible résumé text from
# completely unrelated domains.
TRAINING_PAIRS = [
    # ── Retrieval / Embeddings ──
    (
        "embeddings-based retrieval engineer",
        "experience with FAISS and sentence-transformers for semantic search",
        "marketing campaign management and analytics",
    ),
    (
        "semantic search specialist",
        "built dense retrieval pipelines using bi-encoder models and ANN indices",
        "supply chain logistics and warehouse management",
    ),
    (
        "neural information retrieval researcher",
        "published work on dense passage retrieval with contrastive learning",
        "interior design and residential architecture projects",
    ),
    (
        "embedding model optimization",
        "quantized sentence-transformer models for low-latency serving on CPU",
        "event planning and hospitality management",
    ),
    # ── Vector Databases ──
    (
        "vector database expert",
        "deployed Pinecone/Weaviate at scale for production search",
        "graphic design portfolio with Adobe tools",
    ),
    (
        "vector search infrastructure engineer",
        "managed Qdrant clusters indexing 50M+ vectors with sub-10ms p99 latency",
        "certified public accountant with tax compliance experience",
    ),
    (
        "scalable similarity search architect",
        "designed HNSW-based vector index on Milvus serving 10K QPS in production",
        "social media content creation and influencer outreach",
    ),
    # ── Ranking & Evaluation ──
    (
        "search ranking engineer",
        "improved NDCG@10 by 15% using learning-to-rank with LambdaMART",
        "mechanical engineering CAD modeling and simulations",
    ),
    (
        "relevance evaluation specialist",
        "built evaluation frameworks measuring MRR, MAP, and recall across search verticals",
        "pharmaceutical sales representative covering oncology drugs",
    ),
    (
        "candidate ranking system developer",
        "designed multi-stage ranking pipeline with BM25 recall and cross-encoder reranking",
        "professional photography and video editing",
    ),
    # ── RAG (Retrieval-Augmented Generation) ──
    (
        "RAG system architect",
        "built retrieval-augmented generation pipelines combining vector search with GPT-4",
        "civil engineering structural analysis and bridge design",
    ),
    (
        "conversational AI with retrieval",
        "implemented hybrid RAG with dense retrieval, sparse BM25, and reciprocal rank fusion",
        "culinary arts training and restaurant management",
    ),
    (
        "grounded LLM application developer",
        "reduced hallucinations by 40% via chunk-level citation in RAG pipeline",
        "real estate property valuation and appraisal",
    ),
    # ── LLM Fine-Tuning ──
    (
        "LLM fine-tuning engineer",
        "fine-tuned LLaMA-2 70B with QLoRA on domain-specific instruction datasets",
        "human resources benefits administration and payroll",
    ),
    (
        "parameter-efficient fine-tuning specialist",
        "applied LoRA and PEFT adapters to BERT and T5 for domain adaptation",
        "fashion retail merchandising and trend forecasting",
    ),
    (
        "instruction tuning researcher",
        "created RLHF training pipeline for aligning language models with user intent",
        "agricultural crop science and irrigation planning",
    ),
    # ── Core ML / Deep Learning ──
    (
        "deep learning engineer with NLP focus",
        "trained transformer models in PyTorch with mixed-precision and distributed data parallel",
        "insurance underwriting and actuarial analysis",
    ),
    (
        "machine learning platform engineer",
        "built ML pipelines on Kubernetes with MLflow experiment tracking and model registry",
        "legal contract review and litigation support",
    ),
    (
        "production ML engineer",
        "deployed real-time inference APIs serving transformer models at 50ms SLA",
        "nursing patient care and clinical documentation",
    ),
    # ── Python / Systems ──
    (
        "Python backend engineer for AI systems",
        "developed async FastAPI microservices with Redis caching for model serving",
        "automotive mechanic specializing in diesel engines",
    ),
    (
        "data pipeline engineer",
        "built ETL pipelines processing 10TB+ text data for embedding precomputation",
        "floral arrangement and landscape gardening",
    ),
    # ── Startup / Founding ──
    (
        "founding engineer at AI startup",
        "built search infrastructure from scratch as first ML hire at a seed-stage startup",
        "corporate compliance and regulatory auditing",
    ),
    (
        "full-stack AI engineer in early-stage company",
        "owned end-to-end ML stack from data labeling to model deployment in a 5-person team",
        "broadcast journalism and news anchoring",
    ),
    # ── Hybrid / Cross-Cutting ──
    (
        "search and recommendation engineer",
        "designed hybrid retrieval combining collaborative filtering with semantic embeddings",
        "dental hygiene and orthodontics practice",
    ),
    (
        "A/B testing for search quality",
        "ran interleaving experiments to validate ranking model improvements in production",
        "music production and audio engineering for film",
    ),
]


def generate_training_data():
    """
    Return the list of (query, positive, negative) triplets for
    domain-specific fine-tuning.

    Returns
    -------
    list[tuple[str, str, str]]
        Each element is a (query, positive_text, negative_text) triplet.
    """
    return list(TRAINING_PAIRS)


# ──────────────────────────────────────────────
# FINE-TUNING
# ──────────────────────────────────────────────

def finetune_model(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    base_model: str = BASE_MODEL_NAME,
    epochs: int = 3,
) -> str:
    """
    Fine-tune `base_model` on the JD-derived triplets and save to
    `output_dir`.  Meant to be run **once** offline before submission.

    Parameters
    ----------
    output_dir : str
        Directory to save the fine-tuned model weights.
    base_model : str
        HuggingFace model name for the base sentence-transformer.
    epochs : int
        Number of training epochs.

    Returns
    -------
    str
        Path to the saved fine-tuned model (== `output_dir`).

    Raises
    ------
    ImportError
        If sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers (and PyTorch) must be installed to "
            "fine-tune.  Run: pip install sentence-transformers"
        ) from exc

    logger.info("Loading base model '%s' …", base_model)
    model = SentenceTransformer(base_model)

    # ── Build training examples ──
    triplets = generate_training_data()
    train_examples = [
        InputExample(texts=[q, pos, neg])
        for q, pos, neg in triplets
    ]

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=8,
    )

    # ── TripletLoss (anchor, positive, negative) ──
    train_loss = losses.TripletLoss(model=model)

    # ── Train ──
    warmup_steps = max(1, int(len(train_dataloader) * epochs * 0.1))
    logger.info(
        "Fine-tuning for %d epochs (%d steps, %d warmup) …",
        epochs,
        len(train_dataloader) * epochs,
        warmup_steps,
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True,
    )

    logger.info("Fine-tuned model saved to '%s'", output_dir)
    return output_dir


# ──────────────────────────────────────────────
# MODEL LOADING (with graceful fallback)
# ──────────────────────────────────────────────

def load_model(cache_dir: str = DEFAULT_OUTPUT_DIR):
    """
    Load the fine-tuned SentenceTransformer from `cache_dir` if it
    exists, otherwise fall back to the base model.

    Parameters
    ----------
    cache_dir : str
        Path where the fine-tuned model was saved by `finetune_model()`.

    Returns
    -------
    SentenceTransformer
        Ready-to-use embedding model.

    Notes
    -----
    If the ``sentence-transformers`` package is not installed this
    function logs a warning and returns ``None`` so callers can degrade
    gracefully (e.g. switch to TF-IDF scoring).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers is not installed — returning None.  "
            "Install with: pip install sentence-transformers"
        )
        return None

    # Prefer fine-tuned weights when available
    if os.path.isdir(cache_dir) and os.path.isfile(
        os.path.join(cache_dir, "config.json")
    ):
        logger.info("Loading fine-tuned model from '%s'", cache_dir)
        return SentenceTransformer(cache_dir)

    logger.info(
        "Fine-tuned model not found at '%s'; loading base model '%s'",
        cache_dir,
        BASE_MODEL_NAME,
    )
    return SentenceTransformer(BASE_MODEL_NAME)


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("=== Domain Fine-Tuning for RedRob Challenge ===")
    logger.info("Triplets: %d", len(TRAINING_PAIRS))

    saved_path = finetune_model()
    logger.info("Done. Model at: %s", saved_path)
