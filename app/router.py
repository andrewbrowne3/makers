"""Embeddings-based router → picks a branch (skill) for a request.

Branches are logical skill clusters: image_gen, evaluate, ask_question, ...
Each branch has a small corpus of example phrasings. We embed them into
Chroma once, then route incoming requests by nearest-neighbor similarity.

Cosine similarity with a score threshold (Andrew's Langchain_stuff pattern).
If nothing beats the threshold the router returns ("ask_question", 0.0) as
a safe default — the Question agent clarifies with the user.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from app.config import CFG
from app.logging_config import get_logger
from app.providers.embeddings import EmbeddingProvider, get_embedder

log = get_logger("router")

COLLECTION = "route_intents"

# Seed corpus — one list of phrasings per branch. Grow these as the system matures.
SEED_CORPUS: dict[str, list[str]] = {
    "image_gen": [
        "generate a new design with this logo",
        "make a t-shirt mockup with our logo on the front",
        "create a hat design with the client logo embroidered",
        "render a sock design with the brand color and logo",
        "new logo added, produce garment images",
        "place this logo on a hoodie",
    ],
    "evaluate": [
        "score this design",
        "is the logo readable on the shirt",
        "check this rendered garment for quality",
        "does the logo placement look correct",
        "evaluate whether this mockup matches the brand colors",
        "grade this image against the brief",
    ],
    "ask_question": [
        "what garment type should I use",
        "which template matches this product",
        "clarify the color palette",
        "i'm missing information",
        "please provide more details",
        "what placement does the client want",
    ],
}


@dataclass
class RouteDecision:
    branch: str
    score: float
    matched_example: Optional[str] = None
    distance: Optional[float] = None


class RouterIndex:
    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder
        Path(CFG.embed.chroma_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=CFG.embed.chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
        log.info("🗺️  RouterIndex ready collection=%s count=%d", COLLECTION, self.collection.count())

    def rebuild(self, corpus: Optional[dict[str, list[str]]] = None) -> int:
        corpus = corpus or SEED_CORPUS
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:  # noqa: BLE001
            pass
        self.collection = self.client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        for branch, examples in corpus.items():
            for i, ex in enumerate(examples):
                ids.append(f"{branch}:{i}")
                docs.append(ex)
                metas.append({"branch": branch})

        embeddings = self.embedder.embed(docs)
        self.collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        log.info("🧱 router.rebuild added=%d branches=%d", len(ids), len(corpus))
        return len(ids)

    def classify(self, query: str) -> RouteDecision:
        if self.collection.count() == 0:
            log.warning("⚠️  router collection empty — rebuilding from seed")
            self.rebuild()

        vec = self.embedder.embed([query])[0]
        res = self.collection.query(query_embeddings=[vec], n_results=1)
        doc = (res["documents"] or [[None]])[0][0]
        meta = (res["metadatas"] or [[{}]])[0][0] or {}
        dist = (res["distances"] or [[1.0]])[0][0]
        score = max(0.0, 1.0 - float(dist))
        branch = str(meta.get("branch", "ask_question"))

        if score < CFG.embed.route_threshold:
            log.info("🔀 route FALLBACK score=%.3f below threshold=%.2f → ask_question", score, CFG.embed.route_threshold)
            return RouteDecision(branch="ask_question", score=score, matched_example=doc, distance=float(dist))

        log.info("🔀 route → %s score=%.3f match=%r", branch, score, doc)
        return RouteDecision(branch=branch, score=score, matched_example=doc, distance=float(dist))


_router: Optional[RouterIndex] = None


def get_router() -> RouterIndex:
    global _router
    if _router is None:
        _router = RouterIndex(get_embedder())
    return _router
