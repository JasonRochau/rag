"""
Lightweight retrieval smoke-eval (spec section 10, reduced scope -- see
DELIVERABLES.md for why this is a hand-written smoke check against the real
indexed corpus rather than the full curated gold-question harness with
precision@k targets across every fixture format the spec describes).

Runs a small set of hand-written questions against whatever is currently in
docs/.index/ (schema_version 2 only) via the same HybridRanker used in
production, and reports top-1/top-3/top-5 hit rates against an expected
document-title substring per question.

Usage:
    python -m document_converter.eval_retrieval
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from .config import load_config
from .pipeline import DocumentPipeline
from .index_schema import load_index, is_legacy_schema

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control import HybridRanker  # noqa: E402  (repo-root module, not part of this package)


@dataclass
class GoldQuestion:
    query: str
    expected_doc_substring: str
    note: str = ""


# Hand-written against the real CCNA/Azure/DS&A study corpus this pipeline
# ships with -- update or extend this list as the corpus changes. This is
# intentionally small: it is a smoke check, not the formal evaluation the
# full spec calls for.
GOLD_QUESTIONS: List[GoldQuestion] = [
    GoldQuestion("How does NAT translate a private IP address to a public one?", "ip_services"),
    GoldQuestion("What is the difference between a trunk port and an access port?", "network_access"),
    GoldQuestion("How does Rapid PVST+ spanning tree prevent loops?", "network_access"),
    GoldQuestion("What is the role of DHCP in assigning IP addresses?", "ip_services"),
    GoldQuestion("What are the layers of the OSI model and what do routers do?", "network_fundamentals"),
    GoldQuestion("How do you configure Azure role-based access control?", "role_based_access_control"),
    GoldQuestion("What is Microsoft Entra ID used for?", "entra_id"),
    GoldQuestion("How do you deploy Azure infrastructure using ARM templates?", "arm_templates"),
    GoldQuestion("What is Azure Cloud Shell and how do you access it?", "cloud_shell"),
    GoldQuestion("What are common CCNA interview behavioral questions?", "behavioral_prep"),
]


def _load_chunks(index_dir: str) -> List[dict]:
    chunks: List[dict] = []
    for index_file in sorted(glob.glob(os.path.join(index_dir, "*.json"))):
        try:
            payload = load_index(index_file)
        except Exception:
            continue
        if is_legacy_schema(payload):
            continue
        doc = payload["document"]
        for c in payload.get("chunks", []):
            if not c.get("embedding"):
                continue
            chunks.append({
                "doc_id": doc["document_id"], "doc_title": doc["title"], "source_path": doc["source_path"],
                "chunk_idx": c["chunk_index"], "heading": c["heading"], "text": c["text"], "embedding": c["embedding"],
            })
    return chunks


def run_eval(k: int = 5, alpha: float = 0.7) -> int:
    config = load_config()
    pipeline = DocumentPipeline(config)
    chunks = _load_chunks(config.paths.index_dir)
    if not chunks:
        print("No indexed chunks found -- run `python -m document_converter.cli rebuild` first.")
        return 2

    client = pipeline.embedding_client
    ranker = HybridRanker(alpha=alpha)

    hits_at = {1: 0, 3: 0, 5: 0}
    total = len(GOLD_QUESTIONS)
    print(f"Evaluating {total} question(s) against {len(chunks)} indexed chunk(s)\n")

    for gq in GOLD_QUESTIONS:
        query_embedding = client.embed_one(gq.query, input_type="query")
        results = ranker.rank(gq.query, chunks, query_embedding, k=k, debug=False)
        titles = [r["source_path"] for r in results]
        rank_hit: Optional[int] = None
        for i, t in enumerate(titles, 1):
            if gq.expected_doc_substring.lower() in t.lower():
                rank_hit = i
                break
        for threshold in hits_at:
            if rank_hit is not None and rank_hit <= threshold:
                hits_at[threshold] += 1
        status = f"hit @ {rank_hit}" if rank_hit else "MISS"
        print(f"[{status:>8}] {gq.query}")
        if not rank_hit:
            print(f"           expected doc containing {gq.expected_doc_substring!r}, got: "
                  f"{[os.path.basename(t) for t in titles[:3]]}")

    print()
    for threshold in (1, 3, 5):
        pct = 100.0 * hits_at[threshold] / total if total else 0.0
        print(f"top-{threshold}: {hits_at[threshold]}/{total} ({pct:.0f}%)")

    return 0 if hits_at[5] == total else 1


if __name__ == "__main__":
    raise SystemExit(run_eval())
