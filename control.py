"""
LOCAL AI ASSISTANT - CONTROL LAYER
Pure scoring, ranking, and similarity logic.
BM25 + cosine hybrid retrieval.
Stateless—called by orchestrator.
Query punctuation stripping for better tokenization.
"""
import math
import string
import numpy as np
from typing import List, Dict
from collections import Counter


def clean_query(query: str) -> str:
    """Strip punctuation and lowercase for tokenization."""
    return query.translate(str.maketrans('', '', string.punctuation)).lower()


class BM25:
    """BM25 ranking algorithm for keyword-based scoring."""
    
    def __init__(self, corpus):
        """
        Args:
            corpus: List of tokenized documents (list of lists of words)
        """
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size else 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.k1 = 1.5
        self.b = 0.75
        
        for doc in corpus:
            self.doc_len.append(len(doc))
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                self.idf[word] = self.idf.get(word, 0) + 1
        
        for word, freq in self.idf.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
    
    def get_scores(self, query):
        """
        Score documents against query.
        
        Args:
            query: List of tokens (words)
        
        Returns:
            List of BM25 scores, one per document
        """
        scores = [0.0] * self.corpus_size
        for word in query:
            if word not in self.idf: 
                continue
            for i in range(self.corpus_size):
                f = self.doc_freqs[i].get(word, 0)
                if f == 0: 
                    continue
                score = self.idf[word] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
                scores[i] += score
        return scores


def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec_a: Vector A (list or array)
        vec_b: Vector B (list or array)
    
    Returns:
        float: Cosine similarity in range [0, 1]
    """
    if not vec_a or not vec_b:
        raise ValueError("Both vectors must be non-empty")
    
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute similarity: zero-magnitude vector")
    
    return float(dot_product / (norm_a * norm_b))


class HybridRanker:
    """
    Hybrid ranking combining BM25 (keyword) and semantic (embedding) signals.
    
    Pipeline:
    1. Semantic retrieval (embeddings) on ALL chunks - primary signal
    2. BM25 boost on semantic top-50 candidates only
    3. Return top-k reranked results
    """
    
    def __init__(self, alpha=0.7):
        """
        Args:
            alpha: Blend weight (0.7 = 70% semantic, 30% BM25)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha
    
    def rank(self, query, chunks, query_embedding, k=4, debug=False):
        """
        Rank chunks against query using hybrid scoring.
        
        Args:
            query: Search query string
            chunks: List of chunk dicts with 'text' and 'embedding' keys
            query_embedding: Embedded query vector
            k: Number of results to return
            debug: Print debug output
        
        Returns:
            List of top-k ranked chunks with scores
        """
        if not chunks:
            raise RuntimeError("No chunks to rank")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        # Clean query before processing
        query_clean = clean_query(query)

        if debug:
            print(f"\n[rank] Query: '{query}' (cleaned: '{query_clean}')")

        # ===== STAGE 1: SEMANTIC RETRIEVAL =====
        # `chunks` is the DocStore's shared, long-lived list -- concurrent
        # requests rank the same objects, so scores are tracked in separate
        # records here rather than written onto the chunk dicts themselves.
        # Mutating them in place previously caused one request's in-flight
        # scores to be visible to (and overwritten by) a concurrent request.
        scored = []
        for c in chunks:
            chunk_embedding = c.get("embedding", None)
            sim = cosine_similarity(query_embedding, chunk_embedding) if chunk_embedding else 0
            scored.append({"chunk": c, "vector_score": sim})
        scored.sort(key=lambda x: x["vector_score"], reverse=True)
        top_semantic = scored[:50]

        if debug:
            print(f"[rank] Retrieved top 50 semantic candidates (alpha={self.alpha})")

        # ===== STAGE 2: BM25 BOOST ON TOP-50 ONLY =====
        top_50_texts = [s["chunk"]["text"].lower().split() for s in top_semantic]
        tokenized_query = query_clean.split()

        bm25 = BM25(top_50_texts)
        bm25_scores = bm25.get_scores(tokenized_query)

        max_bm25 = max(bm25_scores) if bm25_scores else 1
        if max_bm25 == 0:
            max_bm25 = 1

        # ===== STAGE 3: NORMALIZED HYBRID SCORING =====
        results = []
        for i, s in enumerate(top_semantic):
            vector_norm = s["vector_score"]
            bm25_norm = bm25_scores[i] / max_bm25
            hybrid_score = (self.alpha * vector_norm) + ((1 - self.alpha) * bm25_norm)
            # Shallow copy: callers get a chunk dict augmented with scores
            # without the shared original ever being written to.
            merged = dict(s["chunk"])
            merged["score"] = hybrid_score
            merged["vector_raw"] = vector_norm
            merged["bm25_raw"] = bm25_scores[i]
            results.append(merged)

        results.sort(key=lambda x: x["score"], reverse=True)

        if debug:
            print(f"[rank] Hybrid reranked top 50 candidates:")
            for i, c in enumerate(results[:15]):
                doc_label = c.get('doc_title', 'Unknown')[:30]
                chunk_idx = c.get('chunk_idx', '?')
                print(f"  {i+1:2d}. {doc_label} chunk {chunk_idx:3} score={c['score']:.4f} (v:{c['vector_raw']:.4f}, b:{c['bm25_raw']:.4f})")

        return results[:k]

    def get_keywords_for_query(self, query_tokens: List[str], chunk_keywords: List[str]) -> Dict:
        """
        Analyze overlap between query tokens and chunk keywords.
        Used for content relevance assessment.
        
        Returns:
            Dict with overlap analysis
        """
        query_set = set(query_tokens)
        keyword_set = set(chunk_keywords)
        
        overlap = query_set & keyword_set
        
        return {
            "query_tokens": len(query_set),
            "chunk_keywords": len(keyword_set),
            "overlap": len(overlap),
            "overlap_terms": list(overlap),
            "query_coverage": len(overlap) / len(query_set) if query_set else 0,
        }


class RankingMetrics:
    """Compute ranking quality metrics for evaluation."""
    
    @staticmethod
    def mean_reciprocal_rank(results: List[Dict], target_doc_id: str) -> float:
        """
        Compute MRR for a target document.
        MRR = 1 / rank_position (or 0 if not found)
        """
        for i, result in enumerate(results, 1):
            if result.get("doc_id") == target_doc_id:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(results: List[Dict], k: int, relevant_doc_ids: List[str]) -> float:
        """
        Compute Normalized Discounted Cumulative Gain.
        Measures ranking quality when multiple relevant documents exist.
        """
        dcg = 0.0
        for i, result in enumerate(results[:k], 1):
            if result.get("doc_id") in relevant_doc_ids:
                dcg += 1.0 / math.log2(i + 1)
        
        ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_doc_ids), k) + 1))
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    @staticmethod
    def precision_at_k(results: List[Dict], k: int, relevant_doc_ids: List[str]) -> float:
        """
        Compute Precision@K: fraction of top-k results that are relevant.
        """
        relevant_in_top_k = sum(1 for r in results[:k] if r.get("doc_id") in relevant_doc_ids)
        return relevant_in_top_k / k if k > 0 else 0.0