"""
Qwen3-VL grounded visual-evidence extraction (spec "Phase 4").

Talks to the existing local Ollama service directly over its HTTP API --
the same integration style `orchestrator.py`'s `ollama_chat` already uses
for session summarization -- rather than adding a new VLM-serving
framework or an `ollama` PyPI client dependency. Qwen3-VL is a grounded
evidence extractor here, not a creative captioner: its only job is to turn
one cropped visual region into retrieval-grade structured text, at
ingestion time, never at question-answering time.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .config import VisualConfig
from . import gpu_lifecycle

REQUIRED_STRING_FIELDS = ("visual_type", "summary")
_ALLOWED_VISUAL_TYPES = {
    "bar_chart", "line_chart", "scatter_plot", "pie_chart", "diagram",
    "architecture_diagram", "network_diagram", "flowchart", "screenshot",
    "table_image", "photo", "other",
}


class VisualAnalysisError(RuntimeError):
    """Base class for a visual-analysis failure. Callers must catch this
    (or a subclass) and record it as a per-visual failure status -- never
    let it propagate up and abort MAIN_TEXT indexing for the document."""


class VLMUnavailableError(VisualAnalysisError):
    pass


class VLMTimeoutError(VisualAnalysisError):
    pass


class VLMInvalidResponseError(VisualAnalysisError):
    pass


GROUNDING_PROMPT = """You are a grounded evidence extractor analyzing ONE visual region cropped from a page of a real document (a chart, diagram, screenshot, or figure). Your only job is to convert what is ACTUALLY VISIBLE into structured retrieval evidence. You are not a creative captioner.

Rules (follow all of them):
- Report only information visible in the supplied image. Never invent, guess, or extrapolate a value you cannot actually read.
- Preserve readable labels, units, series/category names, and dates/times exactly as shown.
- Extract numerical values only when they are legible enough to support the claim; if a value is only approximately readable, say "approximately" in the value itself rather than a false-precision exact number.
- If something is not legible or not confidently present, omit it from the relevant field and add a short note to "uncertainties" instead of guessing.
- Distinguish observation ("the blue bar reaches ~70") from interpretation ("this suggests improved efficiency") -- interpretation belongs only in "trend"/"key_takeaway", phrased cautiously.
- Do not describe colors/styling unless the color encodes meaning (e.g. a legend).
- Do not mention publication metadata, authors, or journal/venue information even if visible in the crop.
- Do not use generic filler like "this image contains a colorful chart". Every sentence must carry retrieval-useful content.
- If the image is decorative, illegible, or contains no analyzable content, still return the JSON with "visual_type": "other" and explain why in "summary" and "uncertainties" -- never refuse and never return prose outside the JSON object.

Output STRICT JSON only -- no markdown code fences, no commentary before or after -- matching exactly this shape:
{
  "visual_type": "bar_chart | line_chart | scatter_plot | pie_chart | diagram | architecture_diagram | network_diagram | flowchart | screenshot | table_image | photo | other",
  "title": "string, the visual's own title/caption text if shown, else \\"\\"",
  "summary": "1-3 sentences, factual, grounded only in what is visible",
  "axes": {"x": {"label": "string or \\"\\"", "units": "string or \\"\\""}, "y": {"label": "string or \\"\\"", "units": "string or \\"\\""}},
  "legend": ["string", "..."],
  "data_points": [{"label": "string", "value": "string"}],
  "relationships": ["string", "..."],
  "trend": "string or \\"\\"",
  "key_takeaway": "string or \\"\\"",
  "uncertainties": ["string", "..."]
}"""


@dataclass
class ValidatedVisualAnalysis:
    visual_type: str
    title: str
    summary: str
    axes: Dict[str, Dict[str, str]]
    legend: List[str]
    data_points: List[Dict[str, str]]
    relationships: List[str]
    trend: str
    key_takeaway: str
    uncertainties: List[str]
    raw: Dict[str, Any] = field(default_factory=dict)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


def _safe_str(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def _safe_str_list(v: Any) -> List[str]:
    if not isinstance(v, list):
        return []
    return [s.strip() for s in v if isinstance(s, str) and s.strip()]


def _safe_axis(v: Any) -> Dict[str, str]:
    if not isinstance(v, dict):
        return {"label": "", "units": ""}
    return {"label": _safe_str(v.get("label")), "units": _safe_str(v.get("units"))}


def _safe_axes_container(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _safe_data_points(v: Any) -> List[Dict[str, str]]:
    if not isinstance(v, list):
        return []
    out = []
    for item in v:
        if isinstance(item, dict) and (item.get("label") or item.get("value")):
            out.append({"label": _safe_str(item.get("label")), "value": _safe_str(item.get("value"))})
        elif isinstance(item, str) and item.strip():
            out.append({"label": item.strip(), "value": ""})
    return out


def parse_and_validate(raw_text: str) -> ValidatedVisualAnalysis:
    """Turns Qwen3-VL's raw response text into a `ValidatedVisualAnalysis`,
    coercing every optional field to a safe default and raising
    `VLMInvalidResponseError` only when the response is unusable outright
    (empty, not JSON, or missing both required fields) -- a malformed
    field inside an otherwise-parseable response is downgraded to its
    default rather than failing the whole visual, since most VLM schema
    drift is partial (an extra/missing optional key), not total."""
    if not raw_text or not raw_text.strip():
        raise VLMInvalidResponseError("empty response from VLM")

    cleaned = _strip_code_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some responses wrap valid JSON in leading/trailing prose despite
        # the prompt/format constraint -- salvage the first top-level
        # object rather than failing outright.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise VLMInvalidResponseError(f"response is not valid JSON: {raw_text[:200]!r}")
        try:
            data = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as e:
            raise VLMInvalidResponseError(f"response is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise VLMInvalidResponseError(f"response JSON is not an object: {type(data).__name__}")

    visual_type = _safe_str(data.get("visual_type")) or "other"
    if visual_type not in _ALLOWED_VISUAL_TYPES:
        visual_type = "other"
    summary = _safe_str(data.get("summary"))
    if not summary:
        raise VLMInvalidResponseError("response has no usable 'summary' field")

    return ValidatedVisualAnalysis(
        visual_type=visual_type,
        title=_safe_str(data.get("title")),
        summary=summary,
        axes={"x": _safe_axis(_safe_axes_container(data.get("axes")).get("x")),
              "y": _safe_axis(_safe_axes_container(data.get("axes")).get("y"))},
        legend=_safe_str_list(data.get("legend")),
        data_points=_safe_data_points(data.get("data_points")),
        relationships=_safe_str_list(data.get("relationships")),
        trend=_safe_str(data.get("trend")),
        key_takeaway=_safe_str(data.get("key_takeaway")),
        uncertainties=_safe_str_list(data.get("uncertainties")),
        raw=data,
    )


def call_vlm(image_bytes: bytes, config: VisualConfig) -> str:
    """Raises `VLMUnavailableError`/`VLMTimeoutError` on transport failure;
    returns the raw response text (still unvalidated) on success."""
    img_b64 = base64.b64encode(image_bytes).decode("ascii")
    try:
        resp = requests.post(
            config.ollama_host.rstrip("/") + "/api/generate",
            json={
                "model": config.model,
                "prompt": GROUNDING_PROMPT,
                "images": [img_b64],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1},
                "keep_alive": config.keep_alive_during_batch,
            },
            timeout=config.timeout_seconds,
        )
    except requests.exceptions.Timeout as e:
        raise VLMTimeoutError(f"Qwen3-VL request timed out after {config.timeout_seconds}s: {e}") from e
    except requests.exceptions.RequestException as e:
        raise VLMUnavailableError(f"Could not reach Ollama at {config.ollama_host}: {e}") from e

    if resp.status_code == 404:
        raise VLMUnavailableError(
            f"Model {config.model!r} is not available on Ollama at {config.ollama_host} (HTTP 404). "
            f"Run `ollama pull {config.model}` or verify the tag."
        )
    if not resp.ok:
        raise VLMUnavailableError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
    except ValueError as e:
        raise VLMInvalidResponseError(f"Ollama response was not valid JSON envelope: {e}") from e
    if data.get("error"):
        raise VLMUnavailableError(f"Ollama reported an error: {data['error']}")
    return data.get("response", "")


def get_model_digest(config: VisualConfig) -> Optional[str]:
    """Best-effort lookup of the exact local Ollama model digest for
    `config.model` (via `/api/tags`, the same registry `ollama list` reads)
    -- recorded on every visual record for cache-key/provenance purposes
    (spec: "Ollama model digest where available"). Returns `None` rather
    than raising if Ollama is unreachable or the tag isn't found."""
    try:
        resp = requests.get(config.ollama_host.rstrip("/") + "/api/tags", timeout=10)
        resp.raise_for_status()
        for m in resp.json().get("models", []):
            if m.get("name") == config.model or m.get("model") == config.model:
                return m.get("digest")
    except requests.exceptions.RequestException:
        pass
    return None


def unload_model(config: VisualConfig) -> bool:
    """Explicitly releases `config.model` from Ollama's GPU residency
    (spec Phase 11: the VLM must not need to stay loaded for ordinary
    question answering). Best-effort -- returns False (never raises) if
    Ollama is unreachable, since failing to unload is not a reason to fail
    an otherwise-successful ingestion pass. Delegates to
    `gpu_lifecycle.unload_ollama_model` -- the same `keep_alive: 0`
    primitive the strict-sequential-GPU-lifecycle phase generalized for
    every Ollama-owned model (this VLM and the `qwen3:14b` generator
    alike), so there is exactly one implementation of "unload an Ollama
    model" in this codebase."""
    return gpu_lifecycle.unload_ollama_model(config.ollama_host, config.model)


# Bound on data_points/relationships list length in the rendered retrieval
# text (see build_retrieval_text) -- keeps one densely-labeled diagram from
# alone exceeding a downstream embedding model's max sequence length.
_MAX_LIST_ITEMS = 15
# Per-field word budget: a per-item count cap alone is not sufficient (a
# small number of individually-long relationship sentences, e.g. a fully-
# connected variable-dependency diagram, can still blow past an embedding
# model's max sequence length -- confirmed on a real corpus PDF during
# acceptance testing, see RAG_BUILD_SOURCE_OF_TRUTH.md's change log).
_MAX_FIELD_WORDS = 90
# Whole-text safety net (see the final cap in build_retrieval_text): even
# with every individual field bounded, several long-but-individually-legal
# fields stacked together could still add up too high. ~220 words stays
# safely under BGE's 512-token ceiling (even at a conservative ~1.8
# tokens/word for jargon-dense technical text) while remaining far below
# Qwen's much larger operational budget.
_MAX_TOTAL_WORDS = 220


def _cap_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " (truncated)"


def _render_lines(
    analysis: ValidatedVisualAnalysis, heading_path: List[str], page: int,
    *,
    include_summary: bool = True,
    include_axes: bool = True,
    include_legend: bool = True,
    data_points_limit: Optional[int] = _MAX_LIST_ITEMS,
    relationships_limit: Optional[int] = _MAX_LIST_ITEMS,
    include_trend: bool = True,
    include_key_takeaway: bool = True,
    include_uncertainties: bool = True,
    field_word_budget: int = _MAX_FIELD_WORDS,
) -> List[str]:
    """Builds the ordered retrieval-text lines for one visual. Every
    optional section can be individually included/excluded or bounded --
    the knobs `build_retrieval_text_within_budget` uses to reduce the
    rendered text field-by-field, in priority order, when it doesn't fit
    the active embedding model's token budget. `visual_type`/`page` are
    never optional here: they're provenance, not detail, and are always
    included by every caller."""
    lines: List[str] = []
    vt_label = analysis.visual_type.replace("_", " ")
    lines.append(f"Visual type: {vt_label.capitalize()}.")
    if analysis.title:
        lines.append(f"Title: {analysis.title}.")
    if heading_path:
        lines.append(f"Section: {' > '.join(heading_path)}.")
    lines.append(f"Page: {page}.")
    if include_summary and analysis.summary:
        lines.append(_cap_words(analysis.summary if analysis.summary.endswith(".") else analysis.summary + ".", field_word_budget))
    if include_axes:
        x, y = analysis.axes.get("x", {}), analysis.axes.get("y", {})
        if x.get("label"):
            unit = f", units: {x['units']}" if x.get("units") else ""
            lines.append(f"X-axis: {x['label']}{unit}.")
        if y.get("label"):
            unit = f", units: {y['units']}" if y.get("units") else ""
            lines.append(f"Y-axis: {y['label']}{unit}.")
    if include_legend and analysis.legend:
        lines.append("Series/legend: " + ", ".join(analysis.legend) + ".")
    if data_points_limit != 0 and analysis.data_points:
        parts = [f"{d['label']}: {d['value']}" if d.get("value") else d["label"] for d in analysis.data_points if d.get("label") or d.get("value")]
        # Bounded so one densely-labeled chart (many series/categories) can
        # never alone blow the retrieval text past a downstream embedding
        # model's max sequence length (spec Phase 6: this text must fit the
        # SAME embedding path MAIN_TEXT chunks use, which enforces its own
        # token budget) -- this is machine-generated summary prose, not
        # retained source text, so bounding list length here is not the
        # "never silently truncate" MAIN_TEXT invariant; the omission is
        # also visible/reportable (see below), never silent.
        limit = data_points_limit if data_points_limit is not None else len(parts)
        if len(parts) > limit:
            parts = parts[:limit] + [f"(+{len(parts) - limit} more, not shown)"]
        if parts:
            lines.append(_cap_words("Data points: " + "; ".join(parts) + ".", field_word_budget))
    if relationships_limit != 0 and analysis.relationships:
        rels = analysis.relationships
        limit = relationships_limit if relationships_limit is not None else len(rels)
        if len(rels) > limit:
            rels = rels[:limit] + [f"(+{len(analysis.relationships) - limit} more relationships not shown)"]
        # Item COUNT alone is not a reliable bound: a fully-connected-looking
        # diagram (e.g. a variable-dependency graph) can produce few
        # relationship sentences that are each individually long ("X
        # influences A, B, C, ... , and Z."), as observed on a real corpus
        # PDF during acceptance testing -- so each rendered field also gets
        # its own word-budget cap (`_cap_words`), not just the list.
        lines.append(_cap_words("Relationships: " + " ".join(rels), field_word_budget))
    if include_trend and analysis.trend:
        lines.append(_cap_words(f"Trend: {analysis.trend}", field_word_budget))
    if include_key_takeaway and analysis.key_takeaway:
        lines.append(_cap_words(f"Key takeaway: {analysis.key_takeaway}", field_word_budget))
    if include_uncertainties and analysis.uncertainties:
        lines.append(_cap_words("Uncertain/not clearly readable: " + "; ".join(analysis.uncertainties) + ".", field_word_budget))
    return lines


def build_retrieval_text(analysis: ValidatedVisualAnalysis, heading_path: List[str], page: int) -> str:
    """Normalizes the structured analysis into concise retrieval-grade
    prose (spec Phase 4's "Retrieval text generation") -- what actually
    gets embedded and BM25-indexed, not the raw JSON (a normalized
    sentence-per-fact rendering keyword-matches and reads far better in
    both BM25 and dense retrieval than a JSON blob would).

    This applies only the word-budget/list-count caps -- a fast,
    tokenizer-free first pass (spec: "Keep those as an optional first-pass
    reduction if useful"). It is NOT the final correctness boundary for
    embedding: callers that need a hard guarantee against
    `EmbeddingOversizedInputError` must use
    `build_retrieval_text_within_budget`, which validates against the
    active embedding backend's own tokenizer/max-sequence-length."""
    lines = _render_lines(analysis, heading_path, page)
    text = "\n".join(lines)
    # Whole-text word-count safety net: drop lowest-priority lines from the
    # end (uncertainties, then trend/key_takeaway, then relationships/data
    # points) until under budget, always keeping the header lines (visual
    # type/title/section/page) and summary -- the minimum needed to
    # identify and cite the visual -- intact. This is a word-count
    # approximation only; see `build_retrieval_text_within_budget` for the
    # real, tokenizer-native guarantee.
    while len(text.split()) > _MAX_TOTAL_WORDS and len(lines) > 5:
        lines.pop()
        text = "\n".join(lines)
    return text


# Token-native reduction order for `build_retrieval_text_within_budget`
# (spec "REDUCTION PRIORITY"): each step is applied cumulatively (on top of
# every previous step, never undoing one), from least- to most-important,
# stopping as soon as the actual tokenizer count fits the active embedding
# model's budget. `visual_type`, `page`, `Section` (heading path), and
# `title` are provenance/identity, never reduced -- removing them to solve
# a length problem would violate "never remove provenance metadata to
# solve an embedding length issue". Uncertainties/trend are VLM caveats and
# interpretation rather than the requested preservation-priority fields
# (1 visual type / 2 title / 3 axes / 4 legend / 5 key takeaway /
# 6 relationships / 7-8 data points / 9 summary), so they are reduced
# first, ahead of the lowest tier on that list (summary detail).
_TOKEN_BUDGET_REDUCTION_STEPS: List["tuple[str, dict]"] = [
    ("drop_uncertainties", {"include_uncertainties": False}),
    ("drop_trend", {"include_trend": False}),
    ("trim_summary_detail", {"field_word_budget": 40}),          # 9. verbose summary/detail
    ("trim_secondary_data_points", {"data_points_limit": 5}),    # 8. secondary data points
    ("trim_to_key_data_points", {"data_points_limit": 2}),       # 7. important readable data points (keep only the most important few)
    ("trim_secondary_relationships", {"relationships_limit": 5}),  # 6. important relationships (secondary)
    ("trim_to_key_relationships", {"relationships_limit": 2}),   # 6. important relationships (down to the most important)
    ("drop_key_takeaway", {"include_key_takeaway": False}),      # 5. key takeaway
    ("drop_legend", {"include_legend": False}),                  # 4. key labels/series
    ("drop_axes", {"include_axes": False}),                      # 3. axis labels/units
    ("drop_summary", {"include_summary": False}),                # last resort before provenance-only
]


@dataclass
class RetrievalTextSizing:
    text: str
    token_count: int
    max_tokens: Optional[int]
    fits_budget: bool
    reduction_steps_applied: List[str] = field(default_factory=list)


def build_retrieval_text_within_budget(
    analysis: ValidatedVisualAnalysis, heading_path: List[str], page: int,
    count_tokens, max_tokens: Optional[int],
) -> RetrievalTextSizing:
    """The final correctness boundary for visual retrieval text (spec
    "TOKEN VALIDATION REQUIREMENT"): builds retrieval text, measures it
    with `count_tokens` (the active embedding backend's own
    tokenizer -- pass `EmbeddingClient.count_tokens`, the same mechanism
    `chunker.py`'s token-native chunking already uses, never a second
    tokenizer implementation), and if it doesn't fit `max_tokens` (the
    active embedding backend's own `max_sequence_tokens`, never a
    hardcoded constant), progressively drops/bounds fields in
    `_TOKEN_BUDGET_REDUCTION_STEPS` order -- least-important evidence
    first, provenance (visual type/title/section/page) never removed --
    re-measuring with the real tokenizer after every step, until it fits
    or every reducible field has been exhausted.

    Never slices token IDs or characters mid-field: every reduction either
    drops a whole optional field or bounds a list/word count at a
    field boundary, so what remains is always well-formed, readable
    evidence, never a mid-sentence cut. `count_tokens`/`max_tokens` may be
    `None` (embedding client unavailable) -- in that case the word-budget
    text from `build_retrieval_text` is returned as-is with
    `fits_budget=True` (nothing to validate against), matching this
    function's caller's existing behavior when no embedding client was
    configured."""
    kwargs: dict = {}
    lines = _render_lines(analysis, heading_path, page, **kwargs)
    text = "\n".join(lines)
    if count_tokens is None or not max_tokens:
        return RetrievalTextSizing(text=text, token_count=-1, max_tokens=max_tokens, fits_budget=True)

    token_count = count_tokens(text)
    if token_count <= max_tokens:
        return RetrievalTextSizing(text=text, token_count=token_count, max_tokens=max_tokens, fits_budget=True)

    applied: List[str] = []
    for step_name, step_kwargs in _TOKEN_BUDGET_REDUCTION_STEPS:
        kwargs.update(step_kwargs)
        applied.append(step_name)
        lines = _render_lines(analysis, heading_path, page, **kwargs)
        text = "\n".join(lines)
        token_count = count_tokens(text)
        if token_count <= max_tokens:
            return RetrievalTextSizing(
                text=text, token_count=token_count, max_tokens=max_tokens,
                fits_budget=True, reduction_steps_applied=applied,
            )

    # Every reducible field was dropped/minimized and it still doesn't fit
    # (e.g. an unusually small `max_tokens`, or a title alone longer than
    # the budget) -- return the most-reduced version anyway (provenance
    # fields intact, never fabricated/paraphrased) and let the caller's
    # existing embed_batch() rejection path record the failure explicitly,
    # exactly as it already does; this function's job is to make that
    # outcome as rare as possible, not to guarantee it never happens.
    return RetrievalTextSizing(
        text=text, token_count=token_count, max_tokens=max_tokens,
        fits_budget=False, reduction_steps_applied=applied,
    )


def analyze_visual(image_bytes: bytes, config: VisualConfig) -> "tuple[ValidatedVisualAnalysis, str]":
    """End-to-end: call the VLM, validate the response, return
    `(analysis, raw_response_text)`. Raises `VisualAnalysisError` subclasses
    on failure -- callers must catch and record a per-visual failure status
    rather than letting this abort document indexing."""
    raw_text = call_vlm(image_bytes, config)
    analysis = parse_and_validate(raw_text)
    return analysis, raw_text
