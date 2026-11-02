"""
Command-line entry point for the document pipeline.

    python -m document_converter.cli process   <file> [--profile main_text|full_fidelity]
    python -m document_converter.cli validate  <file> [--profile main_text|full_fidelity]
    python -m document_converter.cli preview   <file> [--profile main_text|full_fidelity]
    python -m document_converter.cli index     <file> [--force] [--profile main_text|full_fidelity]
    python -m document_converter.cli scan      <dir>  [--force] [--profile main_text|full_fidelity]
    python -m document_converter.cli rebuild            # re-index everything under paths.source_dir + legacy_docs_dir

Every subcommand shares one `DocumentPipeline` built from `load_config()`,
so CLI behavior always matches what ingestion.py / orchestrator.py do.

`--profile` defaults to the pipeline config's profile (MAIN_TEXT unless
overridden by `DOCPIPE_PROFILE` or a config file -- see config.py and
DELIVERABLES.md's "MAIN_TEXT vs FULL_FIDELITY ingestion profiles"). MAIN_TEXT
indexes only the filtered main-text retrieval view and gates on retrieval-
critical findings; FULL_FIDELITY preserves today's original behavior --
the complete document, gated on full structural validation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import load_config, INGESTION_PROFILES
from .pipeline import DocumentPipeline, PipelineValidationError


def _progress_printer():
    """Returns a `progress(stage, payload)` callback that prints one status
    line per event to stderr as the pipeline runs -- `index`/`process`/
    `scan` are the paths actually used for the largest documents (the CLI,
    unlike the orchestrator, has no job registry to poll instead). stdout's
    existing output format is untouched -- that's what the existing tests
    assert on -- so progress only ever goes to stderr."""
    start = time.monotonic()

    def cb(stage: str, payload: dict) -> None:
        elapsed = time.monotonic() - start
        bits = [f"[{elapsed:7.1f}s]", stage]
        if "source" in payload:
            bits.append(payload["source"])
        if "current" in payload and payload.get("total"):
            bits.append(f"{payload['current']}/{payload['total']}")
        elif "current" in payload:
            bits.append(str(payload["current"]))
        print(" ".join(str(b) for b in bits), file=sys.stderr)

    return cb


def _cmd_process(pipeline: DocumentPipeline, args) -> int:
    result = pipeline.process(args.file, profile=args.profile, progress=_progress_printer())
    print(f"Wrote {result.markdown_path}")
    print(f"Wrote {result.report_path}")
    print(f"Profile: {result.profile}  Title: {result.title!r}")
    print(f"Chunks: {len(result.chunking_result.chunks)}  "
          f"Warnings: {result.validation_report.stats['warnings']}  "
          f"Errors: {result.validation_report.stats['errors']}  "
          f"Retrieval-ready: {result.validation_report.retrieval_ready}")
    return 0 if result.validation_report.retrieval_ready else 1


def _cmd_validate(pipeline: DocumentPipeline, args) -> int:
    report = pipeline.validate_only(args.file, profile=args.profile)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.to_human_text())
    return 0 if report.retrieval_ready else 1


def _cmd_preview(pipeline: DocumentPipeline, args) -> int:
    result = pipeline.preview(args.file, profile=args.profile)
    print(f"# {result.title}  (profile: {result.profile})\n")
    for c in result.chunking_result.chunks:
        print(f"--- chunk {c.chunk_index} ({c.word_count}w / ~{c.token_count}t) {c.provenance} ---")
        print(c.text)
        print()
    print(f"Chunks: {len(result.chunking_result.chunks)}  "
          f"Warnings: {result.validation_report.stats['warnings']}  "
          f"Errors: {result.validation_report.stats['errors']}  "
          f"Retrieval-ready: {result.validation_report.retrieval_ready}", file=sys.stderr)
    return 0


def _cmd_index(pipeline: DocumentPipeline, args) -> int:
    try:
        result = pipeline.index(args.file, force=args.force, profile=args.profile, progress=_progress_printer())
    except PipelineValidationError as e:
        print(str(e), file=sys.stderr)
        return 1
    if result.skipped_unchanged:
        print(f"Unchanged, skipped re-indexing: {result.index_path}")
    else:
        print(f"Indexed {result.document_id} ({result.profile}) -> {result.index_path} "
              f"({len(result.chunking_result.chunks)} chunks)")
    return 0


def _cmd_scan(pipeline: DocumentPipeline, args) -> int:
    results = pipeline.scan_and_index(args.dir, force=args.force, profile=args.profile,
                                       progress=_progress_printer())
    failed = 0
    for r in results:
        status = r["status"]
        if status == "failed":
            failed += 1
            print(f"FAILED  {r['source']}: {r['error']}")
        else:
            print(f"{status.upper():8} {r['source']} ({r.get('chunks', 0)} chunks, "
                  f"{r.get('warnings', 0)} warnings, {r.get('errors', 0)} errors)")
    print(f"\n{len(results)} file(s) scanned, {failed} failed")
    return 1 if failed else 0


def _cmd_rebuild(pipeline: DocumentPipeline, args) -> int:
    total_failed = 0
    for root in (pipeline.config.paths.source_dir, pipeline.config.paths.legacy_docs_dir):
        results = pipeline.scan_and_index(root, force=args.force, profile=args.profile)
        failed = sum(1 for r in results if r["status"] == "failed")
        total_failed += failed
        print(f"{root}: {len(results)} file(s), {failed} failed")
        for r in results:
            if r["status"] == "failed":
                print(f"  FAILED {r['source']}: {r['error']}")
    return 1 if total_failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="document_converter", description=__doc__)
    parser.add_argument("--config", help="Path to a docpipe.yaml/json config file")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_profile_arg(subparser):
        subparser.add_argument(
            "--profile", choices=INGESTION_PROFILES, default=None,
            help="Ingestion profile (default: the pipeline config's profile, MAIN_TEXT unless overridden)",
        )

    p = sub.add_parser("process", help="Extract, normalize, chunk, validate; write markdown + report")
    p.add_argument("file")
    _add_profile_arg(p)
    p.set_defaults(func=_cmd_process)

    p = sub.add_parser("validate", help="Run validation only, print the findings report")
    p.add_argument("file")
    p.add_argument("--json", action="store_true")
    _add_profile_arg(p)
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("preview", help="Print the chunks that would be produced, without writing anything")
    p.add_argument("file")
    _add_profile_arg(p)
    p.set_defaults(func=_cmd_preview)

    p = sub.add_parser("index", help="Process + embed + write a versioned index entry")
    p.add_argument("file")
    p.add_argument("--force", action="store_true", help="Re-index even if the source is unchanged")
    _add_profile_arg(p)
    p.set_defaults(func=_cmd_index)

    p = sub.add_parser("scan", help="Index every supported file under a directory")
    p.add_argument("dir")
    p.add_argument("--force", action="store_true")
    _add_profile_arg(p)
    p.set_defaults(func=_cmd_scan)

    p = sub.add_parser("rebuild", help="Re-index everything under the configured source and legacy docs directories")
    p.add_argument("--force", action="store_true")
    _add_profile_arg(p)
    p.set_defaults(func=_cmd_rebuild)

    args = parser.parse_args(argv)
    config = load_config(args.config)
    pipeline = DocumentPipeline(config)
    try:
        return args.func(pipeline, args)
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
