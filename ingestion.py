"""
LOCAL AI ASSISTANT - INGESTION ENTRY POINT

Thin wrapper around document_converter.DocumentPipeline that scans the
legacy docs/ markdown tree (docs/**/*.md, excluding docs/.index and the
newly_formatted_docs_review working directories) and indexes every file
through the shared pipeline: extract -> normalize -> chunk -> validate ->
embed -> versioned index entry (schema_version 2, document_converter/
index_schema.py).

Kept as a script/module (rather than folded entirely into the CLI) because
orchestrator.py's `/ingest` endpoint imports `DocIngester` directly for a
synchronous, in-process rescan triggered from the UI.
"""
import logging
import os

from document_converter.config import load_config
from document_converter.pipeline import DocumentPipeline

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class DocIngester:
    """Scans the legacy docs/ markdown tree and indexes it via DocumentPipeline."""

    def __init__(self, config=None):
        self.config = config or load_config()
        self.pipeline = DocumentPipeline(self.config)

    def ingest(self, force: bool = False) -> dict:
        root = self.config.paths.legacy_docs_dir
        logger.info(f"[ingest] Scanning {root} for markdown sources")
        results = self.pipeline.scan_and_index(root, force=force)

        report = {"new": [], "skipped": [], "failed": [], "total_files": len(results)}
        for r in results:
            if r["status"] == "indexed":
                logger.info(f"  ✓ {r['source']} ({r['chunks']} chunks)")
                report["new"].append({"title": r["source"], "chunks": r["chunks"]})
            elif r["status"] == "skipped":
                logger.debug(f"  = {r['source']} unchanged")
                report["skipped"].append({"title": r["source"], "reason": "unchanged"})
            else:
                logger.error(f"  ✗ {r['source']}: {r.get('error')}")
                report["failed"].append({"title": r["source"], "reason": r.get("error", "unknown error")})

        logger.info(
            f"[ingest] Done: {len(report['new'])} indexed, {len(report['skipped'])} unchanged, "
            f"{len(report['failed'])} failed"
        )
        return report


if __name__ == "__main__":
    ingester = DocIngester()
    result = ingester.ingest()
    print("\nIngestion complete:")
    print(f"  Indexed: {len(result['new'])}")
    print(f"  Unchanged: {len(result['skipped'])}")
    print(f"  Failed: {len(result['failed'])}")
    if result["failed"]:
        print(f"  Failed documents: {result['failed']}")
