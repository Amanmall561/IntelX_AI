"""
run_hitl.py — Pipeline + HITL Wrapper
=======================================
Drop-in wrapper around the existing moduler_call.main() pipeline.
ZERO changes to existing DocX_AI code.

Usage:
    python run_hitl.py /path/to/document.pdf
    python run_hitl.py /path/to/document.pdf --doc-id my-doc-001
    python run_hitl.py /path/to/document.pdf --threshold 0.80

The script:
  1. Runs the IntelX_AI extraction pipeline (moduler_call.main)
  2. Passes the output through the HITL Orchestrator (Phase 4.1→5.2)
  3. Prints the HITLResult as JSON to stdout
  4. Exits with code 0 (COMPLETED) or 2 (PENDING_REVIEW / error)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow running from the MultiLang_DocX directory without installing
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Add DocX_AI to the path so moduler_call.main works
DOCX_DIR = Path("/home/ubuntu/docextract-worker/DocX_AI")
if DOCX_DIR.exists() and str(DOCX_DIR) not in sys.path:
    sys.path.insert(0, str(DOCX_DIR))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(str(ROOT / "hitl.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("run_hitl")


def run(
    file_path: str,
    document_id: Optional[str] = None,
    threshold: float = 0.75,
) -> dict:
    """
    Main integration entry point.

    Args:
        file_path:   Absolute or relative path to the document.
        document_id: Optional document UUID; auto-generated if None.
        threshold:   Confidence threshold (overrides config.py).

    Returns:
        HITLResult as a dict.
    """
    from typing import Optional
    doc_id = document_id or str(uuid.uuid4())
    logger.info("Processing document: %s (ID: %s)", file_path, doc_id)

    # ── Step 1: Run existing IntelX_AI pipeline ───────────────────────────────
    pipeline_output = None
    try:
        from moduler_call import main as pipeline_main
        logger.info("Running IntelX_AI extraction pipeline...")
        pipeline_output = pipeline_main(file_path)
        logger.info("Pipeline complete. Output type: %s", type(pipeline_output).__name__)
    except ImportError as e:
        logger.warning(
            "moduler_call failed to import. Missing dependency? "
            f"Error: {e}\n"
            "Running HITL with empty pipeline output (test/demo mode)."
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        # Continue — HITL will handle the error gracefully

    # ── Step 2: Run HITL Orchestrator ─────────────────────────────────────────
    from hitl.orchestrator import HITLOrchestrator
    orchestrator = HITLOrchestrator(threshold=threshold)

    hitl_result = orchestrator.process(
        pipeline_output=pipeline_output,
        document_id=doc_id,
        file_path=file_path,
    )

    logger.info(
        "HITL Result: state=%s | confidence=%.1f%% | review_item=%s",
        hitl_result.state.value,
        hitl_result.gate_result.overall_confidence * 100,
        hitl_result.review_item_id or "N/A",
    )

    return hitl_result.model_dump(mode="json")


# ── Optional type hint (avoid circular import at module level) ─────────────────
try:
    from typing import Optional
except ImportError:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IntelX_AI Pipeline + HITL Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_hitl.py /docs/aadhaar.jpg
  python run_hitl.py /docs/bank_statement.pdf --threshold 0.80
  python run_hitl.py /docs/fir.pdf --doc-id case-001-fir
        """,
    )
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Custom document ID (UUID auto-generated if not provided)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Confidence threshold for auto-approval (default: 0.75)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    result = run(
        file_path=args.file_path,
        document_id=args.doc_id,
        threshold=args.threshold,
    )

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, default=str))

    # Exit codes:
    #   0 = COMPLETED (auto-approved)
    #   2 = PENDING_REVIEW (routed to human review queue)
    #   1 = REJECTED or error
    state = result.get("state", "")
    if state == "COMPLETED":
        sys.exit(0)
    elif state == "PENDING_REVIEW":
        sys.exit(2)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
