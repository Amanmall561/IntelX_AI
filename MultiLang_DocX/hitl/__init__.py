"""
HITL package initialiser.
Exports the public API surface for convenient importing.
"""
from hitl.models import (
    DocumentState,
    BatchChunk,
    AggregatedDocument,
    ReviewItem,
    Correction,
    FieldCorrection,
    ActiveLearningEntry,
    GateOutcome,
    GateResult,
    HITLResult,
)
from hitl.orchestrator import HITLOrchestrator

__all__ = [
    "DocumentState",
    "BatchChunk",
    "AggregatedDocument",
    "ReviewItem",
    "Correction",
    "FieldCorrection",
    "ActiveLearningEntry",
    "GateOutcome",
    "GateResult",
    "HITLResult",
    "HITLOrchestrator",
]
