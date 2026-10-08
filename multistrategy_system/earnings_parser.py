"""Provider-neutral structured earnings parser contract.

The parser separates document extraction from trading policy.  A model adapter
must return a JSON-like mapping; this module validates it, stamps provenance,
and writes it to :mod:`earnings_events`.  It never places orders and never turns
free-form model prose directly into a broker instruction.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from earnings_events import EASTERN, EarningsAssessment, EarningsEngine


PARSER_VERSION = "earnings-v1"

EARNINGS_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "direction",
        "confidence",
        "fields",
        "evidence",
        "missing_required_fields",
        "contradictions",
    ],
    "properties": {
        "direction": {
            "type": "string",
            "enum": ["bullish", "bearish", "mixed", "neutral", "unknown"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "fields": {"type": "object"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "excerpt"],
                "properties": {
                    "field": {"type": "string"},
                    "excerpt": {"type": "string"},
                    "page_or_section": {"type": ["string", "null"]},
                },
            },
        },
        "missing_required_fields": {
            "type": "array",
            "items": {"type": "string"},
        },
        "contradictions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


@dataclass(frozen=True)
class ParserInput:
    event_id: str
    symbol: str
    document_type: str
    available_at: dt.datetime
    source_text: str
    consensus: Mapping[str, Any]
    source_url: str | None = None

    def __post_init__(self):
        if not self.source_text.strip():
            raise ValueError("earnings parser refuses an empty document")
        if self.available_at.tzinfo is None:
            object.__setattr__(
                self, "available_at", self.available_at.replace(tzinfo=EASTERN)
            )


def build_parser_payload(item: ParserInput) -> dict[str, Any]:
    """Build the complete point-in-time input passed to a model adapter."""

    return {
        "task": (
            "Extract earnings facts from only the supplied document and consensus. "
            "Do not infer missing numbers. Return structured JSON matching the schema."
        ),
        "event_id": item.event_id,
        "symbol": item.symbol.strip().upper(),
        "document_type": item.document_type,
        "available_at": item.available_at.isoformat(),
        "consensus_as_known_at_event": dict(item.consensus),
        "document": item.source_text,
        "output_schema": EARNINGS_OUTPUT_SCHEMA,
    }


def validate_model_output(result: Mapping[str, Any]) -> dict[str, Any]:
    """Strict validation without requiring a JSON-schema dependency."""

    if not isinstance(result, Mapping):
        raise ValueError("earnings parser output must be an object")
    allowed = {
        "direction",
        "confidence",
        "fields",
        "evidence",
        "missing_required_fields",
        "contradictions",
    }
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise ValueError("unexpected parser fields: " + ", ".join(unknown))
    missing = sorted(allowed - set(result))
    if missing:
        raise ValueError("missing parser fields: " + ", ".join(missing))

    direction = str(result["direction"]).lower()
    if direction not in {"bullish", "bearish", "mixed", "neutral", "unknown"}:
        raise ValueError(f"invalid parser direction: {direction}")
    confidence = float(result["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError("parser confidence must be between 0 and 1")
    fields = result["fields"]
    evidence = result["evidence"]
    missing_fields = result["missing_required_fields"]
    contradictions = result["contradictions"]
    if not isinstance(fields, Mapping):
        raise ValueError("parser fields must be an object")
    if not isinstance(evidence, list):
        raise ValueError("parser evidence must be a list")
    normalized_evidence = []
    for item in evidence:
        if not isinstance(item, Mapping):
            raise ValueError("every evidence item must be an object")
        if set(item) - {"field", "excerpt", "page_or_section"}:
            raise ValueError("evidence contains an unsupported field")
        if not str(item.get("field", "")).strip() or not str(
            item.get("excerpt", "")
        ).strip():
            raise ValueError("evidence requires field and excerpt")
        normalized_evidence.append(dict(item))
    if not isinstance(missing_fields, list) or not all(
        isinstance(value, str) for value in missing_fields
    ):
        raise ValueError("missing_required_fields must be a string list")
    if not isinstance(contradictions, list) or not all(
        isinstance(value, str) for value in contradictions
    ):
        raise ValueError("contradictions must be a string list")
    return {
        "direction": direction,
        "confidence": confidence,
        "fields": dict(fields),
        "evidence": normalized_evidence,
        "missing_required_fields": list(missing_fields),
        "contradictions": list(contradictions),
    }


class EarningsParser:
    """Validate one release/call extraction and append it to the event store."""

    def __init__(
        self,
        engine: EarningsEngine,
        extractor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        *,
        model_id: str,
        parser_version: str = PARSER_VERSION,
        clock: Callable[[], dt.datetime] | None = None,
    ):
        # assessed_at is when the assessment really became known, which is
        # what keeps state_at point-in-time; it must not be back-dated to
        # available_at. The clock is injectable for tests and replays.
        self.clock = clock or (lambda: dt.datetime.now(EASTERN))
        self.engine = engine
        self.extractor = extractor
        self.model_id = str(model_id)
        self.parser_version = str(parser_version)

    def parse(self, item: ParserInput) -> EarningsAssessment:
        raw_result = self.extractor(build_parser_payload(item))
        result = validate_model_output(raw_result)
        source_hash = hashlib.sha256(item.source_text.encode("utf-8")).hexdigest()
        payload = {
            "event_id": item.event_id,
            "symbol": item.symbol,
            "document_type": item.document_type,
            "available_at": item.available_at,
            "assessed_at": self.clock(),
            "direction": result["direction"],
            "confidence": result["confidence"],
            # Market confirmation is deliberately supplied by a separate
            # quote/volume stage, never by the document model.
            "market_confirmation": "unavailable",
            "fields": result["fields"],
            "evidence": result["evidence"],
            "missing_required_fields": result["missing_required_fields"],
            "contradictions": result["contradictions"],
            "parser_version": self.parser_version,
            "model_id": self.model_id,
            "source_url": item.source_url,
            "source_hash": source_hash,
        }
        return self.engine.ingest_assessment(payload)


__all__ = [
    "EARNINGS_OUTPUT_SCHEMA",
    "EarningsParser",
    "PARSER_VERSION",
    "ParserInput",
    "build_parser_payload",
    "validate_model_output",
]
