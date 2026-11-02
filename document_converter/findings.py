"""Typed validation findings shared by the parser, normalizer, chunker and validator."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class Severity(str, Enum):
    INFO = "INFO"
    FIXED = "FIXED"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class Finding:
    code: str
    message: str
    severity: Severity
    block_id: Optional[str] = None
    line: Optional[int] = None
    page: Optional[int] = None
    repaired: bool = False
    suggested_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "block_id": self.block_id,
            "line": self.line,
            "page": self.page,
            "repaired": self.repaired,
            "suggested_action": self.suggested_action,
        }


class FindingLog:
    """Accumulator used by every pipeline stage; keeps a flat, ordered list."""

    def __init__(self) -> None:
        self.findings: List[Finding] = []

    def add(self, code: str, message: str, severity: Severity, **kw) -> Finding:
        f = Finding(code=code, message=message, severity=severity, **kw)
        self.findings.append(f)
        return f

    def info(self, code: str, message: str, **kw) -> Finding:
        return self.add(code, message, Severity.INFO, **kw)

    def fixed(self, code: str, message: str, **kw) -> Finding:
        return self.add(code, message, Severity.FIXED, repaired=True, **kw)

    def warning(self, code: str, message: str, **kw) -> Finding:
        return self.add(code, message, Severity.WARNING, **kw)

    def error(self, code: str, message: str, **kw) -> Finding:
        return self.add(code, message, Severity.ERROR, **kw)

    def extend(self, other: "FindingLog") -> None:
        self.findings.extend(other.findings)

    def has_errors(self) -> bool:
        return any(f.severity == Severity.ERROR for f in self.findings)

    def count(self, severity: Severity) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    def to_list(self) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self.findings]
