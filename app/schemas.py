"""Request bodies for the API (responses reuse the TypedDicts in drishti.records)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CaseContext(BaseModel):
    """Every action is tied to an officer and a case reference, for the audit log."""

    officer: str = Field("", max_length=64)
    reason: str = Field("", max_length=200)


class SearchRequest(CaseContext):
    query: str = Field(..., min_length=1, max_length=300)
    top_k: int = Field(12, ge=1, le=50)


class WatchRequest(CaseContext):
    query: str = Field(..., min_length=1, max_length=300)
