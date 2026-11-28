"""Optional API-key check, sent by clients in the ``X-API-Key`` header."""

from __future__ import annotations

import secrets
from collections.abc import Callable

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = "X-API-Key"

# auto_error=False so a missing header reaches our check and gets the same 401.
_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def api_key_guard(expected: str | None) -> Callable[[str | None], None]:
    """Build a dependency that rejects requests without the expected key.

    With no key (``None``, or ``""`` from an empty ``RAG_API_KEY=`` line) auth is
    off and every request passes, so local development and tests need no key.
    """

    def check(provided: str | None = Security(_header)) -> None:
        if not expected:
            return
        # compare_digest takes the same time wherever the strings differ,
        # so response timing does not leak how much of a guess was right.
        if provided is None or not secrets.compare_digest(provided, expected):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing or invalid API key",
                headers={"WWW-Authenticate": API_KEY_HEADER},
            )

    return check
