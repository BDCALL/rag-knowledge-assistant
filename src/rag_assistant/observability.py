"""Structured logging and request correlation.

Every HTTP request gets an ``X-Request-ID`` (taken from the client if it sent a
sane one, generated otherwise). The ID is stored in a :class:`~contextvars.ContextVar`
so that *any* log line emitted while handling the request, including from the
service layer or a worker thread, carries it without passing it around by hand.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

access_logger = logging.getLogger("rag_assistant.access")

# Attributes every LogRecord has; anything else was passed via ``extra=`` and is emitted.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


class RequestIdFilter(logging.Filter):
    """Attach the current request ID (or ``-``) to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy to ship to Loki/CloudWatch/ELK and query."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and key not in payload and key != "request_id":
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure the root logger once at process start (idempotent)."""
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s %(message)s")
        )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


def new_request_id(incoming: str | None) -> str:
    """Trust a client-supplied ID only if it is short and safe to log."""
    if incoming and _VALID_ID.match(incoming):
        return incoming
    return uuid.uuid4().hex


class RequestContextMiddleware:
    """Pure ASGI middleware (no BaseHTTPMiddleware) so streaming responses keep working."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        rid = new_request_id(headers.get(REQUEST_ID_HEADER.lower()))
        token = request_id_var.set(rid)
        started = time.perf_counter()
        status_code = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((REQUEST_ID_HEADER.encode(), rid.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            access_logger.info(
                "%s %s %s",
                scope.get("method"),
                scope.get("path"),
                status_code,
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            request_id_var.reset(token)
