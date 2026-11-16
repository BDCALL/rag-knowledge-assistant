from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.app import create_app
from rag_assistant.config import Settings
from rag_assistant.observability import (
    JsonFormatter,
    RequestIdFilter,
    configure_logging,
    new_request_id,
    request_id_var,
)


@pytest.fixture
def client() -> TestClient:
    settings = Settings(environment="test", embedding_dim=64)
    return TestClient(create_app(settings))


def test_generates_request_id_when_missing(client: TestClient) -> None:
    with client:
        response = client.get("/health")
    rid = response.headers["X-Request-ID"]
    assert len(rid) == 32
    assert rid.isalnum()


def test_echoes_valid_client_request_id(client: TestClient) -> None:
    with client:
        response = client.get("/health", headers={"X-Request-ID": "trace-abc.123"})
    assert response.headers["X-Request-ID"] == "trace-abc.123"


def test_rejects_unsafe_client_request_id(client: TestClient) -> None:
    with client:
        response = client.get("/health", headers={"X-Request-ID": "bad id\twith spaces"})
    assert response.headers["X-Request-ID"] != "bad id\twith spaces"


def test_request_id_on_error_responses(client: TestClient) -> None:
    with client:
        response = client.get("/documents/missing")
    assert response.status_code == 404
    assert "X-Request-ID" in response.headers


def test_access_log_has_request_fields(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="rag_assistant.access")
    with client:
        client.get("/health", headers={"X-Request-ID": "req-1"})
    records = [r for r in caplog.records if r.name == "rag_assistant.access"]
    assert records
    record = records[-1]
    assert record.__dict__["path"] == "/health"
    assert record.__dict__["status"] == 200
    assert record.__dict__["duration_ms"] >= 0


@pytest.mark.parametrize(
    ("incoming", "kept"),
    [("abc", True), ("a" * 64, True), ("a" * 65, False), ("<script>", False), (None, False)],
)
def test_new_request_id(incoming: str | None, kept: bool) -> None:
    assert (new_request_id(incoming) == incoming) is kept


def test_json_formatter_includes_context_and_extras() -> None:
    token = request_id_var.set("rid-42")
    try:
        record = logging.makeLogRecord(
            {"name": "x", "levelname": "INFO", "msg": "hello %s", "args": ("world",)}
        )
        record.chunks = 3
        RequestIdFilter().filter(record)
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["msg"] == "hello world"
    assert payload["request_id"] == "rid-42"
    assert payload["chunks"] == 3


def test_json_formatter_serialises_exceptions() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        record = logging.makeLogRecord({"msg": "failed", "exc_info": sys.exc_info()})
    payload = json.loads(JsonFormatter().format(record))
    assert "RuntimeError: boom" in payload["exc"]


@pytest.mark.parametrize("json_logs", [True, False])
def test_configure_logging_installs_single_handler(json_logs: bool) -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging("debug", json_logs=json_logs)
        configure_logging("debug", json_logs=json_logs)
        assert len(root.handlers) == 1
        assert root.level == logging.DEBUG
        assert isinstance(root.handlers[0].formatter, JsonFormatter) is json_logs
    finally:
        root.handlers, root.level = saved_handlers, saved_level
