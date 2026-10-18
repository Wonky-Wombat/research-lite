#
# mcp_protocol.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-10-18.
#

"""JSON-RPC over stdin and stdout for an MCP host."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

MODERN = "2026-07-28"
LEGACY = "2025-11-25"
SUPPORTED = (MODERN, LEGACY, "2025-06-18", "2025-03-26")
META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNSUPPORTED_VERSION = -32022

# JSON-RPC allows a string, a number or null; a notification carries none at all
RequestId = str | int | float | None

Handler = "Callable[[Mapping[str, Any]], Mapping[str, Any]]"


def requested_version(params: Mapping[str, Any]) -> str | None:
    """The protocol version a modern request declares, or None for a legacy one."""
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        return None
    version = meta.get(META_VERSION)
    return version if isinstance(version, str) else None


def error(
    request_id: RequestId, code: int, message: str, data: object | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        body["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": body}


def unsupported_version(request_id: RequestId, requested: str) -> dict[str, Any]:
    return error(
        request_id,
        UNSUPPORTED_VERSION,
        "Unsupported protocol version",
        {"supported": list(SUPPORTED), "requested": requested},
    )


class Server:
    """Dispatch JSON-RPC messages to handlers, tracking which era the client opened with."""

    def __init__(self, name: str, version: str, tools: Mapping[str, Any]) -> None:
        self.name = name
        self.version = version
        self.tools = tools
        self.era = MODERN

    def _info(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}

    def discover(self, _: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "resultType": "complete",
            "supportedVersions": list(SUPPORTED),
            "capabilities": {"tools": {}},
            "_meta": {META_SERVER_INFO: self._info()},
            "instructions": (
                "Searches one folder of the user's own documents. `search` returns a few "
                "passages rather than whole files; widen with `read_context` only when a "
                "passage is cut off. `overview` answers questions about the collection "
                "itself, such as how many documents mention a word."
            ),
        }

    def initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The legacy opening. Answering it fixes this connection to legacy semantics."""
        self.era = LEGACY
        asked = params.get("protocolVersion")
        agreed = asked if isinstance(asked, str) and asked in SUPPORTED else LEGACY
        return {
            "protocolVersion": agreed,
            "capabilities": {"tools": {}},
            "serverInfo": self._info(),
        }

    def _complete(self, result: dict[str, Any]) -> dict[str, Any]:
        """Modern results are tagged; legacy ones must not be."""
        if self.era == MODERN:
            result.setdefault("resultType", "complete")
        else:
            result.pop("resultType", None)
        return result

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        """Answer one message, or return None for a notification, which gets no reply."""
        request_id: RequestId = message.get("id")
        method = message.get("method")
        if not isinstance(method, str):
            return error(request_id, INVALID_REQUEST, "A request must name a method.")
        params = message.get("params")
        if not isinstance(params, dict):
            params = {}
        notification = "id" not in message

        version = requested_version(params)
        if version is not None and version not in SUPPORTED:
            return None if notification else unsupported_version(request_id, version)

        if method.startswith("notifications/") or notification:
            return None
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": request_id, "result": self.initialize(params)}
        if method == "server/discover":
            return {"jsonrpc": "2.0", "id": request_id, "result": self.discover(params)}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": self._complete({})}
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": self._complete({"tools": list(self.tools["definitions"])}),
            }
        if method == "tools/call":
            return self._call(request_id, params)
        return error(request_id, METHOD_NOT_FOUND, f"Unknown method: {method}")

    def _call(self, request_id: RequestId, params: Mapping[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        handlers = self.tools["handlers"]
        if not isinstance(name, str) or name not in handlers:
            return error(request_id, INVALID_PARAMS, f"Unknown tool: {name}")
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        try:
            text, structured = handlers[name](arguments)
        except Exception as exc:  # noqa: BLE001 - the model can act on a stated failure
            # a tool that fails is a result the model can read and retry, not a protocol
            # fault; a JSON-RPC error here would reach the user instead of the model
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": self._complete(
                    {
                        "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                        "isError": True,
                    }
                ),
            }
        result: dict[str, Any] = {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        }
        if structured is not None:
            result["structuredContent"] = structured
        return {"jsonrpc": "2.0", "id": request_id, "result": self._complete(result)}


def serve(server: Server, stdin: Iterable[str], write: Callable[[str], None]) -> None:
    """Read messages until the stream closes, writing each reply as one line."""
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            write(json.dumps(error(None, PARSE_ERROR, "Invalid JSON.")))
            continue
        if not isinstance(message, dict):
            write(json.dumps(error(None, INVALID_REQUEST, "A request must be an object.")))
            continue
        try:
            reply = server.handle(message)
        except Exception as exc:  # noqa: BLE001 - one bad message must not end the session
            reply = error(message.get("id"), INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        if reply is not None:
            write(json.dumps(reply, ensure_ascii=False))


def main_stream(server: Server) -> None:
    def write(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    serve(server, sys.stdin, write)


__all__ = ["LEGACY", "MODERN", "SUPPORTED", "Server", "main_stream", "serve"]
