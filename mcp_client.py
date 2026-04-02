"""
mcp_client.py — MCP server lifecycle manager for NORT agents.
Lazily connects to configured MCP servers, discovers tools, and bridges
async MCP calls to sync LangChain tool invocations.
"""

import asyncio
import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("nort.mcp")

# ── MCP imports (deferred to avoid hard dependency) ─────────────────────────

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


# ── MCP Client Manager ─────────────────────────────────────────────────────

class MCPClientManager:
    """Manages connections to MCP servers. Thread-safe, lazy-connecting."""

    def __init__(self):
        self._configs: dict[str, dict] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._transports: dict[str, tuple] = {}  # read/write streams
        self._locks: dict[str, threading.Lock] = {}
        self._loops: dict[str, asyncio.AbstractEventLoop] = {}
        self._global_lock = threading.Lock()
        self._tool_cache: dict[str, list[dict]] = {}

    def load_config(self, mcp_servers: dict) -> None:
        """Parse mcp_servers config dict. Does NOT connect."""
        with self._global_lock:
            for name, cfg in mcp_servers.items():
                self._configs[name] = cfg
                self._locks[name] = threading.Lock()
        if self._configs:
            log.info(f"MCP: loaded {len(self._configs)} server config(s): {list(self._configs.keys())}")

    def _resolve_env(self, env_dict: dict[str, str]) -> dict[str, str]:
        """Expand ${VAR} references from os.environ."""
        resolved = {}
        for k, v in env_dict.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                var_name = v[2:-1]
                resolved[k] = os.environ.get(var_name, "")
            else:
                resolved[k] = v
        return resolved

    def _ensure_loop(self, server_name: str) -> asyncio.AbstractEventLoop:
        """Get or create a background event loop thread for this server."""
        if server_name in self._loops:
            return self._loops[server_name]
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            daemon=True,
            name=f"mcp-loop-{server_name}",
        )
        thread.start()
        self._loops[server_name] = loop
        return loop

    def _connect_server(self, server_name: str) -> ClientSession:
        """Connect to an MCP server. Must be called within the server's lock."""
        if server_name in self._sessions:
            return self._sessions[server_name]

        cfg = self._configs.get(server_name)
        if not cfg:
            raise ValueError(f"No MCP server config for '{server_name}'")

        loop = self._ensure_loop(server_name)
        server_type = cfg.get("type", "stdio")

        async def _do_connect():
            if server_type == "stdio":
                env = {**os.environ, **self._resolve_env(cfg.get("env", {}))}
                params = StdioServerParameters(
                    command=cfg["command"],
                    args=cfg.get("args", []),
                    env=env,
                )
                cm = stdio_client(params)
            elif server_type in ("sse", "http"):
                cm = sse_client(cfg["url"])
            else:
                raise ValueError(f"Unknown MCP server type: {server_type}")

            streams = await cm.__aenter__()
            read_stream, write_stream = streams
            session = ClientSession(read_stream, write_stream)
            await session.initialize()
            return session, cm

        future = asyncio.run_coroutine_threadsafe(_do_connect(), loop)
        session, cm = future.result(timeout=30)

        self._sessions[server_name] = session
        self._transports[server_name] = cm
        log.info(f"MCP: connected to '{server_name}' ({server_type})")
        return session

    def _get_session(self, server_name: str) -> ClientSession:
        """Get or create a session. Thread-safe via per-server lock."""
        if server_name in self._sessions:
            return self._sessions[server_name]
        lock = self._locks.get(server_name)
        if not lock:
            raise ValueError(f"No MCP server config for '{server_name}'")
        with lock:
            return self._connect_server(server_name)

    def discover_tools(self, server_name: str) -> list[dict]:
        """Discover tools from a single MCP server. Returns list of tool dicts."""
        if server_name in self._tool_cache:
            return self._tool_cache[server_name]

        try:
            session = self._get_session(server_name)
            loop = self._loops[server_name]

            async def _list():
                result = await session.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": t.description or f"MCP tool: {t.name}",
                        "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else {"type": "object", "properties": {}},
                    }
                    for t in result.tools
                ]

            future = asyncio.run_coroutine_threadsafe(_list(), loop)
            tools = future.result(timeout=30)
            self._tool_cache[server_name] = tools
            log.info(f"MCP: discovered {len(tools)} tools from '{server_name}'")
            return tools

        except Exception as e:
            log.warning(f"MCP: failed to discover tools from '{server_name}': {e}")
            return []

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """Execute an MCP tool. Bridges async to sync. Returns result as string."""
        try:
            session = self._get_session(server_name)
            loop = self._loops[server_name]

            async def _call():
                result = await session.call_tool(tool_name, arguments)
                texts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        texts.append(item.text)
                    else:
                        texts.append(str(item))
                return "\n".join(texts)

            future = asyncio.run_coroutine_threadsafe(_call(), loop)
            return future.result(timeout=30)

        except Exception as e:
            # Attempt one reconnect
            try:
                log.warning(f"MCP: tool call failed, reconnecting '{server_name}': {e}")
                self._sessions.pop(server_name, None)
                self._tool_cache.pop(server_name, None)
                session = self._get_session(server_name)
                loop = self._loops[server_name]

                async def _retry():
                    result = await session.call_tool(tool_name, arguments)
                    texts = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            texts.append(item.text)
                        else:
                            texts.append(str(item))
                    return "\n".join(texts)

                future = asyncio.run_coroutine_threadsafe(_retry(), loop)
                return future.result(timeout=30)
            except Exception as e2:
                return f"MCP tool error ({server_name}.{tool_name}): {e2}"

    def discover_all_tools(self) -> dict[str, list[dict]]:
        """Discover tools from all configured servers."""
        result = {}
        for name in list(self._configs.keys()):
            tools = self.discover_tools(name)
            if tools:
                result[name] = tools
        return result

    def shutdown(self) -> None:
        """Close all sessions and stop event loops."""
        for name, loop in list(self._loops.items()):
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        self._sessions.clear()
        self._transports.clear()
        self._loops.clear()
        self._tool_cache.clear()
        log.info("MCP: shutdown complete")


# ── Module-level singleton ──────────────────────────────────────────────────

_manager: MCPClientManager | None = None


def get_mcp_manager() -> MCPClientManager:
    """Return the singleton MCPClientManager."""
    global _manager
    if _manager is None:
        _manager = MCPClientManager()
    return _manager


def init_mcp_from_config() -> MCPClientManager:
    """Load config.json, parse mcp_servers, return initialized manager."""
    if not MCP_AVAILABLE:
        log.info("MCP: mcp package not installed, skipping")
        return get_mcp_manager()

    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}

    mcp_servers = cfg.get("mcp_servers", {})
    if not mcp_servers:
        return get_mcp_manager()

    manager = get_mcp_manager()
    manager.load_config(mcp_servers)
    return manager


def shutdown_mcp() -> None:
    """Shutdown the singleton manager if it exists."""
    global _manager
    if _manager:
        _manager.shutdown()
        _manager = None
