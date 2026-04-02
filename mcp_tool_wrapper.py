"""
mcp_tool_wrapper.py — LangChain BaseTool wrapper for MCP server tools.
Bridges MCP tools into the NORT TOOL_REGISTRY so they work identically
to builtin @tool functions with bind_tools() and execute_tool_call().
"""

import logging
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

log = logging.getLogger("nort.mcp_wrapper")


def _schema_to_pydantic(name: str, schema: dict) -> Type[BaseModel]:
    """Convert a JSON schema to a Pydantic model for LangChain args_schema."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields = {}
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        python_type = type_map.get(prop_type, str)
        description = prop_schema.get("description", "")
        if prop_name in required:
            fields[prop_name] = (python_type, Field(description=description))
        else:
            fields[prop_name] = (Optional[python_type], Field(default=None, description=description))

    model_name = f"MCPArgs_{name.replace('.', '_')}"
    return create_model(model_name, **fields)


class MCPTool(BaseTool):
    """LangChain tool wrapping a single MCP server tool."""

    name: str
    description: str
    server_name: str
    mcp_tool_name: str

    def _run(self, **kwargs) -> str:
        from mcp_client import get_mcp_manager
        return get_mcp_manager().call_tool(self.server_name, self.mcp_tool_name, kwargs)


def create_mcp_tools(server_name: str, tool_schemas: list[dict]) -> list[MCPTool]:
    """Convert MCP tool schemas to LangChain MCPTool instances.
    Tools are namespaced as server_name.tool_name."""
    tools = []
    for schema in tool_schemas:
        tool_name = f"{server_name}.{schema['name']}"
        input_schema = schema.get("inputSchema", {"type": "object", "properties": {}})

        tool = MCPTool(
            name=tool_name,
            description=schema.get("description", f"MCP tool: {schema['name']}"),
            server_name=server_name,
            mcp_tool_name=schema["name"],
            args_schema=_schema_to_pydantic(tool_name, input_schema),
        )
        tools.append(tool)
    return tools


def register_mcp_tools_in_registry(registry: dict) -> int:
    """Discover all MCP tools and add them to the given registry dict.
    Returns count of tools registered. Does not overwrite existing keys."""
    from mcp_client import get_mcp_manager

    manager = get_mcp_manager()
    count = 0
    for server_name, tool_schemas in manager.discover_all_tools().items():
        for tool in create_mcp_tools(server_name, tool_schemas):
            if tool.name not in registry:
                registry[tool.name] = tool
                count += 1
                log.info(f"Registered MCP tool: {tool.name}")
    return count
