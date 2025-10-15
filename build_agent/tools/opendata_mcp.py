"""MCP client tool for Boston OpenData.

This wraps the local MCP server (bundled under build_agent.mcp_servers)
and exposes a LangChain tool that the Research supervisor—or any agent—can call.

Usage (by an agent tool call):
    boston_opendata(action="search_datasets", args_json='{"query": "311", "limit": 5}')
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

from langchain.tools import tool


def _repo_root() -> Path:
    # repo root is two levels up from this file: build_agent/tools/opendata_mcp.py
    return Path(__file__).resolve().parents[2]


async def _call_mcp_tool(action: str, arguments: Dict[str, Any]) -> str:
    try:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client
        try:
            # Newer mcp exposes a dataclass for parameters
            from mcp.client.stdio import StdioServerParameters  # type: ignore
        except Exception:
            StdioServerParameters = None  # type: ignore
    except Exception as e:  # ImportError or similar
        return (
            "MCP client not available. Please install 'mcp>=1.2' and try again. "
            f"Details: {e}"
        )

    # Launch the bundled server as a stdio subprocess
    module_path = "build_agent.mcp_servers.boston_opendata.main"
    python_cmd = os.environ.get("PYTHON", None) or os.environ.get("PYTHON_EXE", None) or None
    command = python_cmd if python_cmd else os.environ.get("PYTHON_BIN", None)
    if not command:
        # Default to the current Python interpreter
        command = os.getenv("VIRTUAL_ENV_PYTHON") or os.getenv("PYTHON_PATH") or None
    if not command:
        # Fallback to sys.executable
        import sys
        command = sys.executable

    cwd = str(_repo_root())

    # Use StdioServerParameters when available. Otherwise, fall back to positional args
    # and a temporary chdir for older client versions.
    if StdioServerParameters:  # type: ignore
        params = StdioServerParameters(command=command, args=["-m", module_path], cwd=cwd)  # type: ignore
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

            # Optional: verify tool exists
            # tools = await session.list_tools()
            # names = [t.name for t in tools.tools]
            # if action not in names:
            #     return f"Unknown tool '{action}'. Available: {', '.join(names)}"

            result = await session.call_tool(action, arguments)

            # result is a list of content items; concatenate text segments
            def _iter_text(items: Iterable[Any]) -> Iterable[str]:
                for item in items:
                    # item may be dict-like or object with .type/.text depending on lib version
                    t = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
                    if t == "text":
                        text = getattr(item, "text", None)
                        if text is None and isinstance(item, dict):
                            text = item.get("text")
                        if text:
                            yield str(text)

            return "\n".join(_iter_text(result)) or "(no text response)"
    else:
        prev_cwd = os.getcwd()
        try:
            os.chdir(cwd)
            async with stdio_client(command, ["-m", module_path]) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    result = await session.call_tool(action, arguments)

                    def _iter_text(items: Iterable[Any]) -> Iterable[str]:
                        for item in items:
                            t = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
                            if t == "text":
                                text = getattr(item, "text", None)
                                if text is None and isinstance(item, dict):
                                    text = item.get("text")
                                if text:
                                    yield str(text)

                    return "\n".join(_iter_text(result)) or "(no text response)"
        finally:
            try:
                os.chdir(prev_cwd)
            except Exception:
                pass


async def _direct_ckan_call(action: str, arguments: Dict[str, Any]) -> str:
    # Lazy import to keep tool import light
    from build_agent.mcp_servers.boston_opendata.ckan import ckan_api_call
    from build_agent.mcp_servers.boston_opendata.formatters import (
        format_dataset_summary,
        format_resource_info,
    )
    from build_agent.mcp_servers.boston_opendata.config import MAX_RECORDS
    import json as _json

    try:
        if action == "search_datasets":
            query = arguments["query"]
            limit = arguments.get("limit", 10)
            result = await ckan_api_call("package_search", {"q": query, "rows": limit})
            datasets = result.get("results", [])
            total_count = result.get("count", 0)
            if not datasets:
                return f"No datasets found matching '{query}'."
            output = f"Found {total_count} dataset(s) matching '{query}' (showing {len(datasets)}):\n\n"
            for i, ds in enumerate(datasets, 1):
                output += format_dataset_summary(ds, i) + "\n"
            output += "\nNext: use get_dataset_info with a dataset ID, then query_datastore with a resource ID."
            return output

        if action == "list_all_datasets":
            limit = arguments.get("limit", 20)
            names = await ckan_api_call("package_list", {"limit": limit})
            if not names:
                return "No datasets found on Boston Open Data."
            text = f"Boston Open Data Datasets (showing {len(names)}):\n\n"
            text += "\n".join(f"{i}. `{n}`" for i, n in enumerate(names, 1))
            return text

        if action == "get_dataset_info":
            dataset_id = arguments["dataset_id"]
            ds = await ckan_api_call("package_show", {"id": dataset_id})
            title = ds.get("title", "Untitled Dataset")
            name = ds.get("name", "N/A")
            notes = ds.get("notes", "No description available")
            resources = ds.get("resources", [])
            out = f"{title}\n\nID: `{name}`\nURL: https://data.boston.gov/dataset/{name}\n\nDescription:\n{notes}\n\n"
            out += f"Resources ({len(resources)}):\n\n"
            if resources:
                for i, r in enumerate(resources, 1):
                    out += format_resource_info(r, i) + "\n"
            else:
                out += "No resources available.\n"
            queryable = [r for r in resources if r.get("datastore_active")]
            if queryable:
                out += "\nQueryable Resources:\n" + "\n".join(f"• `{r['id']}` - {r.get('name','')}" for r in queryable)
            return out

        if action == "query_datastore":
            resource_id = arguments["resource_id"]
            limit = min(arguments.get("limit", 10), MAX_RECORDS)
            offset = arguments.get("offset", 0)
            search_text = arguments.get("search_text")
            filters = arguments.get("filters", {})
            sort = arguments.get("sort")
            fields = arguments.get("fields")
            params = {"resource_id": resource_id, "limit": limit, "offset": offset}
            if search_text:
                params["q"] = search_text
            if filters:
                params["filters"] = _json.dumps(filters)
            if sort:
                params["sort"] = sort
            if fields:
                params["fields"] = ",".join(fields)
            result = await ckan_api_call("datastore_search", params)
            records = result.get("records", [])
            total = result.get("total", 0)
            fields_info = result.get("fields", [])
            if not records:
                return "No records found matching your query."
            flds = [f.get("id") for f in fields_info if f.get("id") != "_id"]
            out = f"Query Results\nTotal: {total}\nShowing: {len(records)} (offset {offset})\n\n"
            out += f"Fields: {', '.join(flds[:10])}\n\n"
            for i, rec in enumerate(records[:20], 1):
                out += f"Record {i + offset}:\n"
                for fld in (flds[:8] if not fields else fields[:8]):
                    val = rec.get(fld, "N/A")
                    if isinstance(val, str) and len(val) > 100:
                        val = val[:100] + "..."
                    elif val is None:
                        val = "N/A"
                    out += f"  • {fld}: {val}\n"
                out += "\n"
            if total > (offset + limit):
                out += f"Use offset={offset + limit} for next page."
            return out

        if action == "get_datastore_schema":
            resource_id = arguments["resource_id"]
            res = await ckan_api_call("datastore_search", {"resource_id": resource_id, "limit": 0})
            fields = res.get("fields", [])
            if not fields:
                return "No schema information available for this resource."
            lines = [f"Resource ID: `{resource_id}`", f"Total fields: {len(fields)}", ""]
            for f in fields:
                fid = f.get("id")
                if fid == "_id":
                    continue
                ftype = f.get("type", "unknown")
                lines.append(f"• {fid} ({ftype})")
            return "\n".join(lines)

        return f"Unknown action: {action}"
    except Exception as e:  # defensive catch to avoid crashing the agent
        return f"Direct CKAN fallback failed: {e}"


@tool("boston_opendata")
def boston_opendata(action: str, args_json: str) -> str:
    """Call a Boston OpenData MCP action.

    Parameters:
    - action: One of 'search_datasets', 'list_all_datasets', 'get_dataset_info',
      'get_datastore_schema', 'query_datastore'.
    - args_json: JSON string of the action's arguments. Example:
      '{"query": "311", "limit": 5}' or '{"dataset_id": "311-service-requests"}'.

    Returns a formatted text response from the server.
    """

    try:
        arguments = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        return f"Invalid JSON for args_json: {e}"

    # Try MCP first; if it fails, fall back to direct CKAN calls.
    try:
        return asyncio.run(_call_mcp_tool(action, arguments))
    except Exception as e:
        fallback = asyncio.run(_direct_ckan_call(action, arguments))
        return f"[fallback: MCP failed with {type(e).__name__}]\n\n{fallback}"


__all__ = ["boston_opendata"]
