import os
from abc import ABC, abstractmethod
from typing import Any

import pydantic_core
from langchain_core.tools import ToolException
from mcp import (
    ClientSession,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    StdioServerParameters,
    stdio_client,
)

from langgraph_mcp.utils.openapi_spec import OpenAPISpec
from langgraph_mcp.utils.openapi_utils import openapi_spec_to_openai_fn
from langgraph_mcp.utils.utils import (
    extract_inlined_operation_data,
    merge_json_structure,
)


# Abstract base class for MCP session functions
class MCPSessionFunction(ABC):
    @abstractmethod
    async def __call__(self, server_name: str, session: ClientSession) -> Any:
        pass


class RoutingDescription(MCPSessionFunction):
    async def __call__(self, server_name: str, session: ClientSession) -> str:
        tools: ListToolsResult | None = None
        prompts: ListPromptsResult | None = None
        resources: ListResourcesResult | None = None
        content = ""
        try:
            tools = await session.list_tools()
            if tools:
                content += "Provides tools:\n"
                for tool in tools.tools:
                    content += f"- {tool.name}: {tool.description}\n"
                content += "---\n"
        except Exception as e:
            print(f"Failed to fetch tools from server '{server_name}': {e}")

        try:
            prompts = await session.list_prompts()
            if prompts:
                content += "Provides prompts:\n"
                for prompt in prompts.prompts:
                    content += f"- {prompt.name}: {prompt.description}\n"
                content += "---\n"
        except Exception as e:
            print(f"Failed to fetch prompts from server '{server_name}': {e}")

        try:
            resources = await session.list_resources()
            if resources:
                content += "Provides resources:\n"
                for resource in resources.resources:
                    content += f"- {resource.name}: {resource.description}\n"
                content += "---\n"
        except Exception as e:
            print(f"Failed to fetch resources from server '{server_name}': {e}")

        return server_name, content


class GetTools(MCPSessionFunction):
    async def __call__(
        self, server_name: str, session: ClientSession
    ) -> list[dict[str, Any]]:
        tools = await session.list_tools()
        if tools is None:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {},
                },
            }
            for tool in tools.tools
        ]


class GetOpenAPITools(GetTools):
    def __init__(self, openapi_spec: dict, **kwargs):
        self.openapi_spec_dict = openapi_spec
        self.openapi_spec = OpenAPISpec.from_spec_dict(self.openapi_spec_dict)

    async def __call__(
        self, server_name: str, session: ClientSession
    ) -> list[dict[str, Any]]:
        tools = await session.list_tools()
        openai_fns = openapi_spec_to_openai_fn(self.openapi_spec)[0]

        for fn, tool in zip(openai_fns, tools.tools):
            op_id = "".join(tool.name.split("-")[1:])
            fn["metadata"] = {}
            fn["metadata"]["tool_info"] = extract_inlined_operation_data(
                self.openapi_spec_dict, op_id
            )

            fn["parameters"]["properties"] = merge_json_structure(
                fn["parameters"]["properties"]
            )
            fn["name"] = tool.name
            # truncate all openai_fn descriptions to 1024 characters
            if len(fn["description"]) > 1024:
                fn["description"] = fn["description"][:1021] + "..."

        return openai_fns


class RunTool(MCPSessionFunction):
    def __init__(self, tool_name: str, **kwargs):
        self.tool_name = tool_name
        self.kwargs = kwargs

    async def __call__(
        self,
        server_name: str,
        session: ClientSession,
    ) -> Any:
        result = await session.call_tool(self.tool_name, arguments=self.kwargs)
        content = pydantic_core.to_json(result.content).decode()
        if result.isError:
            raise ToolException(content)
        return content


async def apply(server_name: str, server_config: dict, fn: MCPSessionFunction) -> Any:
    server_params = StdioServerParameters(
        command=server_config["command"],
        args=server_config["args"],
        env={**os.environ, **(server_config.get("env") or {})},
    )
    print(f"Starting session with (server: {server_name})")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(server_name, session)
