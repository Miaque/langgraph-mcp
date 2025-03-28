import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from langgraph_mcp import mcp_wrapper as mcp
from langgraph_mcp.configuration import Configuration
from langgraph_mcp.retriever import make_retriever
from langgraph_mcp.state import InputState, State
from langgraph_mcp.utils.utils import format_docs, get_message_text, load_chat_model

logger = logging.getLogger(__name__)

NOTHING_RELEVANT = "No MCP server with an appropriate tool to address current context"  # When available MCP servers seem to be irrelevant for the query
IDK_RESPONSE = "No appropriate tool available."  # Default response where the current MCP Server can't help
OTHER_SERVERS_MORE_RELEVANT = "Other servers are more relevant."  # Default response when other servers are more relevant than the currnet one
AMBIGUITY_PREFIX = (
    "Ambiguity:"  # Prefix to indicate ambiguity when asking the user for clarification
)

##################  MCP Server Router: Sub-graph Components  ###################


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


async def generate_routing_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a routing query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): Configuration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """
    messages = state.messages
    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = get_message_text(messages[-1])
        return {"queries": [human_input]}
    else:
        configuration = Configuration.from_runnable_config(config)
        # Feel free to customize the prompt, model, and other logic!
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.routing_query_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
        model = load_chat_model(
            configuration.routing_query_model
        ).with_structured_output(SearchQuery)

        message_value = await prompt.ainvoke(
            {
                "messages": state.messages,
                "queries": "\n- ".join(state.queries),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        generated = cast(SearchQuery, await model.ainvoke(message_value, config))
        return {
            "queries": [generated.query],
        }


async def retrieve(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    with make_retriever(config) as retriever:
        response = await retriever.ainvoke(state.queries[-1], config)
        return {"retrieved_docs": response}


async def route(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""
    configuration = Configuration.from_runnable_config(config)
    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.routing_response_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.routing_response_model)

    retrieved_docs = format_docs(state.retrieved_docs)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "retrieved_docs": retrieved_docs,
            "nothing_relevant": NOTHING_RELEVANT,
            "ambiguity_prefix": AMBIGUITY_PREFIX,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )

    response = await model.ainvoke(message_value, config)

    if response.content == NOTHING_RELEVANT or response.content.startswith(
        AMBIGUITY_PREFIX
    ):
        # No relevant server found or ambiguity in the response
        return {"messages": [response]}

    mcp_server = (
        response.content.split(":")[1].strip()
        if ":" in response.content
        else response.content
    )

    # if len(mcp_server.split(" ")) > 1:
    #     # Likely a clarification. model has not adhered to the prompt instructions
    #     return {"messages": [response]}

    return {"current_mcp_server": mcp_server}


def decide_mcp_or_not(state: State) -> str:
    """Decide whether to route to MCP server processing or not"""
    if state.current_mcp_server:
        return "mcp_orchestrator"
    return END


##################  MCP Server Router: Sub-graph Components  ###################


async def mcp_orchestrator(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Orchestrates MCP server processing."""
    # Fetch the current MCP server from state
    server_name = state.current_mcp_server

    # Fetch mcp server config
    configuration = Configuration.from_runnable_config(config)
    mcp_servers = configuration.mcp_server_config["mcpServers"]
    server_config = mcp_servers[server_name]

    def list_other_servers(servers: list[tuple[str, str]], current_server: str) -> str:
        """
        Generates a description listing all servers except the current one.

        Args:
            servers (list[tuple[str, str]]): A list of tuples where each tuple contains a server name and its description.
            current_server (str): The name of the current server to exclude from the list.

        Returns:
            str: A formatted string listing the other servers and their descriptions.
        """
        return "\n".join(
            f"- {name}: {description}"
            for name, description in servers
            if name != current_server
        )

    # Fetch tools from the MCP server
    tools = []
    args = (
        server_config["args"][1:]
        if server_config["args"][0] == "-y"
        else server_config["args"]
    )
    # Separate integration for openapi-mcp-server@1.1.0
    if args[0] == "openapi-mcp-server@1.1.0":
        openapi_path = args[1]

        # TODO: refactor
        # Get the openapi file as a json
        with open(openapi_path, "r") as file:
            openapi_spec = json.load(file)  # Converts JSON to a Python dictionary

        # convert the spec to openai tools
        tools = await mcp.apply(
            server_name,
            server_config,
            mcp.GetOpenAPITools(openapi_spec),
        )
    else:
        tools = await mcp.apply(server_name, server_config, mcp.GetTools())

    # Prepare the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.mcp_orchestrator_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.mcp_orchestrator_model)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "idk_response": IDK_RESPONSE,
            "other_servers": list_other_servers(
                configuration.get_mcp_server_descriptions(), current_server=server_name
            ),
            "other_servers_response": OTHER_SERVERS_MORE_RELEVANT,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )

    # Bind tools to model and invoke
    response = await model.bind_tools(tools).ainvoke(message_value, config)

    # If the model has an AI response with a tool_call, find the selected tool
    current_tool = None
    if response.__class__ == AIMessage and response.tool_calls:
        current_tool = next(
            (
                tool
                for tool in tools
                if tool.get("function", {})["name"]
                == response.tool_calls[0].get("name")
            ),
            None,
        )

    if (
        response.content == IDK_RESPONSE
        or response.content == OTHER_SERVERS_MORE_RELEVANT
    ):
        """model doesn't know how to proceed"""
        if state.messages[-1].__class__ != ToolMessage:
            """and this is not immediately after a tool call response"""
            # let's setup for routing again
            return {"current_mcp_server": None}

    return {"messages": [response], "current_tool": current_tool}


async def refine_tool_call(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the MCP server tool."""

    if state.current_tool == None:
        return

    # Fetch the current MCP server from state
    server_name = state.current_mcp_server

    # Fetch mcp server config
    configuration = Configuration.from_runnable_config(config)
    mcp_servers = configuration.mcp_server_config["mcpServers"]
    server_config = mcp_servers[server_name]

    # Get the tool info
    tool_info = state.current_tool.get("metadata", {}).get("tool_info", {})

    # Bind the tool call to the model
    # Prepare the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.tool_refiner_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.tool_refiner_model)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages[:-1],
            "tool_info": str(tool_info),
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )

    # get the last response id
    last_msg_id = state.messages[-1].id

    # Bind tools to model and invoke
    response = await model.bind_tools([state.current_tool]).ainvoke(
        message_value, config
    )
    response.id = last_msg_id

    return {"messages": [response], "current_tool": None}


async def mcp_tool_call(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the MCP server tool."""
    # Fetch the current MCP server from state
    server_name = state.current_mcp_server

    # Fetch mcp server config
    configuration = Configuration.from_runnable_config(config)
    mcp_servers = configuration.mcp_server_config["mcpServers"]
    server_config = mcp_servers[server_name]

    # Execute MCP server Tool
    tool_call = state.messages[-1].tool_calls[0]
    try:
        tool_output = await mcp.apply(
            server_name,
            server_config,
            mcp.RunTool(tool_call["name"], **tool_call["args"]),
        )
    except Exception as e:
        tool_output = f"Error: {e}"
    return {
        "messages": [ToolMessage(content=tool_output, tool_call_id=tool_call["id"])]
    }


def route_tools(state: State) -> str:
    """
    Route to the mcp_tool_call if last message has tool calls.
    Otherwise, route to the END.
    """
    last_message = state.messages[-1]

    if last_message.__class__ == HumanMessage:
        return "generate_routing_query"
    if last_message.model_dump().get("tool_calls"):  # suggests tool calls
        return "refine_tool_call"
    if (
        last_message.__class__ == ToolMessage
    ):  # re-routing. todo: check if HITL makes more sense?
        return "generate_routing_query"
    return END


#############################  Subgraph decider  ###############################
def decide_subgraph(state: State) -> str:
    """
    Route to MCP Server Router sub-graph if there is no state.current_mcp_server
    else route to MCP Server processing sub-graph.
    """
    if state.current_mcp_server:
        return "mcp_orchestrator"
    return "generate_routing_query"


##################################  Wiring  ####################################

builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node(generate_routing_query)
builder.add_node(retrieve)
builder.add_node(route)
builder.add_node(mcp_orchestrator)
builder.add_node(refine_tool_call)
builder.add_node(mcp_tool_call)

builder.add_conditional_edges(
    START,
    decide_subgraph,
    {
        "generate_routing_query": "generate_routing_query",
        "mcp_orchestrator": "mcp_orchestrator",
    },
)
builder.add_edge("generate_routing_query", "retrieve")
builder.add_edge("retrieve", "route")
builder.add_conditional_edges(
    "route",
    decide_mcp_or_not,
    {
        "mcp_orchestrator": "mcp_orchestrator",
        END: END,
    },
)
builder.add_conditional_edges(
    "mcp_orchestrator",
    route_tools,
    {
        "mcp_tool_call": "mcp_tool_call",
        "generate_routing_query": "generate_routing_query",
        "refine_tool_call": "refine_tool_call",
        END: END,
    },
)
builder.add_edge("refine_tool_call", "mcp_tool_call")
builder.add_edge("mcp_tool_call", "mcp_orchestrator")
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = "AssistantGraph"

if __name__ == "__main__":
    try:
        with open(
            "E:\\workspace\\python\\langgraph-mcp\\mcp-servers-config.json",
            "r",
            encoding="utf-8",
        ) as f:
            mcp_tools = json.loads(f.read())
        result = asyncio.run(
            graph.ainvoke(
                input={"messages": [HumanMessage(content="访问www.baidu.com，搜索今天厦门的天气")]},
                config={"configurable": {"mcp_server_config": mcp_tools}},
            )
        )

        print(result)
    except Exception as e:
        logger.exception("异常", exc_info=e)
