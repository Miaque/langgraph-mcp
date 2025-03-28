"""Utility functions for the retrieval graph.

This module contains utility functions for handling messages, documents,
and other common operations in project.

Functions:
    get_message_text: Extract text content from various message formats.
    format_docs: Convert documents to an xml-formatted string.
"""

import copy
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage


def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message.

    This function extracts the text content from various message formats.

    Args:
        msg (AnyMessage): The message object to extract text from.

    Returns:
        str: The extracted text content of the message.

    Examples:
        >>> from langchain_core.messages import HumanMessage
        >>> get_message_text(HumanMessage(content="Hello"))
        'Hello'
        >>> get_message_text(HumanMessage(content={"text": "World"}))
        'World'
        >>> get_message_text(HumanMessage(content=[{"text": "Hello"}, " ", {"text": "World"}]))
        'Hello World'
    """
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    return init_chat_model(
        "glm-4-plus",
        model_provider="openai",
        base_url="xxxx",
        api_key="xxxx",
    )


def inline_refs(node, root, visited=None):
    """Recursively walk through 'node' and inline any $ref in 'root',
    stopping if we revisit a reference to avoid infinite recursion."""
    if visited is None:
        visited = set()

    if isinstance(node, list):
        return [inline_refs(item, root, visited) for item in node]

    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref = node["$ref"]

        # If we’ve already encountered this $ref, return an empty object
        # (or the raw $ref) to avoid recursion loops.
        if ref in visited:
            return {}  # or just return node to leave the ref as-is

        visited.add(ref)

        if ref.startswith("#/"):
            parts = ref.lstrip("#/").split("/")
            target = root
            for p in parts:
                if p not in target:
                    raise ValueError(f"Could not find reference: {ref}")
                target = target[p]
            return inline_refs(target, root, visited)
        else:
            # External references or non-#/ references
            # might need custom handling
            raise NotImplementedError(f"External ref not supported: {ref}")

    # Recurse into all dict keys
    result = {}
    for k, v in node.items():
        result[k] = inline_refs(v, root, visited)
    return result


def inline_operation(openapi_spec, path, method):
    """Inline references in just one operation (path + method),
    avoiding recursion loops by skipping already visited refs."""
    spec_copy = copy.deepcopy(openapi_spec)
    method = method.lower()

    try:
        operation = spec_copy["paths"][path][method]
    except KeyError:
        raise ValueError(f"No such operation: {method.upper()} {path}")

    # Inline requestBody schemas
    rb = operation.get("requestBody", {})
    for media_type, media_obj in rb.get("content", {}).items():
        if "schema" in media_obj:
            media_obj["schema"] = inline_refs(media_obj["schema"], spec_copy)

    # Inline parameter schemas
    if "parameters" in operation:
        for param in operation["parameters"]:
            if "schema" in param:
                param["schema"] = inline_refs(param["schema"], spec_copy)

    # Inline response schemas
    for response_code, resp_obj in operation.get("responses", {}).items():
        for media_type, media_obj in resp_obj.get("content", {}).items():
            if "schema" in media_obj:
                media_obj["schema"] = inline_refs(media_obj["schema"], spec_copy)

    return spec_copy


def find_path_from_operation_id(openapi_spec, operation_id):
    """Find the path and method of an operation with the given 'operation_id'."""
    paths = openapi_spec.get("paths", {})
    for path, path_item in paths.items():
        if isinstance(path_item, dict):
            for method, operation_obj in path_item.items():
                if (
                    isinstance(operation_obj, dict)
                    and operation_obj.get("operationId") == operation_id
                ):
                    return path, method
    return None, None


def extract_inlined_operation_data(openapi_spec, operation_id):
    """
    Find the operation with the given 'operation_id' in 'openapi_spec',
    and return ONLY a minimal dict containing:
      {
        "parameters": ...,
        "requestBody": ...
      }
    with all $ref inlined (cycle-safe).
    """
    # Copy the spec so references can be resolved from a stable root
    spec_copy = copy.deepcopy(openapi_spec)

    # 1. Locate the operation by scanning all paths/methods
    found_operation = None
    paths = spec_copy.get("paths", {})
    for path, path_item in paths.items():
        if isinstance(path_item, dict):
            for method, operation_obj in path_item.items():
                if (
                    isinstance(operation_obj, dict)
                    and operation_obj.get("operationId") == operation_id
                ):
                    found_operation = operation_obj
                    break
        if found_operation:
            break

    if not found_operation:
        raise ValueError(f"No operation found with operationId: {operation_id}")

    # 2. Extract just parameters + requestBody
    op_data = {}
    if "parameters" in found_operation:
        op_data["parameters"] = inline_refs(found_operation["parameters"], spec_copy)
    if "requestBody" in found_operation:
        op_data["requestBody"] = inline_refs(found_operation["requestBody"], spec_copy)
    # if 'responses' in found_operation:
    #     op_data['responses'] = inline_refs(found_operation['responses'], spec_copy)

    # This is all we return: only the pieces we care about, fully inlined.
    return op_data


def merge_json_structure(data):
    params = data.get("params", {})
    json_data = data.get("json", {})

    # Extract properties from 'params' and 'json'
    params_properties = params.get("properties", {})
    json_properties = json_data.get("properties", {})

    # Merge required fields (if any)
    # params_required = set(params.get("required", []))
    # json_required = set(json_data.get("required", []))
    # merged_required = (
    #     list(params_required.union(json_required))
    #     if params_required or json_required
    #     else None
    # )

    # Merge properties
    merged_properties = {
        **params_properties,
        **json_properties,
        # "required": merged_required,
    }

    return merged_properties
