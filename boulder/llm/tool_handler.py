import json
import logging
import uuid
from typing import Any, Callable

from .utils import parse_tool_arguments

logger = logging.getLogger(__name__)

# (tool_name, tool_args_dict, tool_call_id, result)
ToolCallResult = tuple[str, dict, str, Any]

ToolHandler = Callable[[dict, list[dict]], ToolCallResult]


def create_tool_handler(tools_dict: dict) -> ToolHandler:

    def handle_tool_call(tool_call: dict, messages: list[dict]) -> ToolCallResult:
        tool_id = tool_call.get("id", str(uuid.uuid4()))
        tool_name = tool_call["function"]["name"]
        tool_args = parse_tool_arguments(tool_call["function"]["arguments"])

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args,
                    },
                }
            ],
        })

        if tool_name in tools_dict:
            tool = tools_dict[tool_name]
            try:
                parameters = tool.parameters(**tool_args)
                result = tool(parameters)
            except Exception as e:
                result = {"error": f"Error executing tool {tool_name}: {str(e)}"}
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        messages.append({
            "role": "tool",
            "id": tool_id,
            "content": json.dumps(result),
        })

        return tool_name, tool_args, tool_id, result

    return handle_tool_call
