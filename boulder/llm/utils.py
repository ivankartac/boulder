import json
import logging

logger = logging.getLogger(__name__)


def parse_tool_arguments(args: str | dict) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse tool arguments as JSON: %s (error: %s)", args, e)
            return {}
    logger.warning("Tool call arguments is neither string nor dict: %s", type(args))
    return {}


def validate_and_sanitize_tool_calls(messages: list[dict]) -> list[dict]:
    sanitized_messages = []

    for message in messages:
        message_copy = message.copy()

        if message_copy.get("role") == "assistant" and "tool_calls" in message_copy:
            sanitized_tool_calls = []

            for tool_call in message_copy["tool_calls"]:
                tool_call_copy = tool_call.copy()

                if "function" in tool_call_copy:
                    function_copy = tool_call_copy["function"].copy()
                    function_copy["arguments"] = parse_tool_arguments(
                        function_copy.get("arguments", {})
                    )
                    tool_call_copy["function"] = function_copy

                sanitized_tool_calls.append(tool_call_copy)

            message_copy["tool_calls"] = sanitized_tool_calls

        sanitized_messages.append(message_copy)

    return sanitized_messages


def convert_to_openrouter_format(messages: list[dict]) -> list[dict]:
    converted = []
    for msg in messages:
        msg_copy = msg.copy()

        if msg_copy.get("role") == "assistant" and "tool_calls" in msg_copy:
            tool_calls_copy = []
            for tool_call in msg_copy["tool_calls"]:
                tool_call_copy = tool_call.copy()
                if "function" in tool_call_copy:
                    function_copy = tool_call_copy["function"].copy()
                    if isinstance(function_copy.get("arguments"), dict):
                        function_copy["arguments"] = json.dumps(function_copy["arguments"])
                    tool_call_copy["function"] = function_copy
                tool_calls_copy.append(tool_call_copy)
            msg_copy["tool_calls"] = tool_calls_copy

        elif msg_copy.get("role") == "tool":
            if "id" in msg_copy and "tool_call_id" not in msg_copy:
                msg_copy["tool_call_id"] = msg_copy.pop("id")

        converted.append(msg_copy)
    return converted
