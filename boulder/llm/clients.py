import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import ollama
import requests

from .utils import (
    convert_to_openrouter_format,
    parse_tool_arguments,
    validate_and_sanitize_tool_calls,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    content: str | None
    reasoning: str | None
    tool_calls_made: list[dict] = field(default_factory=list)


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host

    def list_models(self) -> list[str]:
        client = ollama.Client(host=self.host)
        return [m.model for m in client.list()["models"]]

    def chat(
        self,
        model: str,
        messages: list[dict],
        think: bool = False,
        options: dict | None = None,
        tool_schemas: list[dict] | None = None,
        tool_handler: Callable | None = None,
        max_tool_iterations: int = 10,
    ) -> LLMResult:
        client = ollama.Client(host=self.host)
        messages = list(messages)

        chat_params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "think": think,
            "options": options,
        }
        if tool_schemas:
            chat_params["tools"] = tool_schemas

        tool_calls_made: list[dict] = []

        sanitized_messages = validate_and_sanitize_tool_calls(chat_params["messages"])
        chat_params_sanitized = chat_params.copy()
        chat_params_sanitized["messages"] = sanitized_messages

        response = client.chat(**chat_params_sanitized)

        iteration = 0
        while (
            "tool_calls" in response["message"]
            and response["message"]["tool_calls"]
            and iteration < max_tool_iterations
        ):
            iteration += 1

            for tool_call in response["message"]["tool_calls"]:
                if tool_handler:
                    tool_name, tool_args, tool_id, result = tool_handler(
                        tool_call, messages
                    )
                else:
                    tool_id = tool_call.get("id", str(uuid.uuid4()))
                    tool_name = tool_call["function"]["name"]
                    tool_args = parse_tool_arguments(
                        tool_call["function"]["arguments"]
                    )

                    result = {
                        "error": f"No tool handler provided for tool: {tool_name}"
                    }

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

                    messages.append({
                        "role": "tool",
                        "id": tool_id,
                        "content": json.dumps(result),
                    })

                tool_calls_made.append({
                    "id": tool_id,
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": result,
                })

            chat_params["messages"] = messages

            sanitized_messages = validate_and_sanitize_tool_calls(
                chat_params["messages"]
            )
            chat_params_sanitized = chat_params.copy()
            chat_params_sanitized["messages"] = sanitized_messages

            response = client.chat(**chat_params_sanitized)

        return LLMResult(
            content=response["message"]["content"],
            reasoning=response["message"].get("thinking", None),
            tool_calls_made=tool_calls_made,
        )


class OpenRouterClient:
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if self.api_key is None:
            raise ValueError(
                "OpenRouter API key is not set. "
                "Set OPENROUTER_API_KEY environment variable."
            )

    def chat(
        self,
        model: str,
        messages: list[dict],
        think: bool = False,
        options: dict | None = None,
        tool_schemas: list[dict] | None = None,
        tool_handler: Callable | None = None,
        max_tool_iterations: int = 10,
    ) -> LLMResult:
        messages = list(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if "deepseek" in model.lower():
            payload["provider"] = {
                "order": ["atlas-cloud/fp8"],
                "allow_fallbacks": False,
            }

        if options:
            if "temperature" in options:
                payload["temperature"] = options["temperature"]
            if "num_predict" in options or "max_tokens" in options:
                payload["max_tokens"] = options.get(
                    "max_tokens", options.get("num_predict", 8192)
                )

        if think:
            payload["reasoning"] = {"enabled": True}
        else:
            payload["reasoning"] = {"enabled": False, "effort": "none"}

        logger.debug("OpenRouter payload: %s", payload)

        if tool_schemas:
            payload["tools"] = tool_schemas

        tool_calls_made: list[dict] = []

        response_json = self._make_api_call(payload)

        if response_json is None:
            return LLMResult(content="", reasoning=None)

        if "choices" not in response_json or not response_json["choices"]:
            self._log_error(response_json)
            return LLMResult(content=None, reasoning=None)

        message = response_json["choices"][0]["message"]

        # Handle tool calls if present
        iteration = 0
        while (
            "tool_calls" in message
            and message["tool_calls"]
            and iteration < max_tool_iterations
        ):
            iteration += 1

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": message["tool_calls"],
            }
            if "gemini" in model.lower():
                if "reasoning_details" in message:
                    assistant_message["reasoning_details"] = message[
                        "reasoning_details"
                    ]
                elif "reasoning" in message:
                    assistant_message["reasoning_details"] = message["reasoning"]
            messages.append(assistant_message)

            for tool_call in message["tool_calls"]:
                tool_id = tool_call.get("id", str(uuid.uuid4()))
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]

                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        pass

                if tool_handler:
                    temp_messages: list[dict] = []
                    tool_name, tool_args, tool_id, result = tool_handler(
                        tool_call, temp_messages
                    )

                    for msg in temp_messages:
                        if msg["role"] == "tool":
                            if "id" in msg and "tool_call_id" not in msg:
                                msg["tool_call_id"] = msg.pop("id")
                            messages.append(msg)
                else:
                    result = {
                        "error": f"No tool handler provided for tool: {tool_name}"
                    }
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": json.dumps(result),
                    })

                tool_calls_made.append({
                    "id": tool_id,
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": result,
                })

            payload["messages"] = messages

            response_json = self._make_api_call(payload)

            if response_json is None:
                return LLMResult(
                    content="", reasoning=None, tool_calls_made=tool_calls_made
                )

            if "choices" not in response_json or not response_json["choices"]:
                self._log_error(response_json, during_tool_iteration=True)
                return LLMResult(
                    content=None, reasoning=None, tool_calls_made=tool_calls_made
                )

            message = response_json["choices"][0]["message"]

        return LLMResult(
            content=message["content"],
            reasoning=message.get("reasoning", None),
            tool_calls_made=tool_calls_made,
        )

    def _make_api_call(self, payload: dict) -> dict | None:
        payload["messages"] = convert_to_openrouter_format(payload["messages"])

        try:
            response = requests.post(
                url=self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=300,
            )
            response_json = response.json()
            logger.debug("OpenRouter response: %s", response_json)
            response.raise_for_status()
            return response_json

        except requests.exceptions.RequestException as e:
            logger.error("Error calling OpenRouter API: %s", e)
            if hasattr(e, "response") and e.response is not None:
                logger.error("Response: %s", e.response.text)
                if e.response.status_code == 500:
                    logger.warning(
                        "Internal server error from OpenRouter - returning empty response"
                    )
                    return None
            raise

    @staticmethod
    def _log_error(
        response_json: dict, *, during_tool_iteration: bool = False
    ) -> None:
        error_msg = response_json.get("error", {})
        if isinstance(error_msg, dict):
            error_text = error_msg.get("message", str(response_json))
        else:
            error_text = str(error_msg) if error_msg else str(response_json)
        suffix = " during tool iteration" if during_tool_iteration else ""
        logger.error(
            "OpenRouter API returned no choices%s. Error: %s", suffix, error_text
        )
