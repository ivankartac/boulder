import json
import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

import ollama
import requests
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, RootModel

ENV = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
PARAPHRASE_PROMPT_TEMPLATE = ENV.get_template("paraphrase_messages.jinja")
TEMPLATE_DIR = Path(__file__).resolve().parent / "prompt_templates"


class Message(BaseModel):
    model_config = {"extra": "forbid"}
    role: Literal["assistant", "user"]
    content: str


class DialogueResponse(RootModel):
    root: list[Message]

    @classmethod
    def get_json_schema_for_api(cls):
        schema = cls.model_json_schema()

        def set_additional_properties_false(obj):
            if isinstance(obj, dict):
                if obj.get("type") == "object":
                    obj["additionalProperties"] = False
                for value in obj.values():
                    set_additional_properties_false(value)
            elif isinstance(obj, list):
                for item in obj:
                    set_additional_properties_false(item)

        set_additional_properties_false(schema)
        return schema


def strip_markdown_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


class DialogueTemplateGenerator:
    def __init__(self, prompt: str, model: str, temperature: float = 1.0, api_key: str = None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.messages = [{'role': 'system', 'content': prompt}]

    def generate_dialogue(self, api: str = "openrouter") -> list[dict]:
        if api == "ollama":
            return self._generate_ollama()
        elif api == "openrouter":
            return self._generate_openrouter()
        else:
            raise ValueError(f"Unknown API: {api!r}. Must be 'ollama' or 'openrouter'.")

    def _generate_ollama(self) -> list[dict]:
        response = ollama.chat(
            model=self.model,
            messages=self.messages,
            options={"temperature": self.temperature},
            format=DialogueResponse.get_json_schema_for_api(),
        )
        try:
            content = strip_markdown_fences(response['message']['content'])
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            logger.debug("Ollama raw content: %s", response['message']['content'])
            raise

    def _generate_openrouter(self) -> list[dict]:
        if self.api_key is None:
            raise ValueError("API key is not set")

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": self.messages,
                "temperature": self.temperature,
                "structured_outputs": True,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": DialogueResponse.get_json_schema_for_api()
                },
            },
        )
        response.raise_for_status()
        response_json = response.json()
        try:
            content = strip_markdown_fences(response_json['choices'][0]['message']['content'])
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            logger.debug("OpenRouter raw content: %s", response_json['choices'][0]['message']['content'])
            raise


def format_dialogue_for_display(dialogue: list[dict]) -> str:
    lines = []
    for message in dialogue:
        role = message["role"].upper()
        content = message["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def remove_messages_and_track_indices(dialogue: list[dict]) -> tuple[list[dict], dict[int, dict]]:
    removed_messages = {}
    filtered_dialogue = []

    last_user_idx = None
    for idx in range(len(dialogue) - 1, -1, -1):
        if dialogue[idx]["role"] == "user":
            last_user_idx = idx
            break

    for idx, message in enumerate(dialogue):
        if message["role"] == "placeholder":
            removed_messages[idx] = message
        elif idx == last_user_idx:
            removed_messages[idx] = message
        else:
            filtered_dialogue.append(message)

    return filtered_dialogue, removed_messages


def reinsert_messages(dialogue: list[dict], removed_messages: dict[int, dict]) -> list[dict]:
    result = list(dialogue)

    for idx in sorted(removed_messages.keys()):
        result.insert(idx, removed_messages[idx])

    return result


def verify_placeholders(original_dialogue: list[dict], generated_dialogue: list[dict]) -> bool:
    pattern = r'\{\{\s*([a-z_]+)\s*\}\}'

    for idx, (orig_msg, gen_msg) in enumerate(zip(original_dialogue, generated_dialogue)):
        if orig_msg["role"] == "placeholder":
            continue

        required_placeholders = set(re.findall(pattern, orig_msg.get("content", "")))
        found_placeholders = set(re.findall(pattern, gen_msg.get("content", "")))

        if required_placeholders != found_placeholders:
            logger.debug("Message %d: Missing %s, Extra %s", idx, required_placeholders - found_placeholders, found_placeholders - required_placeholders)
            return False

    return True


def generate_paraphrases(
    original_dialogue: list[dict],
    model: str,
    temperature: float = 1.0,
    api_key: str | None = None,
    api: str = "openrouter",
    num_paraphrases: int = 10,
    max_retries: int = 3,
) -> list[list[dict]]:
    filtered_dialogue, removed_messages = remove_messages_and_track_indices(original_dialogue)

    first_assistant_message = None
    for msg in original_dialogue:
        if msg["role"] == "assistant":
            first_assistant_message = msg.copy()
            break

    valid_paraphrases: list[list[dict]] = []
    previous_filtered_paraphrases: list[str] = []
    max_attempts = num_paraphrases * max_retries * 3
    attempts = 0

    while len(valid_paraphrases) < num_paraphrases and attempts < max_attempts:
        attempts += 1
        prompt = PARAPHRASE_PROMPT_TEMPLATE.render(
            dialogue=filtered_dialogue,
            previous_paraphrases=previous_filtered_paraphrases,
        )
        generator = DialogueTemplateGenerator(
            prompt, model=model, temperature=temperature, api_key=api_key,
        )

        for retry in range(max_retries):
            try:
                generated_dialogue = generator.generate_dialogue(api=api)

                if not generated_dialogue or not isinstance(generated_dialogue, list):
                    logger.info("Attempt %d retry %d: invalid result format", attempts, retry + 1)
                    continue

                full_dialogue = reinsert_messages(generated_dialogue, removed_messages)

                if len(full_dialogue) != len(original_dialogue):
                    logger.info(
                        "Attempt %d retry %d: wrong message count (%d vs %d)",
                        attempts, retry + 1, len(full_dialogue), len(original_dialogue),
                    )
                    continue

                if not verify_placeholders(original_dialogue, full_dialogue):
                    logger.info("Attempt %d retry %d: invalid placeholders", attempts, retry + 1)
                    continue

                # Restore original first assistant message
                if first_assistant_message:
                    for idx, msg in enumerate(full_dialogue):
                        if msg["role"] == "assistant":
                            full_dialogue[idx] = first_assistant_message.copy()
                            break

                valid_paraphrases.append(full_dialogue)
                previous_filtered_paraphrases.append(
                    format_dialogue_for_display(generated_dialogue)
                )
                break

            except Exception:
                logger.info("Attempt %d retry %d: generation error", attempts, retry + 1, exc_info=True)
                continue

    return valid_paraphrases