import json
import logging
import time
from pathlib import Path

import httpx
import ollama

logger = logging.getLogger(__name__)
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).resolve().parent / "prompt_templates"
ENV = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

PROMPT_AMOUNT = ENV.get_template("parse_amount.jinja")
PROMPT_DISTANCE = ENV.get_template("parse_distance.jinja")
PROMPT_RESTAURANT_NAMES = ENV.get_template("parse_restaurant_names.jinja")
PROMPT_ORDER = ENV.get_template("parse_order.jinja")
PROMPT_FREQUENCY = ENV.get_template("parse_frequency.jinja")
PROMPT_DIRECTIONS = ENV.get_template("parse_directions.jinja")
PROMPT_TIME = ENV.get_template("parse_time.jinja")


class ResponseParser:
    def __init__(
        self,
        model: str,
        host: str,
        timeout: int = 120,
        retries: int = 3,
        retry_delay: int = 1,
        retry_temperature: float = 1.0,
    ):
        self.model = model
        self.client = ollama.Client(host=host, timeout=timeout)
        self.temperature = 0.0
        self.retries = retries
        self.retry_delay = retry_delay
        self.retry_temperature = retry_temperature

    def _parse_with_llm(
        self,
        prompt: str,
        json_field: str = None,
    ):
        messages = [{"role": "user", "content": prompt}]
        temperature = self.temperature
        retries_left = self.retries

        while retries_left > 0:
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    format="json",
                    options={"temperature": temperature},
                )
            except httpx.ReadTimeout:
                logger.warning("Timeout reading response from model %s", self.model)
                time.sleep(self.retry_delay)
                logger.info("Retrying... (%d retries left)", retries_left)
                retries_left -= 1
                continue

            content = response["message"]["content"]
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            try:
                result = json.loads(content)
                if json_field:
                    result = result.get(json_field)
                return result

            except json.JSONDecodeError as e:
                logger.warning("Error parsing answer: %s", content)
                logger.warning("Error: %s", e)
                time.sleep(self.retry_delay)
                logger.info("Retrying... (%d retries left)", retries_left)
                retries_left -= 1
                temperature = self.retry_temperature

        return None

    def parse_answer(self, answer: str, answer_type: str, context=None):
        if not answer:
            return None
        if answer_type == "amount":
            return self.parse_amount(answer)
        elif answer_type == "restaurants":
            return self.parse_restaurant_names(answer, context)
        elif answer_type == "distance":
            return self.parse_distance(answer)
        elif answer_type == "directions":
            return self.parse_directions(answer, context)
        elif answer_type == "order":
            return self.parse_order(answer, context)
        elif answer_type == "frequency":
            return self.parse_frequency(answer)
        elif answer_type == "times":
            return self.parse_time(answer)
        else:
            raise ValueError(f"Invalid answer type: {answer_type}")

    def parse_amount(self, answer: str) -> float:
        prompt = PROMPT_AMOUNT.render(answer=answer)
        return self._parse_with_llm(prompt, json_field="total_price")

    def parse_restaurant_names(self, answer: str, context: list[str]) -> list[str]:
        prompt = PROMPT_RESTAURANT_NAMES.render(
            restaurant_names=", ".join(context),
            answer=answer,
        )
        return self._parse_with_llm(prompt, json_field="restaurant_names")

    def parse_distance(self, answer: str) -> float:
        prompt = PROMPT_DISTANCE.render(answer=answer)
        return self._parse_with_llm(prompt, json_field="distance")

    def parse_directions(self, answer: str, context: dict) -> str:
        asked_direction = context["asked_direction"]
        prompt = PROMPT_DIRECTIONS.render(
            attraction_name=context["attraction_name"].title(),
            asked_direction=asked_direction,
            restaurant_name=context["restaurant_name"].title(),
            answer=answer,
        )
        return self._parse_with_llm(prompt, json_field=f"is_{asked_direction}")

    def parse_order(self, answer: str, context: list[str]) -> list[str]:
        prompt = PROMPT_ORDER.render(
            attraction_names=", ".join(context),
            answer=answer,
        )
        result = self._parse_with_llm(prompt)
        if isinstance(result, dict):
            return result.get("order")
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0].get("order")
        return result

    def parse_frequency(self, answer: str) -> float:
        prompt = PROMPT_FREQUENCY.render(answer=answer)
        return self._parse_with_llm(prompt, json_field="average_interval_minutes")

    def parse_time(self, answer: str) -> str:
        prompt = PROMPT_TIME.render(answer=answer)
        return self._parse_with_llm(prompt, json_field="latest_departure_time")
