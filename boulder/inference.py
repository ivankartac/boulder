import argparse
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

from jinja2 import Environment, FileSystemLoader, Template

from boulder.domains.db import AttractionDB, HotelDB, RestaurantDB, TrainDB
from boulder.response_parser import ResponseParser
from boulder.domains.tools import (
    BaseTool,
    SearchAttractionsTool,
    SearchHotelsTool,
    SearchRestaurantsTool,
    SearchTrainsTool,
    RestaurantReservationTool,
    BookHotelTool,
    BuyTrainTicketsTool,
)

VALID_SETUPS = [
    "baseline",
    "dialogue",
    "dialogue-concise",
    "baseline-multi-turn",
    "baseline-with-role",
    "dialogue-single-turn",
    "dialogue-reasoning",
    "dialogue-no-tools",
    "dialogue-reduced-domains",
]

ALL_DOMAINS = ["restaurants", "hotels", "attractions", "trains"]

DEFAULTS = {
    "models": [],
    "dataset": [],
    "setup_ids": ["baseline", "dialogue", "dialogue-concise"],
    "domains": None,
    "output_dir": None,
    "parser_model": "qwen3:30b-a3b-instruct-2507-q8_0",
    "api": "ollama",
    "host": "http://localhost:11434",
    "think": False,
    "num_ctx": 8192,
    "num_predict": 8192,
    "temperature": 0.0,
    "seed": 42,
}

VALID_ANSWER_TYPES = ["amount", "restaurants", "distance", "order", "directions", "frequency", "times"]

PROMPT_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompt_templates"
PROMPT_ENV = Environment(loader=FileSystemLoader(PROMPT_TEMPLATE_DIR))

# Maps setup id -> (jinja template filename or None, is_dialogue)
SETUP_TEMPLATE_PATH_MAPPING = {
    "baseline": (None, False),
    "baseline-with-role": (None, False),
    "dialogue": ("dialogue_baseline.jinja", True),
    "dialogue-reduced-domains": ("dialogue_reduced_domains.jinja", True),
    "dialogue-concise": ("dialogue_concise.jinja", True),
    "dialogue-reasoning": ("dialogue_reasoning.jinja", True),
    "dialogue-no-tools": ("dialogue_no_tools.jinja", True),
    "dialogue-single-turn": ("dialogue_single_turn.jinja", True),
    "baseline-multi-turn": (None, False),
}


def load_config(argv: list[str] | None = None) -> argparse.Namespace:
    """Load config from YAML file with CLI overrides.

    Usage:
        python run_inference.py config.yaml [--temperature 0.1 ...]
        python run_inference.py --models qwen3 --dataset data.json ...
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", help="YAML config file")
    parser.add_argument("--models", type=str, nargs="+")
    parser.add_argument("--dataset", type=str, nargs="+")
    parser.add_argument("--setup-ids", type=str, nargs="+")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--parser-model", type=str)
    parser.add_argument("--api", type=str, choices=["ollama", "openrouter"])
    parser.add_argument("--host", type=str)
    parser.add_argument("--think", action="store_true", default=None)
    parser.add_argument("--num-ctx", type=int)
    parser.add_argument("--num-predict", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    config = dict(DEFAULTS)

    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f) or {}
        # Normalize keys: YAML uses underscores, keep consistent
        for key, value in yaml_config.items():
            key = key.replace("-", "_")
            config[key] = value

    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value

    if not config["models"]:
        parser.error("models must be specified (in config file or via --models)")
    if not config["dataset"]:
        parser.error("dataset must be specified (in config file or via --dataset)")
    if not config["output_dir"]:
        parser.error("output_dir must be specified (in config file or via --output-dir)")
    for s in config["setup_ids"]:
        if s not in VALID_SETUPS:
            parser.error(f"invalid setup id: {s}")

    # Validate domains: must be a list of domain lists, one per dataset, or None
    domains = config.get("domains")
    if domains is not None:
        if not isinstance(domains, list) or not domains:
            parser.error("domains must be a list of domain lists (one per dataset)")
        # Normalize: if first element is a string, treat as single domain list for all datasets
        if isinstance(domains[0], str):
            domains = [domains]
            config["domains"] = domains
        n_datasets = len(config["dataset"])
        if len(domains) != n_datasets:
            parser.error(
                f"domains has {len(domains)} entries but dataset has {n_datasets}; "
                f"provide one domain list per dataset"
            )
        for i, domain_list in enumerate(domains):
            for d in domain_list:
                if d not in ALL_DOMAINS:
                    parser.error(f"invalid domain '{d}' in domains[{i}]; valid: {ALL_DOMAINS}")

    config["config_file"] = args.config

    return argparse.Namespace(**config)


def get_tools_for_domains(
    domains: list[str],
    attraction_db: AttractionDB,
    hotel_db: HotelDB,
    restaurant_db: RestaurantDB,
    train_db: TrainDB,
) -> list[BaseTool]:
    tools: list[BaseTool] = []
    for domain in domains:
        if domain == "attractions":
            tools.append(SearchAttractionsTool(attraction_db))
        elif domain == "hotels":
            tools.extend([SearchHotelsTool(hotel_db), BookHotelTool(hotel_db)])
        elif domain == "restaurants":
            tools.extend([SearchRestaurantsTool(restaurant_db), RestaurantReservationTool(restaurant_db)])
        elif domain == "trains":
            tools.extend([SearchTrainsTool(train_db), BuyTrainTicketsTool(train_db)])
    return tools


def get_tools_for_prompt(
    setup_id: str,
    attraction_db: AttractionDB,
    hotel_db: HotelDB,
    restaurant_db: RestaurantDB,
    train_db: TrainDB,
    domains: list[str] | None = None,
) -> list[BaseTool]:
    if setup_id in ("baseline-multi-turn", "dialogue-no-tools", "dialogue-single-turn"):
        return []
    if setup_id == "dialogue-reduced-domains":
        if not domains:
            raise ValueError("dialogue-reduced-domains requires a domains list")
        return get_tools_for_domains(domains, attraction_db, hotel_db, restaurant_db, train_db)
    return get_tools_for_domains(ALL_DOMAINS, attraction_db, hotel_db, restaurant_db, train_db)


def build_chat_messages(
    example: dict,
    setup_id: str,
    dialogue_template: Template | None,
    answer_type: str,
    domains: list[str] | None = None,
) -> tuple[str, list[dict], bool]:
    messages = example.get("messages", [])
    if messages and messages[0].get("role") == "system":
        baseline_prompt = messages[0]["content"]
    else:
        baseline_prompt = None

    current_date = example.get("current_date", datetime.datetime.now().strftime('%Y-%m-%d'))
    current_weekday = datetime.date.fromisoformat(current_date).strftime('%A')
    current_time = example.get("current_time", datetime.datetime.now().strftime('%H:%M'))

    if dialogue_template is not None:
        if answer_type in ["distance", "order", "directions"]:
            units_info = "\n\nSpatial coordinates are given in meters from the origin, a (0, 0) point in the southwest corner of the map."
        elif answer_type == "times":
            units_info = f" Sunset time is {example['sunset_time']}."
        else:
            units_info = ""

        template_params = {
            "weekday": current_weekday,
            "date": current_date,
            "time": current_time,
            "user_name": None,
            "units_info": units_info,
            "domains": domains or ALL_DOMAINS,
        }

        if setup_id in ("dialogue-no-tools", "dialogue-single-turn"):
            template_params["restaurant_data"] = example.get("restaurant_data", "")
            template_params["hotel_data"] = example.get("hotel_data", "")
            template_params["train_data"] = example.get("train_data", "")
            template_params["attraction_data"] = example.get("attraction_data", "")

        dialogue_prompt = dialogue_template.render(**template_params)
    else:
        dialogue_prompt = None

    history = messages[1:] if baseline_prompt else messages

    if setup_id == "baseline-multi-turn":
        if not baseline_prompt:
            raise ValueError("Expected first message in messages to have role 'system' for baseline-multi-turn")
        return baseline_prompt, messages, True
    elif dialogue_template is not None:
        return dialogue_prompt, [{"role": "system", "content": dialogue_prompt}] + history, True
    else:
        if not baseline_prompt:
            raise ValueError(f"Expected system message in messages for setup '{setup_id}'")
        return baseline_prompt, [{"role": "user", "content": baseline_prompt}], False


def extract_targets(example: dict, answer_type: str) -> tuple[Any, Any]:
    parser_enum = None
    if answer_type == "amount":
        targets = example["total_price"]
    elif answer_type == "distance":
        targets = example["targets"]["distance"]
    elif answer_type == "order":
        targets = example["targets"]["optimal_route_order"]
    elif answer_type == "directions":
        targets = example["targets"]["is_correct"]
        parser_enum = {
            "asked_direction": example["targets"]["asked_direction"],
            "restaurant_name": example["restaurant"]["name"],
            "attraction_name": example["attraction"]["name"],
        }
    elif answer_type == "frequency":
        targets = example["targets"]["average_interval_minutes"]
    elif answer_type == "times":
        targets = example["last_departure_time"]
    elif answer_type == "restaurants":
        targets = example["matching_restaurants"]
        parser_enum = [r["name"].lower() for r in example["all_restaurants"]]
    else:
        raise ValueError(f"Unknown answer_type: {answer_type}")
    return targets, parser_enum


def parse_results(output_path: str, answer_type: str, parser: ResponseParser) -> None:
    results = []
    with open(output_path, "r") as f:
        for line in f:
            results.append(json.loads(line))

    for result in results:
        if answer_type == "order":
            parser_enum = result["target"]
        elif answer_type in ["restaurants", "directions"]:
            parser_enum = result["parser_enum"]
        else:
            parser_enum = None
        if result["response"] and len(result["response"]) > 1000:
            response = "..." + result["response"][-1000:]
        else:
            response = result["response"]
        parsed_answer = parser.parse_answer(response, answer_type=answer_type, context=parser_enum)
        result["parsed_answer"] = parsed_answer
        logger.debug("Parsed: %s", parsed_answer)

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def get_prompt_templates(setup_ids: list[str]) -> list[tuple[str, Template | None, bool]]:
    result = []
    for name in setup_ids:
        template_file, is_dialogue = SETUP_TEMPLATE_PATH_MAPPING[name]
        template = PROMPT_ENV.get_template(template_file) if template_file else None
        result.append((name, template, is_dialogue))
    return result


def load_datasets(dataset_paths: list[str]) -> list[dict]:
    datasets = []
    for dataset_path in dataset_paths:
        with open(dataset_path, "r") as f:
            content = json.load(f)

        if not (isinstance(content, dict) and "examples" in content and "metadata" in content):
            raise ValueError(f"Dataset {dataset_path} must have 'examples' and 'metadata' keys")

        metadata = content["metadata"]
        task_name = metadata.get("task_name") or os.path.splitext(os.path.basename(dataset_path))[0]
        answer_type = metadata.get("answer_type")
        if not answer_type:
            raise ValueError(f"Dataset {dataset_path} missing 'answer_type' in metadata")
        if answer_type not in VALID_ANSWER_TYPES:
            raise ValueError(f"Invalid answer_type '{answer_type}' in dataset {dataset_path}")

        datasets.append({
            "path": dataset_path,
            "examples": content["examples"],
            "metadata": metadata,
            "task_name": task_name,
            "answer_type": answer_type,
        })
    return datasets


class ColorPrinter:
    COLORS = {
        "SYSTEM": '\033[38;5;208m',
        "TOOL": '\033[38;5;208m',
        "USER": '\033[34m',
        "ASSISTANT": '\033[35m',
    }
    BOLD = '\033[1m'
    RESET = '\033[0m'

    def _print(self, text: str, role: str) -> None:
        print(f"{self.COLORS[role]}{self.BOLD}{role}: {text}{self.RESET}")

    def print_user(self, text: str) -> None:
        self._print(text, "USER")

    def print_assistant(self, text: str) -> None:
        self._print(text, "ASSISTANT")

    def print_system(self, text: str) -> None:
        self._print(text, "SYSTEM")

    def print_tool(self, text: str) -> None:
        self._print(text, "TOOL")

    def print_messages(self, prompt: str, messages: list[dict], is_dialogue: bool) -> None:
        """Print prompt and message history."""
        self.print_system(prompt)
        if is_dialogue:
            for message in messages:
                role = message["role"]
                if role == "system":
                    continue
                elif role == "user":
                    self.print_user(message["content"])
                elif role == "assistant":
                    if "tool_calls" in message:
                        for tc in message["tool_calls"]:
                            self.print_assistant(
                                f"{tc['id']}: Calling \"{tc['function']['name']}\" "
                                f"with args {tc['function']['arguments']}"
                            )
                    else:
                        self.print_assistant(message["content"])
                elif role == "tool":
                    self.print_tool(message["content"])
