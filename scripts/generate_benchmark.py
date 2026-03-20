"""Generate benchmark datasets using the benchmark synthesizer.

Usage:
    # Generate all tasks (100 examples each, default):
    python scripts/generate_benchmark.py -t data/dialogue_templates.json -o data/benchmark/

    # Generate specific tasks:
    python scripts/generate_benchmark.py -t data/dialogue_templates.json -o data/benchmark/ \
        --tasks train_ticket_price directional_relations

    # Custom number of examples and seed:
    python scripts/generate_benchmark.py -t data/dialogue_templates.json -o data/benchmark/ \
        -n 50 --seed 123
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from boulder.benchmark_synthesizer import (
    DirectionalRelationsGenerator,
    HotelAttractionWalkingOrderGenerator,
    HotelPriceGenerator,
    HotelRestaurantDistanceGenerator,
    RestaurantOpenTimeGenerator,
    TrainFrequencyGenerator,
    TrainPriceGenerator,
    TrainSunsetGenerator,
)

logger = logging.getLogger(__name__)

# Maps task name -> (generator class, template key in the dialogue templates file, answer_type)
TASK_REGISTRY = {
    "train_ticket_price": (TrainPriceGenerator, "train_ticket_price", "amount"),
    "accommodation_price": (HotelPriceGenerator, "accommodation_price", "amount"),
    "train_departure_time": (TrainSunsetGenerator, "train_departure_time", "times"),
    "restaurants_opening_hours": (RestaurantOpenTimeGenerator, "restaurants_opening_hours", "restaurants"),
    "train_departure_frequency": (TrainFrequencyGenerator, "train_departure_frequency", "frequency"),
    "directional_relations": (DirectionalRelationsGenerator, "directional_relations", "directions"),
    "hotel_to_restaurant_distance": (HotelRestaurantDistanceGenerator, "hotel_to_restaurant_distance", "distance"),
    "shortest_walking_path": (HotelAttractionWalkingOrderGenerator, "shortest_walking_path", "order"),
}


def generate_task(
    task_name: str,
    generator_cls: type,
    templates: list[list[dict]],
    rng: np.random.Generator,
    n_examples: int,
    max_attempts_factor: int = 10,
) -> list[dict]:
    generator = generator_cls(rng)
    examples = []
    attempts = 0
    max_attempts = n_examples * max_attempts_factor

    while len(examples) < n_examples and attempts < max_attempts:
        attempts += 1
        template = rng.choice(templates).tolist() if templates else []
        example = generator.generate(dialogue_template=template)
        if example is None:
            logger.debug("Task %s: attempt %d returned None, retrying", task_name, attempts)
            continue
        example["id"] = len(examples)
        examples.append(example)

    if len(examples) < n_examples:
        logger.warning(
            "Task %s: only generated %d/%d examples after %d attempts",
            task_name, len(examples), n_examples, max_attempts,
        )

    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark datasets")
    parser.add_argument("-t", "--templates", required=True, help="Path to a dialogue templates file")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for benchmark JSON files")
    parser.add_argument("-n", "--num-examples", type=int, default=100, help="Number of examples per task (default: 100)")
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_REGISTRY.keys()),
                        help="Tasks to generate (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--num-paraphrases", type=int, default=9,
                        help="Max paraphrases to use per task (default: 9)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    with open(args.templates) as f:
        all_templates = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    tasks = args.tasks or list(TASK_REGISTRY.keys())

    for task_name in tasks:
        generator_cls, template_key, answer_type = TASK_REGISTRY[task_name]

        template_data = all_templates.get(template_key, {})
        original = template_data.get("original", [])
        paraphrases = template_data.get("paraphrases", [])[:args.num_paraphrases]
        templates = [original] + paraphrases if original else []

        logger.info("Generating %s (%d templates, %d examples)", task_name, len(templates), args.num_examples)

        examples = generate_task(task_name, generator_cls, templates, rng, args.num_examples)

        dataset = {
            "examples": examples,
            "metadata": {
                "task_name": task_name,
                "answer_type": answer_type,
            },
        }

        output_path = output_dir / f"{task_name}.json"
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        logger.info("Wrote %d examples to %s", len(examples), output_path)

    logger.info("Done. Generated %d tasks.", len(tasks))


if __name__ == "__main__":
    main()
