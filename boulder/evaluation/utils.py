import argparse
import csv
import json
import logging
import os
from collections import defaultdict

import yaml

logger = logging.getLogger(__name__)

from boulder.evaluation.bias_correction import load_annotation_params


DEFAULTS = {
    "datasets": None,
    "setup_id": None,
    "normalize_mae": True,
    "aggregate": True,
    "k_minutes": 5.0,
    "k_meters": 500.0,
    "ppi": False,
    "parser_annotations": "data/parser_validation.csv",
    "output_csv": "data/evaluations/results.csv",
}


def load_eval_config(argv: list[str] | None = None) -> argparse.Namespace:
    """Load eval config from YAML file with CLI overrides.

    Usage:
        python run_evaluation.py config.yaml [--k-minutes 10]
        python run_evaluation.py --datasets results/ --ppi
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", help="YAML config file")
    parser.add_argument("--datasets", type=str, nargs="+")
    parser.add_argument("--setup-id", nargs="+", type=str)
    parser.add_argument("--aggregate", action="store_true", default=None)
    parser.add_argument("--normalize-mae", action="store_true", default=None)
    parser.add_argument("--k-minutes", type=float)
    parser.add_argument("--k-meters", type=float)
    parser.add_argument("--ppi", action="store_true", default=None)
    parser.add_argument("--parser-annotations", type=str)
    parser.add_argument("--output-csv", type=str)
    args = parser.parse_args(argv)

    config = dict(DEFAULTS)

    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f) or {}
        for key, value in yaml_config.items():
            key = key.replace("-", "_")
            config[key] = value

    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value

    return argparse.Namespace(**config)


def load_datasets(paths: list[str] | None) -> dict:
    all_datasets = {}
    for path in (paths or []):
        dataset_name = os.path.basename(path).replace('.jsonl', '')
        items_by_prompt = defaultdict(list)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                items_by_prompt[item.get("setup_id", "default")].append(item)
        all_datasets[dataset_name] = dict(items_by_prompt)
    return all_datasets


def load_ppi_params(*, ppi: bool, parser_annotations: str) -> dict:
    if not ppi:
        return {}
    ppi_params = load_annotation_params(parser_annotations)
    if ppi_params:
        logger.info("PPI Measurement Error Correction Parameters")
        for task in sorted(ppi_params):
            for model in sorted(ppi_params[task]):
                for prompt in sorted(ppi_params[task][model]):
                    p = ppi_params[task][model][prompt]
                    label = f"{task}/{model}/{prompt}"
                    logger.info("  %s: Se=%.3f, Sp=%.3f (TP=%d, FP=%d, FN=%d, TN=%d, N=%d)",
                                label, p['sensitivity'], p['specificity'],
                                p['tp'], p['fp'], p['fn'], p['tn'], p['n'])
    return ppi_params


CSV_FIELDNAMES = ['task', 'model', 'setup_id', 'metric_type', 'n', 'average', 'corrected',
                  'ci_lo', 'ci_hi', 'ci_sigma', 'avg_response_length', 'method']


def write_csv(csv_rows: list[dict], output_csv: str):
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info("CSV results saved to '%s'", output_csv)
