from dataclasses import dataclass
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

from boulder.evaluation.bias_correction import (
    _k_for_task,
    apply_correction,
    compute_aggregate_ci,
    normalize_mae_scores,
)
from boulder.evaluation.evaluators import TASK_EVALUATORS
from boulder.evaluation.utils import load_datasets, load_ppi_params, write_csv


@dataclass
class EvalResults:
    csv_rows: list[dict]
    scores_by_task: dict
    metric_types: dict[str, str]
    lengths_by_task: dict
    ci_by_task: dict


class EvaluationPipeline:
    def __init__(
        self,
        *,
        datasets: list[str],
        output_csv: str,
        setup_id: list[str] | None = None,
        ppi: bool = False,
        parser_annotations: str | None = None,
        aggregate: bool = True,
        normalize_mae: bool = True,
        k_minutes: float = 5.0,
        k_meters: float = 500.0,
    ):
        self.dataset_paths = datasets
        self.output_csv = output_csv
        self.setup_id = setup_id
        self.ppi = ppi
        self.parser_annotations = parser_annotations
        self.do_aggregate = aggregate
        self.normalize_mae = normalize_mae
        self.k_minutes = k_minutes
        self.k_meters = k_meters

        self.ppi_params: dict = {}
        self.results: EvalResults | None = None

    @classmethod
    def from_config(cls, args) -> "EvaluationPipeline":
        return cls(
            datasets=args.datasets,
            output_csv=args.output_csv,
            ppi=args.ppi,
            parser_annotations=args.parser_annotations,
            setup_id=args.setup_id,
            normalize_mae=args.normalize_mae,
            aggregate=args.aggregate,
            k_minutes=args.k_minutes,
            k_meters=args.k_meters,
        )

    def run(self):
        self.load()
        self.evaluate()
        if self.normalize_mae:
            self.normalize()
        elif any(r['metric_type'] == 'mae' and r.get('method') == 'ppi' for r in self.results.csv_rows):
            logger.warning("PPI correction applied to MAE tasks without --normalize-mae.")
        if self.do_aggregate and self.results.scores_by_task:
            self.aggregate()
        self.write()

    def load(self):
        self.datasets = load_datasets(self.dataset_paths)
        self.ppi_params = load_ppi_params(
            ppi=self.ppi,
            parser_annotations=self.parser_annotations,
        )

    def evaluate(self):
        csv_rows = []
        scores_by_task = {}
        metric_types = {}
        lengths_by_task = {}
        ci_by_task = {}

        for dataset_name, data in self.datasets.items():
            logger.info("Processing dataset: %s", dataset_name)

            for setup_id, items in data.items():
                if self.setup_id:
                    setup_id_base = re.sub(r'-\d+$', '', setup_id)
                    if setup_id_base not in self.setup_id:
                        continue

                if not items:
                    continue

                first = items[0]
                task_name = first["task_name"]
                model_name = first["model"]
                answer_type = first["answer_type"]
                if answer_type not in TASK_EVALUATORS:
                    logger.warning("Unknown answer_type '%s' in %s/%s", answer_type, dataset_name, setup_id)
                    continue

                logger.info("Evaluating %s/%s/%s", task_name, model_name, setup_id)
                result = TASK_EVALUATORS[answer_type](items)

                correction = apply_correction(
                    metric=result.metric,
                    average_score=result.average_score,
                    scores=result.scores,
                    task_name=task_name,
                    model_name=model_name,
                    setup_id=setup_id,
                    ppi_params=self.ppi_params,
                    k_minutes=self.k_minutes,
                    k_meters=self.k_meters,
                )

                logger.info("%s: %.2f", result.metric, result.average_score)
                if correction.method == 'ppi':
                    ci_lo, ci_hi = correction.ci[0], correction.ci[1]
                    logger.info("  -> PPI: %.4f [%.4f, %.4f]", correction.corrected_value, ci_lo, ci_hi)

                ci_by_task.setdefault(task_name, {}).setdefault(model_name, {})[setup_id] = correction.ci
                scores_by_task.setdefault(task_name, {}).setdefault(model_name, {})[setup_id] = np.array(result.scores).astype(float)
                metric_types[task_name] = result.metric
                if result.response_lengths:
                    lengths_by_task.setdefault(task_name, {}).setdefault(model_name, {})[setup_id] = result.response_lengths

                ci_sigma = correction.ci[2] if len(correction.ci) > 2 and correction.ci[2] is not None else ''
                csv_rows.append({
                    'task': task_name,
                    'model': model_name,
                    'setup_id': setup_id,
                    'metric_type': result.metric,
                    'n': len(items),
                    'average': result.average_score,
                    'corrected': correction.corrected_value if correction.method == 'ppi' else '',
                    'ci_lo': correction.ci[0],
                    'ci_hi': correction.ci[1],
                    'ci_sigma': ci_sigma,
                    'avg_response_length': float(np.mean(result.response_lengths)) if result.response_lengths else 0.0,
                    'method': correction.method,
                })

        self.results = EvalResults(csv_rows, scores_by_task, metric_types, lengths_by_task, ci_by_task)

    def normalize(self):
        """Normalize MAE rows in-place using sigmoid: 1/(1 + MAE/k)."""
        for row in self.results.csv_rows:
            if row['metric_type'] != 'mae':
                continue
            k = _k_for_task(row['task'], self.k_minutes, self.k_meters)
            if k == 1.0:
                continue

            ppi_e = self.ppi_params.get(row['task'], {}).get(row['model'], {}).get(row['setup_id'])
            if ppi_e is not None and 'mae_true_errors' in ppi_e:
                continue

            row['average'] = 1.0 / (1.0 + row['average'] / k)
            if row['corrected'] != '':
                row['corrected'] = 1.0 / (1.0 + row['corrected'] / k)
            old_lo, old_hi = row['ci_lo'], row['ci_hi']
            row['ci_hi'] = 1.0 / (1.0 + old_lo / k)
            row['ci_lo'] = 1.0 / (1.0 + old_hi / k)

    def aggregate(self):
        agg_ci = compute_aggregate_ci(
            individual_scores_by_task=self.results.scores_by_task,
            task_metric_types=self.results.metric_types,
            ppi_params=self.ppi_params,
            ci_by_task=self.results.ci_by_task,
            k_minutes=self.k_minutes,
            k_meters=self.k_meters,
        )

        normalized = normalize_mae_scores(
            self.results.scores_by_task, self.results.metric_types,
            self.k_minutes, self.k_meters)

        pooled_scores = {}
        pooled_lengths = {}
        for task_name, task_data in normalized.items():
            for model_name, model_data in task_data.items():
                for setup_id, scores in model_data.items():
                    pooled_scores.setdefault(model_name, {}).setdefault(setup_id, []).extend(scores)

        for task_name, task_data in self.results.lengths_by_task.items():
            for model_name, model_data in task_data.items():
                for setup_id, lengths in model_data.items():
                    pooled_lengths.setdefault(model_name, {}).setdefault(setup_id, []).extend(lengths)

        for model_name, model_data in pooled_scores.items():
            for setup_id, scores in model_data.items():
                ci_entry = agg_ci.get(model_name, {}).get(setup_id) if agg_ci else None
                corrected_avg = ci_entry[3] if self.ppi_params and ci_entry is not None and len(ci_entry) > 3 else ''
                lengths = pooled_lengths.get(model_name, {}).get(setup_id, [0])

                self.results.csv_rows.append({
                    'task': '_aggregate',
                    'model': model_name,
                    'setup_id': setup_id,
                    'metric_type': 'aggregate',
                    'n': len(scores),
                    'average': float(np.mean(scores)),
                    'corrected': corrected_avg,
                    'ci_lo': ci_entry[0] if ci_entry is not None else '',
                    'ci_hi': ci_entry[1] if ci_entry is not None else '',
                    'ci_sigma': ci_entry[2] if ci_entry is not None else '',
                    'avg_response_length': float(np.mean(lengths)),
                    'method': 'ppi' if self.ppi_params else 'bootstrap',
                })

    def write(self):
        write_csv(self.results.csv_rows, self.output_csv)
