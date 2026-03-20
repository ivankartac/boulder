import csv
from dataclasses import dataclass
import logging
import os
from typing import Literal
import re

import numpy as np

logger = logging.getLogger(__name__)
from ppi_py import ppi_mean_ci, ppi_mean_pointestimate
from scipy import stats


TASK_COMPARISON_TYPE = {
    'train_ticket_price': 'integer',
    'accommodation_price': 'integer',
    'train_departure_time': 'string',
    'train_departure_frequency': 'integer',
    'hotel_to_restaurant_distance': 'integer',
    'directional_relations': 'direction_bool',
    'shortest_walking_path': 'ordered_list',
    'restaurants_opening_hours': 'list',
}


def _parse_numbered_list(s):
    items = []
    for line in str(s).strip().split('\n'):
        line = line.strip()
        match = re.match(r'\d+\.\s*(.*)', line)
        if match:
            items.append(match.group(1).strip())
        elif line:
            items.append(line.strip())
    return items


def _annotation_pipeline_says_correct(parsed, target, comparison_type):
    if comparison_type == 'integer':
        try:
            return int(round(float(parsed))) == int(round(float(target)))
        except (ValueError, TypeError):
            return False
    elif comparison_type == 'string':
        return str(parsed).strip() == str(target).strip()
    elif comparison_type == 'direction_bool':
        p = str(parsed).lower().strip()
        t = str(target).strip().lower()
        if t == 'true':
            t = 'yes'
        elif t == 'false':
            t = 'no'
        return p == t
    elif comparison_type == 'ordered_list':
        parsed_list = _parse_numbered_list(parsed)
        target_list = _parse_numbered_list(target)
        return (len(parsed_list) == len(target_list) and
                all(p.lower() == t.lower() for p, t in zip(parsed_list, target_list)))
    elif comparison_type == 'list':
        parsed_list = [x.lower() for x in _parse_numbered_list(parsed)]
        target_list = [x.lower() for x in _parse_numbered_list(target)]
        return set(parsed_list) == set(target_list)
    return False


def load_annotation_params(csv_path: str) -> dict:
    """Load parser annotations from a single CSV and compute PPI parameters.

    Expected columns: task, model, setup_id, parsed_answer, target, correct,
    corrected_parsed_answer.

    Returns:
        dict: {task_name: {model_name: {setup_id: {
            'Y', 'Yhat', 'sensitivity', 'specificity', 'tp', 'fp', 'fn', 'tn', 'n',
            'precision_rectifiers'?, 'mae_true_errors'?, 'mae_pipeline_errors'?
        }}}}
    """
    Y_by_key: dict[tuple, list[float]] = {}
    Yhat_by_key: dict[tuple, list[float]] = {}
    precision_rectifiers_by_key: dict[tuple, list[float]] = {}
    mae_errors_by_key: dict[tuple, list[tuple[float, float]]] = {}

    if not os.path.exists(csv_path):
        logger.warning("Annotations file not found: %s", csv_path)
        return {}

    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            task_name = row['task'].strip()
            if task_name not in TASK_COMPARISON_TYPE:
                continue
            comparison_type = TASK_COMPARISON_TYPE[task_name]

            model_name = row['model'].strip()
            setup_id = row['setup_id'].strip()
            parsed_val = row['parsed_answer']
            target_val = row['target']
            correct_str = row['correct'].strip()
            corrected_val = row.get('corrected_parsed_answer', '').strip() or None

            if not model_name:
                continue

            if correct_str in ('1', 'True', 'true'):
                verdict_correct = True
            elif correct_str in ('0', 'False', 'false'):
                verdict_correct = False
            else:
                continue

            pipeline_positive = _annotation_pipeline_says_correct(
                parsed_val, target_val, comparison_type)

            if verdict_correct:
                model_correct = pipeline_positive
            else:
                corrected_for_comparison = corrected_val or 'null'
                if corrected_for_comparison.lower() == 'null' and comparison_type in ('list', 'ordered_list'):
                    corrected_for_comparison = ''
                model_correct = _annotation_pipeline_says_correct(
                    corrected_for_comparison, target_val, comparison_type)

            key = (task_name, model_name, setup_id)
            Y_by_key.setdefault(key, []).append(float(model_correct))
            Yhat_by_key.setdefault(key, []).append(float(pipeline_positive))

            if comparison_type == 'list':
                rect_val = _compute_precision_rectifier(
                    verdict_correct, parsed_val, target_val, corrected_val)
                precision_rectifiers_by_key.setdefault(key, []).append(rect_val)

            if comparison_type == 'integer':
                error_pair = _compute_mae_error_pair(
                    verdict_correct, parsed_val, target_val, corrected_val)
                if error_pair is not None:
                    mae_errors_by_key.setdefault(key, []).append(error_pair)

    params: dict = {}
    for key, Y_list in Y_by_key.items():
        Y = np.array(Y_list)
        Yhat = np.array(Yhat_by_key[key])
        n = len(Y)
        if n == 0:
            continue

        tp = int(np.sum(Y * Yhat))
        fp = int(np.sum((1 - Y) * Yhat))
        fn = int(np.sum(Y * (1 - Yhat)))
        tn = int(np.sum((1 - Y) * (1 - Yhat)))

        entry = {
            'Y': Y,
            'Yhat': Yhat,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 1.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 1.0,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'n': n,
        }

        if key in precision_rectifiers_by_key:
            entry['precision_rectifiers'] = np.array(precision_rectifiers_by_key[key])
        if key in mae_errors_by_key:
            errors = mae_errors_by_key[key]
            entry['mae_true_errors'] = np.array([e[0] for e in errors])
            entry['mae_pipeline_errors'] = np.array([e[1] for e in errors])

        task_name, model_name, setup_id = key
        params.setdefault(task_name, {}).setdefault(model_name, {})[setup_id] = entry

    return params


def _compute_precision_rectifier(verdict_correct, parsed_val, target_val,
                                 corrected_val: str | None) -> float:
    """Compute real-valued rectifier Y_i - f(X_i) for precision tasks."""
    if verdict_correct:
        return 0.0

    parsed_lower = [x.lower() for x in _parse_numbered_list(parsed_val)]
    target_lower = [x.lower() for x in _parse_numbered_list(target_val)]
    f_xi = _list_precision(parsed_lower, target_lower)

    if corrected_val is None:
        return 1.0 - f_xi

    if corrected_val.lower() != 'null':
        corrected_lower = [x.lower() for x in _parse_numbered_list(corrected_val)]
    else:
        corrected_lower = []
    return _list_precision(corrected_lower, target_lower) - f_xi


def _list_precision(predicted: list[str], target: list[str]) -> float:
    if len(predicted) == 0:
        return 0.0 if len(target) > 0 else 1.0
    return sum(x in target for x in predicted) / len(predicted)


def _compute_mae_error_pair(verdict_correct, parsed_val, target_val,
                            corrected_val: str | None) -> tuple[float, float] | None:
    if verdict_correct:
        try:
            target_num = float(target_val)
            parsed_num = float(parsed_val)
            pipeline_mae = abs(parsed_num - target_num)
            return (pipeline_mae, pipeline_mae)
        except (ValueError, TypeError):
            try:
                float(target_val)
                return (float('inf'), float('inf'))
            except (ValueError, TypeError):
                return None

    if not corrected_val:
        return None
    try:
        true_mae = abs(float(corrected_val) - float(target_val))
        pipeline_mae = abs(float(parsed_val) - float(target_val))
        return (true_mae, pipeline_mae)
    except (ValueError, TypeError):
        return None


def _ppi_lib_mean_and_ci(Yhat_unlabeled, Y, Yhat, alpha=0.05):
    """Call ppi_py library for point estimate and CI (lam=1 for vanilla PPI).

    Returns:
        (point_estimate, ci_lo, ci_hi, sigma)
    """
    Yhat_unlabeled = np.asarray(Yhat_unlabeled, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Yhat = np.asarray(Yhat, dtype=float)
    est = float(ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled, lam=1).item())
    ci = ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=alpha, lam=1)
    ci_lo = float(ci[0].item())
    ci_hi = float(ci[1].item())
    sigma = (ci_hi - ci_lo) / (2 * stats.norm.ppf(1 - alpha / 2))
    return est, ci_lo, ci_hi, sigma


def ppi_ci(individual_scores, Y, Yhat, alpha=0.05):
    """Compute PPI-corrected estimate and confidence interval.

    Returns:
        (estimate, ci_lower, ci_upper, sigma): all clipped to [0, 1]
    """
    Yhat_unlabeled = np.asarray(individual_scores, dtype=float)
    est, ci_lo, ci_hi, sigma = _ppi_lib_mean_and_ci(Yhat_unlabeled, Y, Yhat, alpha)
    return (
        float(np.clip(est, 0.0, 1.0)),
        float(np.clip(ci_lo, 0.0, 1.0)),
        float(np.clip(ci_hi, 0.0, 1.0)),
        float(sigma),
    )


def ppi_ci_from_rectifier(scores, rectifier, alpha=0.05):
    """Compute PPI-corrected estimate and CI given pre-computed rectifier.

    Returns:
        (estimate, ci_lower, ci_upper, sigma): all clipped to [0, 1]
    """
    scores = np.asarray(scores, dtype=float)
    rectifier = np.asarray(rectifier, dtype=float)
    n = len(rectifier)
    Yhat_labeled = np.zeros(n)
    Y_labeled = rectifier
    est, ci_lo, ci_hi, sigma = _ppi_lib_mean_and_ci(scores, Y_labeled, Yhat_labeled, alpha)
    return (
        float(np.clip(est, 0.0, 1.0)),
        float(np.clip(ci_lo, 0.0, 1.0)),
        float(np.clip(ci_hi, 0.0, 1.0)),
        float(sigma),
    )


def bootstrap_ci(individual_scores, n_bootstrap=2000, alpha=0.05, rng_seed=42):
    """Compute bootstrap estimate and confidence interval.

    Returns:
        (estimate, ci_lower, ci_upper, sigma): sigma is None for bootstrap
    """
    rng = np.random.RandomState(rng_seed)
    individual_scores = np.asarray(individual_scores)
    n = len(individual_scores)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(individual_scores, size=n, replace=True)
        means[i] = resample.mean()
    return (
        float(individual_scores.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
        None,
    )


def ci_sigma(ci_tuple):
    """Extract sigma from a CI tuple (ci_lo, ci_hi[, sigma]).

    Uses the stored sigma (3rd element) if available, otherwise recovers from CI width.
    """
    z_95 = stats.norm.ppf(0.975)
    if len(ci_tuple) > 2 and ci_tuple[2] is not None:
        return ci_tuple[2]
    return (ci_tuple[1] - ci_tuple[0]) / (2 * z_95) if ci_tuple[1] > ci_tuple[0] else 0.0


def ci_z_test_pvalue(val1, val2, ci1, ci2):
    """Compute two-sided p-value from CI-based z-test between two estimates.

    z = |t1 - t2| / sqrt(s1^2 + s2^2)
    """
    sigma1 = ci_sigma(ci1)
    sigma2 = ci_sigma(ci2)
    sigma_diff = np.sqrt(sigma1**2 + sigma2**2)
    if sigma_diff > 0:
        z_stat = abs(val1 - val2) / sigma_diff
        return float(2 * (1 - stats.norm.cdf(z_stat)))
    return 1.0 if val1 == val2 else 0.0


def significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


@dataclass
class CorrectionResult:
    corrected_value: float
    ci: tuple
    method: Literal['ppi', 'bootstrap']


def apply_correction(
    *,
    metric: str,
    average_score: float,
    scores: np.ndarray,
    task_name: str,
    model_name: str,
    setup_id: str,
    ppi_params: dict,
    k_minutes: float = 5.0,
    k_meters: float = 500.0,
) -> CorrectionResult:
    """Apply PPI or bootstrap correction to evaluation metrics.

    For accuracy/direction/path tasks: PPI on binary accuracy scores.
    For precision tasks: PPI using precision rectifiers.
    For MAE tasks: PPI in normalized (sigmoid) space.
    Fallback: plain bootstrap CI with no correction.
    """
    ppi_entry = ppi_params.get(task_name, {}).get(model_name, {}).get(setup_id)

    if ppi_entry is not None and metric == 'accuracy':
        est, ci_lo, ci_hi, sigma = ppi_ci(scores, ppi_entry['Y'], ppi_entry['Yhat'])
        return CorrectionResult(corrected_value=est, ci=(ci_lo, ci_hi, sigma), method='ppi')

    if ppi_entry is not None and metric == 'precision' and 'precision_rectifiers' in ppi_entry:
        est, ci_lo, ci_hi, sigma = ppi_ci_from_rectifier(
            np.asarray(scores, dtype=float), ppi_entry['precision_rectifiers'])
        return CorrectionResult(corrected_value=est, ci=(ci_lo, ci_hi, sigma), method='ppi')

    if ppi_entry is not None and metric == 'mae' and 'mae_true_errors' in ppi_entry:
        k = _k_for_task(task_name, k_minutes, k_meters)
        norm_scores = _sigmoid_normalize(scores, k)
        norm_rect = _sigmoid_normalize(ppi_entry['mae_true_errors'], k) - _sigmoid_normalize(ppi_entry['mae_pipeline_errors'], k)
        est, ci_lo, ci_hi, sigma = ppi_ci_from_rectifier(norm_scores, norm_rect)
        return CorrectionResult(corrected_value=est, ci=(ci_lo, ci_hi, sigma), method='ppi')

    # Fallback: bootstrap CI, no correction
    if metric == 'mae':
        k = _k_for_task(task_name, k_minutes, k_meters)
        est, ci_lo, ci_hi, sigma = bootstrap_ci(_sigmoid_normalize(scores, k))
    else:
        est, ci_lo, ci_hi, sigma = bootstrap_ci(scores)
    return CorrectionResult(corrected_value=est, ci=(ci_lo, ci_hi, sigma), method='bootstrap')


def _k_for_task(task_name: str, k_minutes: float, k_meters: float) -> float:
    if "frequency" in task_name.lower():
        return k_minutes
    elif "distance" in task_name.lower():
        return k_meters
    return 1.0


def _sigmoid_normalize(scores, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.asarray(scores, dtype=float) / k)


def normalize_mae_scores(
    individual_scores_by_task: dict,
    task_metric_types: dict[str, str],
    k_minutes: float,
    k_meters: float,
) -> dict:
    normalized = {}
    for task_name, task_data in individual_scores_by_task.items():
        metric_type = task_metric_types.get(task_name, 'accuracy')
        k = _k_for_task(task_name, k_minutes, k_meters) if metric_type == 'mae' else None
        normalized[task_name] = {}
        for model_name, model_data in task_data.items():
            normalized[task_name][model_name] = {}
            for setup_id, scores in model_data.items():
                arr = np.array(scores)
                normalized[task_name][model_name][setup_id] = _sigmoid_normalize(arr, k) if k else arr
    return normalized


def compute_aggregate_ci(
    *,
    individual_scores_by_task: dict,
    task_metric_types: dict[str, str],
    ppi_params: dict,
    ci_by_task: dict,
    k_minutes: float = 5.0,
    k_meters: float = 500.0,
) -> dict:
    """Compute PPI-corrected micro-average CIs across tasks.

    When ppi_params is non-empty, computes PPI-corrected averages
    with normal-approximation CIs propagating per-task sigmas.
    Otherwise, computes plain bootstrap CIs on pooled scores.

    Returns:
        {model_name: {setup_id: (ci_lo, ci_hi, sigma, point_estimate)}}
    """
    # Normalize MAE scores pointwise
    normalized = normalize_mae_scores(
        individual_scores_by_task, task_metric_types, k_minutes, k_meters)

    # Pool scores and track per-task breakdown
    pooled: dict[str, dict[str, list]] = {}
    per_task: dict[str, dict[str, list[tuple[str, np.ndarray]]]] = {}

    for task_name, task_data in normalized.items():
        for model_name, model_data in task_data.items():
            pooled.setdefault(model_name, {})
            per_task.setdefault(model_name, {})
            for setup_id, scores in model_data.items():
                pooled[model_name].setdefault(setup_id, []).extend(scores)
                per_task[model_name].setdefault(setup_id, []).append(
                    (task_name, np.array(scores)))

    agg_ci: dict[str, dict[str, tuple]] = {}

    for model_name, model_data in pooled.items():
        agg_ci[model_name] = {}
        for setup_id, scores in model_data.items():
            task_entries = per_task[model_name][setup_id]

            if ppi_params:
                # Pre-compute normalized rectifiers for MAE tasks
                mae_norm_rects: dict[str, np.ndarray] = {}
                for t_name, _ in task_entries:
                    metric_type = task_metric_types.get(t_name, 'accuracy')
                    ppi_e = ppi_params.get(t_name, {}).get(model_name, {}).get(setup_id)
                    if ppi_e is not None and metric_type == 'mae' and 'mae_true_errors' in ppi_e:
                        k = _k_for_task(t_name, k_minutes, k_meters)
                        true_err = ppi_e['mae_true_errors']
                        pipe_err = ppi_e['mae_pipeline_errors']
                        mae_norm_rects[t_name] = _sigmoid_normalize(true_err, k) - _sigmoid_normalize(pipe_err, k)

                # Point estimate: weighted mean of per-task PPI-corrected scores
                total_n = 0
                weighted_sum = 0.0
                for t_name, t_scores in task_entries:
                    n_t = len(t_scores)
                    ppi_e = ppi_params.get(t_name, {}).get(model_name, {}).get(setup_id)
                    if ppi_e is not None:
                        if t_name in mae_norm_rects:
                            rect = mae_norm_rects[t_name]
                        elif 'precision_rectifiers' in ppi_e:
                            rect = ppi_e['precision_rectifiers']
                        else:
                            rect = ppi_e['Y'] - ppi_e['Yhat']
                        corrected = max(0.0, min(1.0, float(np.mean(t_scores) + np.mean(rect))))
                        weighted_sum += corrected * n_t
                    else:
                        weighted_sum += float(np.sum(t_scores))
                    total_n += n_t
                theta_agg = weighted_sum / total_n if total_n > 0 else 0.0

                # Normal approximation CI: propagate per-task PPI variances
                sigma_sq_agg = 0.0
                for t_name, t_scores in task_entries:
                    n_t = len(t_scores)
                    w_t = n_t / total_n
                    ci_entry = ci_by_task.get(t_name, {}).get(model_name, {}).get(setup_id)
                    if ci_entry is not None and len(ci_entry) > 2 and ci_entry[2] is not None:
                        sigma_t = ci_entry[2]
                    else:
                        sigma_t = float(np.std(t_scores, ddof=1) / np.sqrt(n_t)) if n_t > 1 else 0.0
                    sigma_sq_agg += w_t**2 * sigma_t**2
                sigma_agg = float(np.sqrt(sigma_sq_agg))
                z = stats.norm.ppf(0.975)
                ci_lo = float(np.clip(theta_agg - z * sigma_agg, 0.0, 1.0))
                ci_hi = float(np.clip(theta_agg + z * sigma_agg, 0.0, 1.0))
                agg_ci[model_name][setup_id] = (ci_lo, ci_hi, sigma_agg, theta_agg)
            else:
                # Plain bootstrap CI
                est, ci_lo, ci_hi, _ = bootstrap_ci(np.array(scores))
                agg_ci[model_name][setup_id] = (ci_lo, ci_hi, None, est)

    return agg_ci
