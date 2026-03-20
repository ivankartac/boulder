import argparse
import csv

from boulder.evaluation.bias_correction import (
    ci_sigma,
    ci_z_test_pvalue,
    significance_stars,
)


def _parse_float(value: str) -> float | None:
    if value == '' or value is None:
        return None
    return float(value)


def load_results(csv_path: str, setup_a: str, setup_b: str) -> dict:
    pairs = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            setup = row['setup_id']
            if setup not in (setup_a, setup_b):
                continue
            key = (row['task'], row['model'])
            label = 'a' if setup == setup_a else 'b'
            pairs.setdefault(key, {})[label] = row
    return pairs


def compare(row_a: dict, row_b: dict) -> dict:
    def _value_and_ci(row):
        corrected = _parse_float(row['corrected'])
        average = _parse_float(row.get('average') or row.get('aggregate'))
        value = corrected if corrected is not None else average
        ci_lo = _parse_float(row['ci_lo'])
        ci_hi = _parse_float(row['ci_hi'])
        sigma = _parse_float(row['ci_sigma'])
        ci = (ci_lo, ci_hi, sigma) if ci_lo is not None else None
        return value, ci

    val_a, ci_a = _value_and_ci(row_a)
    val_b, ci_b = _value_and_ci(row_b)

    if val_a is None or val_b is None or ci_a is None or ci_b is None:
        return {'val_a': val_a, 'val_b': val_b, 'diff': None, 'p_value': None,
                'stars': '', 'method_a': '', 'method_b': ''}

    p_value = ci_z_test_pvalue(val_a, val_b, ci_a, ci_b)
    return {
        'val_a': val_a,
        'val_b': val_b,
        'diff': val_b - val_a,
        'p_value': p_value,
        'stars': significance_stars(p_value),
        'method_a': row_a.get('method', ''),
        'method_b': row_b.get('method', ''),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two setups for statistical significance")
    parser.add_argument("--csv", type=str, required=True, help="Path to evaluation CSV")
    parser.add_argument("--setups", type=str, nargs=2, required=True,
                        metavar=("SETUP_A", "SETUP_B"),
                        help="Two setup_ids to compare")
    parser.add_argument("--output-csv", type=str, help="Save results to CSV")
    args = parser.parse_args()

    setup_a, setup_b = args.setups
    pairs = load_results(args.csv, setup_a, setup_b)

    if not pairs:
        print(f"No matching rows found for setups: {setup_a}, {setup_b}")
        return

    print(f"\nSignificance test: {setup_a} vs {setup_b}")
    print(f"{'task':<35} {'model':<30} {setup_a:>10} {setup_b:>10} {'diff':>8} {'p-value':>10} {'sig':>5} {'method':>10}")
    print("-" * 120)

    results = []
    for (task, model), pair in sorted(pairs.items()):
        if 'a' not in pair or 'b' not in pair:
            continue
        result = compare(pair['a'], pair['b'])
        results.append((task, model, result))

        if result['p_value'] is not None:
            method = result['method_a'] if result['method_a'] == result['method_b'] else f"{result['method_a']}/{result['method_b']}"
            print(f"{task:<35} {model:<30} {result['val_a']:>10.4f} {result['val_b']:>10.4f} "
                  f"{result['diff']:>+8.4f} {result['p_value']:>10.4f} {result['stars']:>5} {method:>10}")
        else:
            print(f"{task:<35} {model:<30} {'N/A':>10} {'N/A':>10} {'':>8} {'N/A':>10}")

    p_values = [r['p_value'] for _, _, r in results if r['p_value'] is not None]
    total = len(p_values)
    print("-" * 120)
    for threshold, label in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
        count = sum(1 for p in p_values if p < threshold)
        print(f"  p < {threshold:<6} {label:>3}  {count}/{total}")

    if args.output_csv:
        fieldnames = ['task', 'model', 'setup_a', 'setup_b', 'value_a', 'value_b',
                      'diff', 'p_value', 'significance', 'method']
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for task, model, r in results:
                writer.writerow({
                    'task': task,
                    'model': model,
                    'setup_a': setup_a,
                    'setup_b': setup_b,
                    'value_a': r['val_a'] if r['val_a'] is not None else '',
                    'value_b': r['val_b'] if r['val_b'] is not None else '',
                    'diff': r['diff'] if r['diff'] is not None else '',
                    'p_value': r['p_value'] if r['p_value'] is not None else '',
                    'significance': r['stars'],
                    'method': r['method_a'] if r['method_a'] == r['method_b'] else f"{r['method_a']}/{r['method_b']}",
                })
        print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
