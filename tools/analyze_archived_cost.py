"""Descriptive cost sensitivity from an already verified prediction audit.

No estimator, prediction or training code is imported. No manuscript is edited.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import argparse

_parser = argparse.ArgumentParser(description="Recalculate hypothetical costs from archived audit counts; no models.")
_parser.add_argument("--source", type=Path, required=True)
_parser.add_argument("--output", type=Path, required=True)
_args = _parser.parse_args()
SOURCE = _args.source.resolve()
OUT = _args.output.resolve()
PRESERVE = [SOURCE]
GRID = [Fraction(0), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(5), Fraction(10)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def save_csv(name, rows):
    assert rows
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio_info(a, f):
    if a:
        return {'break_even_lambda': float(Fraction(f, a)),
                'break_even_exact': str(Fraction(f, a)),
                'status': 'strict_improvement_if_lambda_above_break_even'}
    if f:
        return {'break_even_lambda': None, 'break_even_exact': None,
                'status': 'always_worse_for_finite_lambda_and_positive_cFP'}
    return {'break_even_lambda': None, 'break_even_exact': None,
            'status': 'always_tied'}


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserving existing output: {OUT}')
    before = {str(p): {'sha256_before': digest(p), 'bytes': p.stat().st_size} for p in PRESERVE}
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    rows = data['by_split']
    assert len(rows) == 20
    OUT.mkdir()
    groups = defaultdict(list)
    split_results, grid_split = [], []

    for r in rows:
        a, b, c = (r[k] for k in ('A_wrong_to_correct', 'B_correct_to_wrong', 'C_wrong_to_other_wrong'))
        f, n = b + c, r['test_rows']
        assert f == r['additional_target_fp']
        assert r['final_target_fn'] - r['graph_target_fn'] == -a
        assert r['final_target_fp'] - r['graph_target_fp'] == f
        assert r['alarm_changes'] == r['protected_prediction_changes'] == r['target_prediction_losses'] == 0
        item = {'dataset': r['dataset'], 'scope': r['scope'], 'seed': r['seed'],
                'test_rows': n, 'A': a, 'B': b, 'C': c, 'F_B_plus_C': f, **ratio_info(a, f)}
        split_results.append(item)
        groups[r['dataset']].append(item)
        for lam in GRID:
            delta = Fraction(f) - lam * a
            grid_split.append({'dataset': r['dataset'], 'scope': r['scope'], 'seed': r['seed'],
                               'test_rows': n, 'lambda': float(lam), 'delta_total_in_cFP_units': float(delta),
                               'delta_per_1000_in_cFP_units': float(1000 * delta / n),
                               'direction': 'improved' if delta < 0 else 'worse' if delta > 0 else 'tied'})

    cohorts, grid_cohorts = [], []
    for dataset, rr in groups.items():
        assert len(rr) == 5
        sum_a = sum(r['A'] for r in rr)
        sum_f = sum(r['F_B_plus_C'] for r in rr)
        sum_a_per_record = sum((Fraction(r['A'], r['test_rows']) for r in rr), Fraction())
        sum_f_per_record = sum((Fraction(r['F_B_plus_C'], r['test_rows']) for r in rr), Fraction())
        primary = sum_f_per_record / sum_a_per_record
        secondary = Fraction(sum_f, sum_a)
        all_improve = all(r['A'] > 0 for r in rr)
        all_bound = max(Fraction(r['F_B_plus_C'], r['A']) for r in rr) if all_improve else None
        cohorts.append({'dataset': dataset, 'scope': rr[0]['scope'], 'paired_runs': len(rr),
                        'sum_A_across_overlapping_runs': sum_a, 'sum_F_across_overlapping_runs': sum_f,
                        'lambda_primary_equal_run_per_record': float(primary), 'lambda_primary_exact': str(primary),
                        'lambda_secondary_equal_run_total': float(secondary), 'lambda_secondary_exact': str(secondary),
                        'all_five_strictly_improve_possible': all_improve,
                        'lambda_all_five_strictly_improve_above': float(all_bound) if all_bound is not None else None,
                        'always_worse_runs': sum(r['A'] == 0 and r['F_B_plus_C'] > 0 for r in rr),
                        'always_tied_runs': sum(r['A'] == 0 and r['F_B_plus_C'] == 0 for r in rr)})
        for lam in GRID:
            dd = [Fraction(r['F_B_plus_C']) - lam * r['A'] for r in rr]
            per = [float(1000 * d / r['test_rows']) for d, r in zip(dd, rr)]
            totals = [float(d) for d in dd]
            grid_cohorts.append({'dataset': dataset, 'scope': rr[0]['scope'], 'lambda': float(lam),
                                 'mean_delta_per_1000': statistics.mean(per),
                                 'sample_SD_delta_per_1000': statistics.stdev(per),
                                 'mean_delta_total': statistics.mean(totals),
                                 'sample_SD_delta_total': statistics.stdev(totals),
                                 'improved_runs': sum(d < 0 for d in dd),
                                 'tied_runs': sum(d == 0 for d in dd), 'worse_runs': sum(d > 0 for d in dd)})

    stage_groups = defaultdict(list)
    for r in data['TON_stages']:
        stage_groups[r['stage']].append(r)
    ton_n = {r['seed']: r['test_rows'] for r in rows if r['dataset'] == 'TON'}
    stages = []
    for stage, rr in stage_groups.items():
        a = sum(r['A_wrong_to_correct'] for r in rr)
        f = sum(r['B_correct_to_wrong'] + r['C_wrong_to_other_wrong'] for r in rr)
        an = sum((Fraction(r['A_wrong_to_correct'], ton_n[r['seed']]) for r in rr), Fraction())
        fn = sum((Fraction(r['B_correct_to_wrong'] + r['C_wrong_to_other_wrong'], ton_n[r['seed']]) for r in rr), Fraction())
        stages.append({'dataset': 'TON', 'stage': stage, 'paired_runs': len(rr), 'sum_A': a, 'sum_F': f,
                       'lambda_equal_run_per_record': float(fn / an) if an else None,
                       **ratio_info(a, f), 'interpretation': 'conditional_sequential_change_not_counterfactual_ablation'})

    save_csv('01_逐轮成本界限.csv', split_results)
    save_csv('02_队列平均成本界限.csv', cohorts)
    save_csv('03_假设成本网格_逐轮.csv', grid_split)
    save_csv('04_假设成本网格_队列汇总.csv', grid_cohorts)
    save_csv('05_TON顺序分支成本界限.csv', stages)
    save_json('cost_analysis.json', {'comparison': 'saved_graph_stage_to_saved_final_prediction',
                                    'formula': 'delta_L/cFP = (B+C) - lambda*A; cFP>0; lambda=cFN/cFP>=0',
                                    'primary_estimand': 'equal run mean of loss change per complete test record',
                                    'costs': 'hypothetical parameters, not observed operational costs',
                                    'by_split': split_results, 'by_cohort': cohorts,
                                    'grid_by_split': grid_split, 'grid_by_cohort': grid_cohorts,
                                    'TON_sequential_stages': stages})
    for p in PRESERVE:
        before[str(p)]['sha256_after'] = digest(p)
        before[str(p)]['unchanged'] = before[str(p)]['sha256_before'] == before[str(p)]['sha256_after']
        assert before[str(p)]['unchanged']
    save_json('source_manifest.json', {'created_at_local': datetime.now().isoformat(),
                                      'read_only_inputs': before, 'no_training_no_inference': True,
                                      'all_20_FN_FP_identities_verified': True,
                                      'upstream_raw_prediction_audit': str(SOURCE.parent / 'source_manifest.json')})
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'output': str(OUT), 'primary_break_even': {r['dataset']: r['lambda_primary_equal_run_per_record'] for r in cohorts},
                      'rows': {'splits': len(split_results), 'grid_split': len(grid_split), 'grid_cohort': len(grid_cohorts)},
                      'all_sources_unchanged': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
