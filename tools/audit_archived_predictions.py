"""Recount frozen prediction records. Standard library only; no model code."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "audits/jisa_20261110"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(expected, actual, path="root"):
    """Expected fields must match; recomputation may add its own verification fields."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            require(key in actual, f"Missing field: {path}.{key}")
            compare(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(len(expected) == len(actual), f"Length mismatch: {path}")
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f"{path}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool), f"Not numeric: {path}")
        require(math.isclose(expected, actual, rel_tol=1e-11, abs_tol=1e-12), f"Numeric mismatch: {path}: {expected} != {actual}")
    else:
        require(expected == actual, f"Value mismatch: {path}: {expected!r} != {actual!r}")


def records(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return [{k: v if k == "test_hash" else int(v) for k, v in row.items()} for row in reader]


def class_stats(y, p, c):
    tp = sum(a == c and b == c for a, b in zip(y, p))
    fp = sum(a != c and b == c for a, b in zip(y, p))
    fn = sum(a == c and b != c for a, b in zip(y, p))
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def ratio(a, b):
    return a / b if b else None


def metrics(y, p, f, g):
    a, b = class_stats(y, p, f), class_stats(y, p, g)
    return {**{"target_" + key: value for key, value in a.items()},
            "two_class_f1": (a["f1"] + b["f1"]) / 2,
            "benign_fpr": ratio(sum(a == 0 and b != 0 for a, b in zip(y, p)), y.count(0)),
            "target_any_attack_recall": ratio(sum(a == f and b != 0 for a, b in zip(y, p)), y.count(f)),
            "correct_count": sum(a == b for a, b in zip(y, p)),
            "accuracy": sum(a == b for a, b in zip(y, p)) / len(y)}


def ledger(y, old, new, f, g):
    codes = []
    for truth, before, after in zip(y, old, new):
        if before == after:
            codes.append(0)
        else:
            require(before not in (0, g, f) and after == f, "Reassignment outside permitted route")
            codes.append(1 if truth == f else 2 if before == truth else 3)
    counts = Counter(codes)
    a, b, c = (counts[i] for i in (1, 2, 3))
    alarm = sum((x != 0) != (z != 0) for x, z in zip(old, new))
    guard = sum((x == g) != (z == g) for x, z in zip(old, new))
    lost = sum(x == f and z != f for x, z in zip(old, new))
    require(alarm == guard == lost == 0, "Protected prediction set changed")
    require(sum(t == z for t, z in zip(y, new)) - sum(t == x for t, x in zip(y, old)) == a - b, "Net-correct identity")
    require(sum(t != f and z == f for t, z in zip(y, new)) - sum(t != f and x == f for t, x in zip(y, old)) == b + c, "False-positive identity")
    return {"changed": a + b + c, "A_wrong_to_correct": a, "B_correct_to_wrong": b,
            "C_wrong_to_other_wrong": c, "net_correct_count": a - b, "additional_target_fp": b + c,
            "alarm_changes": alarm, "protected_prediction_changes": guard, "target_prediction_losses": lost}, codes


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.exists(), f"Preserving existing output: {out}")
    metadata = json.loads((ARCHIVE / "run_metadata.json").read_text(encoding="utf-8"))
    expected = json.loads((ARCHIVE / "expected/analysis_results.json").read_text(encoding="utf-8"))
    by_split, stages, class_rows = [], [], []
    id_sets, saved_runs = defaultdict(set), {}
    for spec in metadata["runs"]:
        ds, seed, f, g = (spec[k] for k in ("dataset", "seed", "target_id", "protected_id"))
        rows = records(ARCHIVE / spec["prediction_file"])
        y, h, final = ([r[k] for r in rows] for k in ("true_class", "graph_class", "final_class"))
        n = len(rows)
        require([r["row_position"] for r in rows] == list(range(n)), "Row order changed")
        require(len({r["test_id"] for r in rows}) == n, "Duplicated run IDs")
        names = metadata["classes_by_cohort"][ds]
        require(all(0 <= x < len(names) for vector in (y, h, final) for x in vector), "Invalid class ID")
        ld, codes = ledger(y, h, final, f, g)
        require(codes == [r["transition_code"] for r in rows], "Stored transition code mismatch")
        route = [0 if t != f else 1 if p == 0 else 2 if p == g else 3 if p == f else 4 for t, p in zip(y, h)]
        require(route == [r["target_route_code"] for r in rows], "Stored route code mismatch")
        count = Counter(route)
        target_n, t0, k = y.count(f), count[3], count[4]
        a = ld["A_wrong_to_correct"]
        require(target_n == sum(count[i] for i in (1, 2, 3, 4)), "Routing partition")
        m0, m1 = metrics(y, h, f, g), metrics(y, final, f, g)
        require(m1["target_tp"] == t0 + a and a <= k, "Recall ceiling identity")
        require(m1["target_fn"] == count[1] + count[2] + k - a, "Unrecovered target partition")
        result = {"dataset": ds, "scope": spec["scope"], "seed": seed, "test_rows": n,
                  "target_id": f, "protected_id": g, "target_support": target_n, **ld,
                  "original_target_tp": t0, "eligible_true_target_K": k,
                  "blocked_target_as_benign": count[1], "blocked_target_as_protected": count[2],
                  "unrecovered_eligible_target": k - a, "route_upper_recall": (t0 + k) / target_n,
                  "eligible_target_recovery_rate": ratio(a, k), "additional_recall_pp": 100 * a / target_n,
                  **{key: spec[key] for key in ("target_annotations_first50", "target_annotations_next20", "target_annotations_total70")},
                  **{"graph_" + key: value for key, value in m0.items()},
                  **{"final_" + key: value for key, value in m1.items()}}
        by_split.append(result)
        id_sets[ds].update(r["test_id"] for r in rows)
        saved_runs[(ds, seed)] = rows
        for c, name in enumerate(names):
            cc = Counter(code for truth, code in zip(y, codes) if truth == c)
            class_rows.append({"dataset": ds, "scope": spec["scope"], "seed": seed, "true_class_id": c,
                               "true_class_name": name, "class_support": y.count(c),
                               "A_wrong_to_correct": cc[1], "B_correct_to_wrong": cc[2], "C_wrong_to_other_wrong": cc[3]})
        if ds == "TON":
            sr = records(ARCHIVE / f"ton_stages/TON_{seed}.csv.gz")
            require(len(sr) == n, "Stage length mismatch")
            for key in ("row_position", "test_id", "test_group", "true_class", "graph_class", "final_class"):
                require([r[key] for r in sr] == [r[key] for r in rows], "Stage alignment: " + key)
            require([r["svm_direct_class"] for r in sr] == final, "Conditional verification is not zero-increment")
            for label, before, after in [("graph_to_memory", "graph_class", "memory_class"),
                                         ("memory_to_direct_svm", "memory_class", "svm_direct_class"),
                                         ("direct_svm_to_verification", "svm_direct_class", "final_class")]:
                step, _ = ledger(y, [r[before] for r in sr], [r[after] for r in sr], f, g)
                stages.append({"dataset": ds, "seed": seed, "stage": label, **step})
    by_cohort = []
    numeric = [k for k, v in by_split[0].items() if isinstance(v, (int, float)) and k not in ("seed", "target_id", "protected_id")]
    for ds in dict.fromkeys(r["dataset"] for r in by_split):
        rr = [r for r in by_split if r["dataset"] == ds]
        require(len(rr) == 5, "Expected five paired runs per cohort")
        summary = {"dataset": ds, "scope": rr[0]["scope"], "paired_repetitions": len(rr),
                   "test_record_occurrences": sum(r["test_rows"] for r in rr),
                   "distinct_test_ids_across_repetitions": len(id_sets[ds]),
                   "note": "Descriptive equal-weight split means; overlapping repeats are not independent deployment samples."}
        for key in numeric:
            vals = [r[key] for r in rr if r[key] is not None]
            summary[key + "_n_nonmissing"] = len(vals)
            summary[key + "_mean"] = statistics.mean(vals) if vals else None
            summary[key + "_sample_sd"] = statistics.stdev(vals) if len(vals) > 1 else None
        by_cohort.append(summary)
    for key, actual in [("by_split", by_split), ("by_cohort", by_cohort), ("TON_stages", stages)]:
        compare(expected[key], actual, key)
    out.mkdir(parents=True)
    output = {"by_split": by_split, "by_cohort": by_cohort, "TON_stages": stages,
              "public_checks": {"row_and_group_alignment": True, "transition_and_route_codes": True,
                                "published_counts_match_frozen_summaries": True, "no_model_imports_or_calls": True}}
    (out / "analysis_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(out / "per_run_ledger.csv", by_split)
    write_csv(out / "cohort_summary.csv", by_cohort)
    write_csv(out / "TON_sequential_stages.csv", stages)
    write_csv(out / "true_class_sources.csv", class_rows)
    with (ARCHIVE / "tables/prediction/04_按真实类别的改标来源.csv").open(encoding="utf-8-sig", newline="") as stream:
        old = list(csv.DictReader(stream))
    require(len(old) == len(class_rows), "True-class source row count")
    for archived, calculated in zip(old, class_rows):
        for key, value in calculated.items():
            require(str(value) == archived[key], "True-class source mismatch")
    print("PUBLIC_PREDICTION_AUDIT_OK", len(by_split), "runs;", sum(r["test_rows"] for r in by_split), "record occurrences (overlapping)")


if __name__ == "__main__":
    main()
