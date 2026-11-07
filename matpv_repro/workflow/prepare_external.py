"""Outcome-blind, bounded external cohorts with exact-input group isolation."""
from .common import *
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def chunks():
    offset = 0
    for file in CONFIG['source_files']:
        path = Path(file['path'])
        assert sha(path) == file['sha256'], 'Raw data SHA256 mismatch'
        for frame in pd.read_csv(path, chunksize=25000, keep_default_na=False):
            frame.columns = frame.columns.str.strip()
            numeric = frame[CONFIG['numeric_features']].apply(pd.to_numeric, errors='coerce').to_numpy(np.float64)
            valid = np.isfinite(numeric).all(1)
            numeric = np.sign(numeric) * np.log1p(np.abs(numeric))
            numeric = numeric.astype(np.float32)
            numeric[numeric == 0] = 0  # Canonicalize negative zero.
            cats = frame[CONFIG['categorical_features']].astype(str).to_numpy()
            if cats.size:
                cats = np.array([[v.strip() or '__MISSING__' for v in row] for row in cats])
            native = (np.full(len(frame), file['label'], dtype=object) if 'label' in file
                      else frame[CONFIG['label_column']].astype(str).str.strip().to_numpy())
            labels = np.array([CONFIG['label_mapping'][str(v)] for v in native], np.int64)
            raw_ids = np.arange(offset, offset + len(frame), dtype=np.int64)
            offset += len(frame)
            hashes = np.array([hashlib.blake2b(n.tobytes() + json.dumps(c.tolist(), ensure_ascii=True,
                               separators=(',', ':')).encode(), digest_size=16).hexdigest()
                               for n, c in zip(numeric, cats)], dtype='U32')
            yield numeric[valid], cats[valid], labels[valid], raw_ids[valid], hashes[valid], len(frame), int((~valid).sum())


def build_cohort():
    path = ROOT / 'inputs' / 'cohort.npz'
    if path.exists():
        cached = read(ROOT / 'outputs/cohort_audit.json')
        assert cached['config_sha256'] == sha(ROOT / 'config.json'), 'Prepared cohort config changed'
        assert sha(path) == cached['cohort_sha256']
        return load(path)
    # Cryptographic input hashes supply a fixed, class-stratified bounded cohort.
    # Whole collision groups are restored in pass two; no fitted outcomes exist.
    selected = {c: np.empty(0, dtype='U32') for c in range(K)}
    raw_count = np.zeros(K, np.int64)
    total = bad = 0
    for num, cat, y, rid, h, n, invalid in chunks():
        total += n
        bad += invalid
        raw_count += np.bincount(y, minlength=K)
        for c in np.unique(y):
            cap = int(CONFIG['cohort_normal_cap'] if c == 0 else CONFIG['cohort_attack_cap'])
            selected[int(c)] = np.unique(np.r_[selected[int(c)], h[y == c]])[:cap]
    wanted = set(np.concatenate(list(selected.values())).tolist())
    kept = {}
    for num, cat, y, rid, h, _, _ in chunks():
        ix = [i for i, v in enumerate(h) if v in wanted]
        for i in ix:
            key = (str(h[i]), int(y[i]))
            if key not in kept:
                kept[key] = (num[i].copy(), cat[i].copy(), int(rid[i]))
            else:
                assert np.array_equal(kept[key][0], num[i]) and np.array_equal(kept[key][1], cat[i]), 'Input digest collision'
    ordered = sorted(kept)
    nh = np.array([k[0] for k in ordered], dtype='U32')
    y = np.array([k[1] for k in ordered], dtype=np.int64)
    numeric = np.array([kept[k][0] for k in ordered], dtype=np.float32)
    categorical = np.array([kept[k][1] for k in ordered])
    ids = np.array([kept[k][2] for k in ordered], dtype=np.int64)
    _, group = np.unique(nh, return_inverse=True)
    d = dict(numeric=numeric, categorical=categorical, y=y, raw_id=ids, hash=nh, group=group)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)
    _, cnt = np.unique(group, return_counts=True)
    audit = {'utc': now(), 'raw_rows': total, 'uniform_nonfinite_removals': bad,
             'finite_raw_class_counts': raw_count.tolist(), 'cohort_rows': len(y),
             'cohort_class_counts': np.bincount(y, minlength=K).tolist(),
             'cohort_components': int(group.max()) + 1,
             'cross_label_input_groups': int((cnt > 1).sum()),
             'same_input_same_class_deduplicated': True,
             'same_input_different_labels_preserved_in_same_group': True,
             'cohort_sha256': sha(path), 'config_sha256': sha(ROOT / 'config.json'), 'outcomes_used_for_selection': False,
             'full_corpus': False, 'natural_prevalence': False,
             'selection': 'Smallest 128-bit canonical model-input hashes per true class, then restore all labels in selected hash groups',
             'caps_before_whole_group_restoration': {'normal': CONFIG['cohort_normal_cap'], 'each_attack': CONFIG['cohort_attack_cap']},
             'independence_limit': 'Exact-input groups only; source release does not permit verified session, device or time isolation'}
    dump(ROOT / 'outputs/cohort_audit.json', audit)
    print('COHORT_READY', BENCHMARK, len(y), np.bincount(y, minlength=K).tolist(), flush=True)
    return d


def exact_quota(eligible, counts, quota, rng):
    chosen, n = [], 0
    for g in rng.permutation(eligible):
        if n + counts[g] <= quota:
            chosen.append(g)
            n += int(counts[g])
        if n == quota:
            break
    assert n == quota, ('Insufficient independent calibration support', n, quota)
    return np.array(chosen, np.int64)


def prepare():
    assert CLASSES[0] == 'normal' and K >= 3 and GUARD != FOCUS
    z = build_cohort()
    y, group = z['y'], z['group']
    ng = int(group.max()) + 1
    counts = np.bincount(group)
    gc = np.zeros((ng, K), np.int64)
    np.add.at(gc, (group, y), 1)
    dominant = gc.argmax(1)
    dominant[gc[:, GUARD] > 0] = GUARD
    dominant[gc[:, FOCUS] > 0] = FOCUS
    audits = {}
    for seed in SEEDS:
        dest = ROOT / 'inputs' / str(seed)
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / 'audit.json').exists():
            audits[str(seed)] = read(dest / 'audit.json')
            assert audits[str(seed)]['config_sha256'] == sha(ROOT / 'config.json')
            assert audits[str(seed)]['features_sha256'] == sha(dest / 'features.npz')
            assert audits[str(seed)]['oracle_sha256'] == sha(dest / 'oracle_labels.npz')
            continue
        rng = np.random.default_rng(seed)
        role = np.full(ng, 2, np.int8)
        for c in range(K):
            candidate = np.flatnonzero(dominant == c)
            if len(candidate):
                n = min(len(candidate) - 1, max(1, int(round(.2 * len(candidate)))))
                role[rng.permutation(candidate)[:n]] = 3
        for c, quota in [(0, 2000), (GUARD, 100)]:
            pure = np.flatnonzero((role == 2) & (gc[:, c] == counts))
            role[exact_quota(pure, counts, quota, rng)] = 1
        source_candidates = []
        ordinary = (role == 2) & (gc[:, FOCUS] == 0) & (gc[:, GUARD] == 0)
        for c in range(K):
            candidate = np.flatnonzero(ordinary & (dominant == c))
            source_candidates.extend(rng.permutation(candidate)[:int(round(.8 * len(candidate)))].tolist())
        nsource = 0
        cap = int(CONFIG['source_row_cap']) - 20
        for g in rng.permutation(source_candidates):
            if nsource + counts[g] <= cap:
                role[g] = 0
                nsource += int(counts[g])
        remaining = 20
        candidates = np.flatnonzero((role == 2) & (gc[:, FOCUS] == 0) & (gc[:, GUARD] > 0))
        for g in rng.permutation(candidates):
            cost = int(gc[g, GUARD])
            if cost <= remaining and nsource + counts[g] <= int(CONFIG['source_row_cap']):
                role[g] = 0
                remaining -= cost
                nsource += int(counts[g])
        roles = role[group]
        si = np.flatnonzero(roles == 0)
        assert (y[si] == FOCUS).sum() == 0 and 0 < (y[si] == GUARD).sum() <= 20
        assert len(si) <= int(CONFIG['source_row_cap'])
        X = z['numeric'].copy()
        if len(CONFIG['categorical_features']):
            vocabulary = CONFIG.get('categorical_vocabulary', 'auto')
            encoder = OneHotEncoder(categories=vocabulary, handle_unknown='ignore', sparse_output=False, dtype=np.float32).fit(z['categorical'][si])
            X = np.c_[X, encoder.transform(z['categorical'])].astype(np.float32)
            dump(dest / 'categorical_encoding.json', {'features': CONFIG['categorical_features'], 'categories': [a.tolist() for a in encoder.categories_], 'fit_on_source_only': vocabulary == 'auto', 'fixed_schema_vocabulary': vocabulary != 'auto'})
        median = np.median(X[si], axis=0)
        q25, q75 = np.quantile(X[si], [.25, .75], axis=0)
        output = {'distance_median': median, 'distance_scale': np.maximum(q75 - q25, .25)}
        oracle = {}
        audit = {'seed': seed, 'cohort_rows': len(y), 'cohort_components': ng,
                 'roles': {}, 'feature_count': X.shape[1], 'source_guard_cap': 20,
                 'unfilled_guard_cap': remaining, 'source_focus_rows': 0,
                 'all_rows_assigned': True, 'class_stratified_capped_cohort': True,
                 'source_row_cap': CONFIG['source_row_cap']}
        for r, code in [('source', 0), ('cal', 1), ('pool', 2), ('test', 3)]:
            ix = np.flatnonzero(roles == code)
            output.update({r + '_X': X[ix], r + '_id': z['raw_id'][ix],
                           r + '_hash': z['hash'][ix], r + '_group': group[ix]})
            if r in ('source', 'cal'):
                output[r + '_y'] = y[ix]
            else:
                oracle[r + '_y'] = y[ix]
            audit['roles'][r] = {'rows': len(ix), 'components': len(np.unique(group[ix])),
                                 'class_counts': np.bincount(y[ix], minlength=K).tolist()}
        for a in ['source', 'cal', 'pool', 'test']:
            for b in ['source', 'cal', 'pool', 'test']:
                if a < b:
                    for f in ['id', 'hash', 'group']:
                        assert not set(output[a + '_' + f]) & set(output[b + '_' + f])
        assert sum(a['rows'] for a in audit['roles'].values()) == len(y)
        assert audit['roles']['test']['class_counts'][GUARD] > 0 and audit['roles']['test']['class_counts'][FOCUS] > 0
        np.savez_compressed(dest / 'features.npz', **output)
        np.savez_compressed(dest / 'oracle_labels.npz', **oracle)
        audit.update(features_sha256=sha(dest / 'features.npz'), oracle_sha256=sha(dest / 'oracle_labels.npz'), config_sha256=sha(ROOT / 'config.json'))
        dump(dest / 'audit.json', audit)
        audits[str(seed)] = audit
        print('PAIRED_INPUTS_READY', BENCHMARK, seed, {k: v['rows'] for k, v in audit['roles'].items()}, flush=True)
    dump(ROOT / 'outputs/data_audit.json', {'utc': now(), 'benchmark': BENCHMARK,
         'classes': CLASSES, 'guard_id': GUARD, 'focus_id': FOCUS, 'seeds': audits,
         'previous_model_predictions_on_this_data': False,
         'rows_removed_for_favorable_outcomes': 0, 'cohort': read(ROOT / 'outputs/cohort_audit.json')})


if __name__ == '__main__':
    prepare()
