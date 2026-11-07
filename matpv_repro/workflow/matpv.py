"""Retained MAT-PV, with dataset class roles supplied by common.CONFIG."""
from .common import *
import gc, pickle, time
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

SVM_CONFIG = dict(kernel='rbf', C=1., class_weight='balanced', probability=False,
                  tol=.001, shrinking=True, cache_size=200, max_iter=200000)


def folds(seed, groups):
    rng = np.random.default_rng(seed + 9191)
    ug, iv = np.unique(groups, return_inverse=True)
    assignment = np.empty(len(ug), int)
    counts = np.zeros(5, int)
    for j in rng.permutation(len(ug)):
        f = int(rng.choice(np.flatnonzero(counts == counts.min())))
        assignment[j] = f
        counts[f] += (iv == j).sum()
    return assignment[iv]


def validation(seed, y, groups):
    rng = np.random.default_rng(seed + 31601)
    held = set()
    for c in sorted(np.unique(y), key=lambda k: len(np.unique(groups[y == k]))):
        cg = np.unique(groups[y == c])
        want = min(len(cg) - 1, max(1, int(np.floor(.2 * len(cg)))))
        for g in rng.permutation(cg):
            if sum(v in held for v in cg) >= want:
                break
            if g in held:
                continue
            tentative = held | {int(g)}
            fit = ~np.isin(groups, list(tentative))
            if set(np.unique(y)).issubset(set(y[fit])):
                held = tentative
    v = np.flatnonzero(np.isin(groups, list(held)))
    assert len(v) > 0 and not set(groups[v]) & set(groups[np.setdiff1d(np.arange(len(y)), v)])
    return v


def first_queries(seed, p, group, tau):
    rng = np.random.default_rng(seed + 18731)
    ug, iv = np.unique(group, return_inverse=True)
    assignment = np.empty(len(ug), int)
    assignment[rng.permutation(len(ug))] = np.arange(len(ug)) % 5
    client = assignment[iv]
    alarm = 1 - p[:, 0] > tau
    selected, strata = [], []
    for c in range(5):
        rr = np.random.default_rng(seed + 18731 + c)
        a = np.flatnonzero((client == c) & alarm)
        b = np.flatnonzero((client == c) & ~alarm)
        na, nb = min(8, len(a)), min(2, len(b))
        left = 10 - na - nb
        extra = min(left, len(a) - na)
        na += extra
        nb += left - extra
        assert na + nb == 10 and nb <= len(b), 'Insufficient target rows in acquisition stratum'
        for name, ix, n in [('alarm', a, na), ('nonalarm', b, nb)]:
            selected.extend(rr.permutation(ix)[:n].tolist())
            strata.append({'client': c, 'stratum': name, 'rows': len(ix), 'selected': n})
    q = np.sort(selected).astype(np.int64)
    assert len(np.unique(q)) == 50
    return q, strata


def diffusion(seed, d, q, yy):
    f = folds(seed, d['pool_group'][q])
    oof = np.zeros((len(q), K))
    sx, sy, px = d['source_X'], d['source_y'], d['pool_X']
    begin = time.monotonic()
    for j in range(5):
        held = f == j
        m = tree(seed).fit(np.r_[sx, px[q[~held]]], np.r_[sy, yy[~held]])
        oof[held] = probability(m, px[q[held]])
        print('MATPV_OOF', seed, j + 1, flush=True)
    base = tree(seed).fit(np.r_[sx, px[q]], np.r_[sy, yy])
    tree_seconds = time.monotonic() - begin
    begin = time.monotonic()
    x = geometry(px, d)
    assert len(x) >= 11, 'Retained ten-neighbor graph needs at least eleven pool rows'
    ni = NearestNeighbors(n_neighbors=11, algorithm='brute', n_jobs=1).fit(x)
    idx = ni.kneighbors(x, return_distance=False)
    neigh = np.array([row[row != i][:10] for i, row in enumerate(idx)])
    a = sparse.csr_matrix((np.ones(len(x) * 10),
                           (np.repeat(np.arange(len(x)), 10), neigh.ravel())),
                          shape=(len(x), len(x)))
    a = a.maximum(a.T)
    a.setdiag(0)
    a.eliminate_zeros()
    degree = np.asarray(a.sum(1)).ravel()
    assert (degree > 0).all()
    w = sparse.diags(1 / degree) @ a
    r0 = np.zeros((len(x), K))
    r0[q] = np.eye(K)[yy] - oof
    r = r0.copy()
    for _ in range(50):
        r = .8 * (w @ r) + .2 * r0
        r[q] = r0[q]
    assert np.allclose(r.sum(1), 0, atol=1e-10)
    index = NearestNeighbors(n_neighbors=10, algorithm='brute', n_jobs=1).fit(x)
    model = {'tree50': base, 'pool_index': index, 'residual': r,
             'active': base.classes_, 'distance_median': d['distance_median'],
             'distance_scale': d['distance_scale'], 'classes': CLASSES,
             'guard_id': GUARD, 'focus_id': FOCUS, 'competitors': COMPETITORS}
    return model, {'oof_and_tree50_seconds': tree_seconds,
                   'graph_seconds': time.monotonic() - begin, 'graph_nodes': len(x),
                   'graph_edges': int(a.nnz), 'fold': f.tolist()}


def graph_proba(model, x):
    ix = model['pool_index'].kneighbors(geometry(x, model), return_distance=False)
    return simplex(probability(model['tree50'], x) + model['residual'][ix].mean(1),
                   model['active'])


def gaussian(head, x):
    delta = x[:, None, :] - head['means'][None, :, :]
    ds = .5 * np.einsum('nci,cij,ncj->nc', delta, head['precisions'], delta)
    return -ds - .5 * np.linalg.slogdet(head['covariances'])[1][None, :]


def evidence(head, x, h):
    scores = gaussian(head, x)
    lookup = {int(c): j for j, c in enumerate(head['classes'])}
    # A zero attack-probability tie can name an unobserved class. Such a class
    # has no Gaussian comparator, so it supplies no promotion evidence.
    result = np.full(len(h), -np.inf)
    known = np.array([int(c) in lookup for c in h], dtype=bool)
    positions = np.flatnonzero(known)
    result[positions] = (scores[positions, lookup[FOCUS]]
                         - scores[positions, [lookup[int(c)] for c in h[positions]]])
    return result


def final_decision(model, x, p=None):
    if p is None:
        p = graph_proba(model, x)
    original = decision(p, model['threshold'])
    current = original.copy()
    gx = geometry(x, model)
    if model['gaussian_head'] is not None:
        h = 1 + p[:, 1:].argmax(1)
        e = evidence(model['gaussian_head'], gx, h)
        current[np.isin(original, COMPETITORS) & (e > model['eta'])] = FOCUS
    full = current.copy()
    if model['binary_svm'] is not None:
        promote = (np.isin(original, COMPETITORS)
                   & (model['binary_svm'].decision_function(gx) > model['svm_threshold']))
        added = promote & (current != FOCUS)
        for h in COMPETITORS:
            rows = added & (original == h)
            pair = model['pairs'][h]
            if pair is not None and rows.any():
                positions = np.flatnonzero(rows)
                rows[positions] = pair.decision_function(gx[positions]) > 0
            full[rows] = FOCUS
    assert np.array_equal(full > 0, original > 0)
    assert np.array_equal(full == GUARD, original == GUARD)
    assert np.all(full[current == FOCUS] == FOCUS)
    return full, p


def ordinary_tree(seed, d, dest, x, y):
    if (dest / 'tree70_complete.json').exists():
        return
    start = time.monotonic()
    base = tree(seed).fit(x, y)
    fitsec = time.monotonic() - start
    with (dest / 'tree70_model.pkl').open('wb') as f:
        pickle.dump(base, f, protocol=pickle.HIGHEST_PROTOCOL)
    cp = probability(base, d['cal_X'])
    bt = threshold(cp, d['cal_y'])
    start = time.monotonic()
    bp = probability(base, d['test_X'])
    seconds = time.monotonic() - start
    np.savez_compressed(dest / 'tree70_predictions.npz', pred=decision(bp, bt),
                        test_proba=bp, cal_proba=cp, threshold=np.array(bt))
    dump(dest / 'tree70_complete.json', {'utc': now(), 'fit_seconds': fitsec,
         'inference_seconds': seconds, 'inference_ms_per_row': 1000 * seconds / len(bp),
         'model_bytes': (dest / 'tree70_model.pkl').stat().st_size,
         'model_sha256': sha(dest / 'tree70_model.pkl'),
         'training_sha256': sha(dest / 'shared_training.npz'),
         'prediction_sha256': sha(dest / 'tree70_predictions.npz'), 'threshold': bt})


def run(seed):
    assert K >= 3 and 0 < GUARD < K and 0 < FOCUS < K and GUARD != FOCUS
    dest = ROOT / 'outputs' / str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'matpv_complete.json').exists() and (dest / 'tree70_complete.json').exists():
        return
    check()
    d = load(ROOT / 'inputs' / str(seed) / 'features.npz')
    assert not any(k in d for k in ['pool_y', 'test_y'])
    assert set(np.unique(d['source_y'])).issubset(set(range(K)))
    assert (d['cal_y'] == GUARD).any(), 'Historical protected-class guard is required'
    if (dest / 'matpv_complete.json').exists():
        t = load(dest / 'shared_training.npz', ['X', 'y'])
        ordinary_tree(seed, d, dest, t['X'], t['y'])
        return
    times = {}
    start = time.monotonic()
    teacher = tree(seed).fit(d['source_X'], d['source_y'])
    cp, pp = probability(teacher, d['cal_X']), probability(teacher, d['pool_X'])
    tau = threshold(cp, d['cal_y'])
    q, strata = first_queries(seed, pp, d['pool_group'], tau)
    if not (dest / 'query50_frozen.json').exists():
        dump(dest / 'query50_frozen.json', {'utc': now(), 'indices': q.tolist(),
             'raw_ids': d['pool_id'][q].tolist(), 'strata': strata,
             'pool_labels_received': False})
    else:
        assert read(dest / 'query50_frozen.json')['indices'] == q.tolist()
    with np.load(ROOT / 'inputs' / str(seed) / 'oracle_labels.npz', allow_pickle=False) as oracle:
        yy = oracle['pool_y'][q].copy()
    times['source_teacher_and_query50_seconds'] = time.monotonic() - start
    del teacher, cp, pp
    model, dt = diffusion(seed, d, q, yy)
    times.update(dt)
    start = time.monotonic()
    poolp = simplex(probability(model['tree50'], d['pool_X']) + model['residual'], model['active'])
    _, canonical = np.unique(d['pool_group'], return_index=True)
    eligible = canonical[~np.isin(d['pool_group'][canonical], d['pool_group'][q])]
    entropy = -(poolp * np.log(np.maximum(poolp, 1e-300))).sum(1)
    order = np.lexsort((eligible, d['pool_id'][eligible], d['pool_group'][eligible], -entropy[eligible]))
    q20 = eligible[order[:20]]
    assert len(q20) == 20 and len(np.unique(d['pool_group'][q20])) == 20
    if not (dest / 'query20_frozen.json').exists():
        dump(dest / 'query20_frozen.json', {'utc': now(), 'indices': q20.tolist(),
             'raw_ids': d['pool_id'][q20].tolist(), 'selected_by': 'corrected_tree50_entropy',
             'new_labels_received': False})
    else:
        assert read(dest / 'query20_frozen.json')['indices'] == q20.tolist()
    with np.load(ROOT / 'inputs' / str(seed) / 'oracle_labels.npz', allow_pickle=False) as oracle:
        y20 = oracle['pool_y'][q20].copy()
    times['query20_seconds'] = time.monotonic() - start
    x = np.r_[d['source_X'], d['pool_X'][q], d['pool_X'][q20]]
    y = np.r_[d['source_y'], yy, y20]
    ids = np.r_[d['source_id'], d['pool_id'][q], d['pool_id'][q20]]
    groups = np.r_[d['source_group'], d['pool_group'][q], d['pool_group'][q20]]
    vi = validation(seed, y, groups)
    np.savez_compressed(dest / 'shared_training.npz', X=x, y=y, raw_id=ids, group=groups,
                        validation_indices=vi, query50=q, query50_y=yy, query20=q20, query20_y=y20)
    dump(dest / 'label_budget.json', {'source': len(d['source_y']), 'target': 70,
         'calibration_normal': int((d['cal_y'] == 0).sum()),
         'guard_validation': int((d['cal_y'] == GUARD).sum()), 'guard_id': GUARD,
         'focus_id': FOCUS, 'classes': CLASSES,
         'source_plus_reviews_counts': np.bincount(y, minlength=K).tolist(),
         'query50_counts': np.bincount(yy, minlength=K).tolist(),
         'query20_counts': np.bincount(y20, minlength=K).tolist(),
         'training_sha256': sha(dest / 'shared_training.npz'), 'test_labels_used': False})
    start = time.monotonic()
    calp = graph_proba(model, d['cal_X'])
    model['threshold'] = threshold(calp, d['cal_y'])
    gx, calx = geometry(x, model), geometry(d['cal_X'], model)
    cguard = d['cal_y'] == GUARD
    from . import gaussian_memory as head_helper
    y50 = np.r_[d['source_y'], yy]
    x50 = np.r_[d['source_X'], d['pool_X'][q]]
    model.update(gaussian_head=None, eta=None, binary_svm=None, svm_threshold=None,
                 pairs={h: None for h in COMPETITORS})
    if (y50 == FOCUS).any() and int((y50 > 0).sum()) > 1:
        head = head_helper.fit_head(geometry(x50[y50 > 0], d), y50[y50 > 0])
        model['gaussian_head'] = head
        e = evidence(head, calx, 1 + calp[:, 1:].argmax(1))
        model['eta'] = max(0., float(e[cguard].max()))
    audits = {}
    if (y == FOCUS).any():
        by = (y == FOCUS).astype(np.int64)
        assert len(np.unique(by)) == 2
        svm = SVC(**SVM_CONFIG, gamma='scale').fit(gx, by)
        assert svm.fit_status_ == 0
        model['binary_svm'] = svm
        model['svm_threshold'] = max(0., float(svm.decision_function(calx[cguard]).max()))
        for h in COMPETITORS:
            mask = (y == h) | (y == FOCUS)
            if (y == h).any():
                pair = SVC(**SVM_CONFIG, gamma=svm._gamma).fit(gx[mask], (y[mask] == FOCUS).astype(np.int64))
                assert pair.fit_status_ == 0
                model['pairs'][h] = pair
                audits[str(h)] = {'rows': int(mask.sum()), 'n_iter': pair.n_iter_.tolist(),
                                  'support': pair.n_support_.tolist()}
    times['memory_and_svm_heads_seconds'] = time.monotonic() - start
    with (dest / 'matpv_model.pkl').open('wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    start = time.monotonic()
    tp = graph_proba(model, d['test_X'])
    pred, _ = final_decision(model, d['test_X'], tp)
    infer = time.monotonic() - start
    cpred, _ = final_decision(model, d['cal_X'], calp)
    np.savez_compressed(dest / 'matpv_predictions.npz', pred=pred, cal_pred=cpred,
                        test_proba=tp, cal_proba=calp, threshold=np.array(model['threshold']))
    dump(dest / 'matpv_complete.json', {'utc': now(), 'times': times,
         'inference_seconds': infer, 'inference_ms_per_row': 1000 * infer / len(pred),
         'model_bytes': (dest / 'matpv_model.pkl').stat().st_size,
         'model_sha256': sha(dest / 'matpv_model.pkl'),
         'training_sha256': sha(dest / 'shared_training.npz'),
         'prediction_sha256': sha(dest / 'matpv_predictions.npz'),
         'unsupported_gaussian': model['gaussian_head'] is None,
         'unsupported_svm': model['binary_svm'] is None, 'pair_audits': audits,
         'unsupported_gaussian_reason': ('focus absent from source plus first50 reviews'
             if not (y50 == FOCUS).any() else 'fewer than two observed attack rows'
             if int((y50 > 0).sum()) <= 1 else None),
         'unsupported_svm_reason': 'focus absent from all shared training labels'
             if not (y == FOCUS).any() else None,
         'threshold': model['threshold'], 'eta': model['eta'],
         'svm_threshold': model['svm_threshold'], 'source_review_rows': len(y),
         'guard_id': GUARD, 'focus_id': FOCUS, 'classes': CLASSES})
    del model, tp, calp, cpred, gx, calx, x50, y50
    gc.collect()
    ordinary_tree(seed, d, dest, x, y)
    print('MATPV_AND_TREE_COMPLETE', seed, 'train_rows', len(y),
          'query_FOCUS', int((np.r_[yy, y20] == FOCUS).sum()), flush=True)


if __name__ == '__main__':
    threadpool_limits(limits=1)
    run(int(sys.argv[1]))
