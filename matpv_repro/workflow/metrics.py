from .common import *
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score

def metrics(y, pred, p, new=None):
    """Correct-subtype metrics and binary alarm metrics have separate names."""
    y, pred, p = np.asarray(y), np.asarray(pred), np.asarray(p)
    assert K >= 3 and 0 < GUARD < K and 0 < FOCUS < K and GUARD != FOCUS
    assert y.shape == pred.shape and p.shape == (len(y), K) and len(y)
    assert np.isin(y, np.arange(K)).all() and np.isin(pred, np.arange(K)).all()
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(K), zero_division=0)
    alarm, attack, score = pred > 0, y > 0, 1 - p[:, 0]
    bp = float(attack[alarm].mean()) if alarm.any() else 0.
    br = float(alarm[attack].mean()) if attack.any() else None
    new = np.zeros(len(y), dtype=bool) if new is None else np.asarray(new, dtype=bool)
    assert new.shape == y.shape and not (new & attack).any()
    out = {'rare_f1': float(f1[[GUARD, FOCUS]].mean()),
           'macro_f1': float(f1.mean()), 'accuracy': float((pred == y).mean()),
           'fpr': float(alarm[~attack].mean()) if (~attack).any() else None,
           'new_normal_fpr': float(alarm[new].mean()) if new.any() else None,
           'attack_alarm_recall': br, 'attack_alarm_precision': bp,
           'binary_f1': (2 * bp * br / (bp + br) if bp + br else 0.) if br is not None else None,
           'binary_average_precision': float(average_precision_score(attack, score)) if attack.any() else None,
           'binary_roc_auc': float(roc_auc_score(attack, score)) if len(np.unique(attack)) == 2 else None,
           'class_precision': precision.tolist(), 'class_recall': recall.tolist(),
           'class_f1': f1.tolist(), 'support': support.tolist(),
           'confusion': confusion_matrix(y, pred, labels=np.arange(K)).tolist()}
    for c, name in [(GUARD, 'guard'), (FOCUS, 'focus')]:
        out.update({name + '_f1': float(f1[c]), name + '_precision': float(precision[c]),
                    name + '_subtype_recall': float(recall[c]),
                    name + '_alarm_recall': float(alarm[y == c].mean()) if (y == c).any() else None})
    return out
