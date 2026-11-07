"""Explicit run context; no directories, downloads or training on import."""
from pathlib import Path
import datetime, hashlib, json, os, sys
import numpy as np
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = None
CONFIG = {}
CLASSES = []
K = 0
GUARD = FOCUS = -1
COMPETITORS = []
SEEDS = []
ARMS = ['tree70', 'matpv']
NUM = []
CAT = []
BENCHMARK = ''

def configure(root):
    global ROOT, CONFIG, CLASSES, K, GUARD, FOCUS, COMPETITORS, SEEDS, NUM, CAT, BENCHMARK
    ROOT = Path(root).resolve()
    CONFIG = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    CLASSES = CONFIG['classes']; K = len(CLASSES)
    GUARD, FOCUS = CONFIG['guard_id'], CONFIG['focus_id']
    COMPETITORS = [c for c in range(1, K) if c not in [GUARD, FOCUS]]
    SEEDS = CONFIG['seeds']; NUM = CONFIG['numeric_features']; CAT = CONFIG['categorical_features']
    BENCHMARK = CONFIG['benchmark']
    assert CLASSES[0] == 'normal' and 0 < GUARD < K and 0 < FOCUS < K and GUARD != FOCUS

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()

def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (z.files if keys is None else keys)}

def threshold(p,y):
    import math
    assert np.isfinite(p).all()
    v=np.sort(1-p[y==0,0]);assert len(v)>0
    return float(v[min(len(v)-1,math.ceil((len(v)+1)*.99)-1)])

def decision(p,tau):
    assert p.shape[1]==K and np.isfinite(p).all()
    return np.where(1-p[:,0]>tau,1+p[:,1:].argmax(1),0).astype(np.int64)

def geometry(x,d):return np.clip((x-d['distance_median'])/d['distance_scale'],-10,10).astype(np.float64)

def probability(model,x):
    p=np.zeros((len(x),K));p[:,model.classes_]=model.predict_proba(x);return p

def tree(seed):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=300,max_features=1.,min_samples_leaf=1,bootstrap=False,n_jobs=1,class_weight='balanced',random_state=seed+4123)

def simplex(v,active):
    a=v[:,active];u=np.sort(a,axis=1)[:,::-1];cs=np.cumsum(u,axis=1)-1
    j=np.arange(1,len(active)+1);rho=(u-cs/j>0).sum(1)-1
    theta=cs[np.arange(len(a)),rho]/(rho+1);p=np.zeros_like(v);p[:,active]=np.maximum(a-theta[:,None],0)
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1) and p.min()>=0
    return p

def dump(p, v):
    p = Path(p).resolve()
    if ROOT is None or not p.is_relative_to(ROOT):
        raise ValueError('Output must remain inside the explicit run directory')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+chr(10), encoding='utf-8')

def check():
    from ..integrity import verify_package
    verify_package()
    for audit in (ROOT/'inputs').glob('*/audit.json'):
        data = read(audit)
        for name, key in [('features.npz', 'features_sha256'), ('oracle_labels.npz', 'oracle_sha256')]:
            assert sha(audit.parent/name) == data[key], f'Prepared input changed: {audit.parent/name}'
