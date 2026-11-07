"""Release and split checks; no estimator fit or predict."""
from pathlib import Path
import hashlib, json
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
def verify_package():
    manifest = json.loads((PACKAGE_ROOT/'release_manifest.json').read_text(encoding='utf-8'))
    failures = [p for p, expected in manifest['files'].items()
                if not (PACKAGE_ROOT/p).is_file() or file_sha(PACKAGE_ROOT/p) != expected]
    if failures:
        raise ValueError('Release files changed or missing: '+', '.join(failures))
    return len(manifest['files'])
def verify_splits(dataset, run_dir):
    import numpy as np
    run_dir = Path(run_dir)
    reference = PACKAGE_ROOT/'reference'/dataset/'splits'
    done = []
    for path in sorted(reference.glob('*.npz')):
        actual = run_dir/'inputs'/path.stem/'features.npz'
        if not actual.exists():
            continue
        with np.load(path, allow_pickle=False) as expected, np.load(actual, allow_pickle=False) as observed:
            for key in expected.files:
                if key not in observed or not np.array_equal(expected[key], observed[key]):
                    raise ValueError(f'Split mismatch for {dataset}/{path.stem}/{key}')
        done.append(int(path.stem))
    return done
