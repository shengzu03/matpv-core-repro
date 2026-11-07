"""Explicit command-line entry point. Only `run` fits models."""
from pathlib import Path
import argparse
import importlib
import importlib.metadata
import json
import os
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('ton_iot', 'rt_iot2022', 'ciciot2023')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def config_check(config, template):
    """Paths may change; frozen scientific fields and source order may not."""
    portable = json.loads(json.dumps(config))
    for file in portable['source_files']:
        file.pop('path', None)
    if portable != template:
        raise ValueError('Scientific configuration differs from the release template; use a separate study copy.')


def run_directory(args):
    return Path(args.run_root).expanduser().resolve() / args.dataset


def initialized(args):
    root = run_directory(args)
    path = root / 'config.json'
    if not path.is_file():
        raise ValueError('Run init first for this dataset and run-root.')
    config = read(path)
    config_check(config, read(PACKAGE_ROOT / 'configs' / (args.dataset + '.json')))
    return root, config


def source_check(config):
    from .integrity import file_sha
    missing, mismatches = [], []
    for entry in config['source_files']:
        path = Path(entry['path'])
        if not path.is_file():
            missing.append(str(path))
        elif file_sha(path) != entry['sha256']:
            mismatches.append(str(path))
    if missing or mismatches:
        raise ValueError(json.dumps({'missing_data_files': missing, 'sha256_mismatches': mismatches}, ensure_ascii=False))
    return len(config['source_files'])


def runtime(root):
    """Keep caches inside the explicitly selected run directory."""
    import tempfile
    for key, sub in [('TMP', 'tmp'), ('TEMP', 'tmp'), ('PIP_CACHE_DIR', 'pip'),
                     ('JOBLIB_TEMP_FOLDER', 'joblib'), ('MPLCONFIGDIR', 'mpl')]:
        path = root / 'cache' / sub
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    tempfile.tempdir = str(root / 'cache' / 'tmp')
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    from .workflow import common
    common.configure(root)
    return common


def selected_seeds(args, config):
    requested = args.seed or config['seeds']
    if len(requested) != len(set(requested)) or not set(requested).issubset(config['seeds']):
        raise ValueError('Use unique seeds from the frozen dataset configuration.')
    return requested


def train(args):
    root, config = initialized(args)
    if sys.flags.optimize:
        raise ValueError('Do not use python -O: frozen code relies on integrity assertions.')
    for seed in selected_seeds(args, config):
        if not (root / 'inputs' / str(seed) / 'audit.json').is_file():
            raise ValueError(f'Prepared inputs missing for seed {seed}; run prepare first.')
    common = runtime(root)
    common.check()
    from .integrity import verify_splits
    verified = verify_splits(args.dataset, root)
    if not set(selected_seeds(args, config)).issubset(verified):
        raise ValueError('Prepared inputs lack verified frozen split identities.')
    module = importlib.import_module('.workflow.' + ('matpv_ton' if args.dataset == 'ton_iot' else 'matpv'), __package__)
    from threadpoolctl import threadpool_limits
    versions = environment_report()
    write(root / 'environment_at_run.json', versions)
    with threadpool_limits(limits=1):
        for seed in selected_seeds(args, config):
            from .integrity import file_sha
            for arm in ['tree70', 'matpv']:
                folder = root / 'outputs' / str(seed)
                marker = folder / (arm + '_complete.json')
                if marker.exists():
                    path = folder / (arm + '_predictions.npz')
                    if not path.exists() or file_sha(path) != read(marker)['prediction_sha256']:
                        raise ValueError(f'Completed prediction missing or changed: {path}')
            print('START_CORE', args.dataset, seed, flush=True)
            module.run(seed)
    print('Core run finished. Run evaluate for test metrics.', flush=True)


def evaluate(args):
    """Evaluate only files produced in the user's explicit run directory."""
    root, config = initialized(args)
    common = runtime(root)
    common.check()
    from .workflow.metrics import metrics
    import numpy as np
    from .integrity import file_sha
    records, missing, locked = [], [], {}
    requested = selected_seeds(args, config)
    # Freeze the entire explicitly requested set before opening any test labels.
    for seed in requested:
        folder = root / 'outputs' / str(seed)
        paths = {arm: folder / (arm + '_predictions.npz') for arm in ['tree70', 'matpv']}
        absent = [arm for arm, path in paths.items()
                  if not path.exists() or not (folder / (arm + '_complete.json')).exists()]
        if absent:
            missing.append({'seed': seed, 'missing': absent})
            continue
        for arm, path in paths.items():
            complete = read(folder / (arm + '_complete.json'))
            digest = file_sha(path)
            if digest != complete['prediction_sha256']:
                raise ValueError(f'Prediction hash mismatch: {path}')
            locked[f'{seed}/{arm}'] = digest
    if missing:
        report = {'dataset': args.dataset, 'requested_seeds': requested, 'status': 'incomplete',
                  'missing': missing, 'test_labels_opened': False,
                  'note': 'No scoring before all requested predictions are complete and frozen.'}
        write(root / 'outputs' / 'core_summary.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    write(root / 'outputs' / 'core_predictions_frozen.json', {
        'dataset': args.dataset, 'requested_seeds': requested, 'prediction_sha256': locked,
        'test_labels_opened_before_lock': False})
    for seed in requested:
        folder = root / 'outputs' / str(seed)
        paths = {arm: folder / (arm + '_predictions.npz') for arm in ['tree70', 'matpv']}
        with np.load(root / 'inputs' / str(seed) / 'oracle_labels.npz', allow_pickle=False) as oracle:
            y = oracle['test_y']
            new = oracle['test_new_normal'] if 'test_new_normal' in oracle.files else None
        row = {'seed': seed, 'arms': {}}
        for arm, path in paths.items():
            if file_sha(path) != locked[f'{seed}/{arm}']:
                raise ValueError(f'Prediction hash mismatch: {path}')
            with np.load(path, allow_pickle=False) as z:
                row['arms'][arm] = metrics(y, z['pred'], z['test_proba'], new)
                row['arms'][arm]['threshold'] = float(z['threshold'])
        write(folder / 'core_results.json', row)
        records.append(row)
    report = {'dataset': args.dataset, 'requested_seeds': selected_seeds(args, config),
              'completed_pairs': len(records), 'missing': missing, 'per_seed': records,
              'status': 'complete' if not missing else 'incomplete', 'summary': {},
              'note': 'Five overlapping retrospective splits; no independent deployment claim.'}
    for arm in ['tree70', 'matpv']:
        report['summary'][arm] = {}
        for key in ['rare_f1', 'guard_subtype_recall', 'focus_subtype_recall', 'focus_precision', 'fpr']:
            values = [r['arms'][arm][key] * 100 for r in records]
            report['summary'][arm][key] = {'values_percent': values,
                'mean_percent': float(np.mean(values)) if values else None,
                'sample_sd_percent': float(np.std(values, ddof=1)) if len(values) > 1 else None}
    write(root / 'outputs' / 'core_summary.json', report)
    print(json.dumps({'status': report['status'], 'completed_pairs': len(records),
                      'summary_file': str(root / 'outputs' / 'core_summary.json')}, ensure_ascii=False, indent=2))


def environment_report():
    out = {'python': sys.version.split()[0], 'packages': {}, 'scope': 'Current portable package environment, not claimed as historical environment.'}
    for module, distribution in [('numpy', 'numpy'), ('scipy', 'scipy'), ('sklearn', 'scikit-learn'), ('pandas', 'pandas'), ('threadpoolctl', 'threadpoolctl')]:
        try:
            importlib.import_module(module)
            out['packages'][distribution] = importlib.metadata.version(distribution)
        except Exception as exc:
            out['packages'][distribution] = {'error': type(exc).__name__ + ': ' + str(exc)}
    return out


def parser():
    p = argparse.ArgumentParser(description='MAT-PV核心复现入口。只有run命令训练模型，prepare构造输入，evaluate读取新运行结果。')
    commands = p.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='检查依赖导入和版本，不训练')
    commands.add_parser('check-package', help='核对代码包SHA256，不训练')
    for command in ['init', 'check-data', 'prepare', 'verify-splits', 'run', 'evaluate', 'status']:
        sub = commands.add_parser(command)
        sub.add_argument('--dataset', required=True, choices=DATASETS)
        sub.add_argument('--run-root', default=str(PACKAGE_ROOT / 'runs'))
        if command == 'init':
            sub.add_argument('--data-root', required=True)
        if command in ['run', 'evaluate']:
            sub.add_argument('--seed', nargs='+', type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'doctor':
        report = environment_report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if any(isinstance(v, dict) for v in report['packages'].values()) else 0
    if args.command == 'check-package':
        from .integrity import verify_package
        print('PACKAGE_HASHES_OK', verify_package())
        return 0
    if args.command == 'init':
        root = run_directory(args)
        # Avoid writing into the shipped code, configurations or saved results.
        if root == PACKAGE_ROOT or any(root.is_relative_to(PACKAGE_ROOT / part) for part in ['configs', 'reference', 'metadata', 'matpv_repro', 'provenance', 'docs', 'environment']):
            raise ValueError('Choose a separate writable run directory.')
        template = read(PACKAGE_ROOT / 'configs' / (args.dataset + '.json'))
        data_root = Path(args.data_root).expanduser().resolve()
        for entry in template['source_files']:
            entry['path'] = str(data_root / entry['relative_path'])
        path = root / 'config.json'
        if path.exists() and read(path) != template:
            raise ValueError('Existing run configuration differs; choose another run-root.')
        write(path, template)
        print('INITIALIZED', root)
        return 0
    root, config = initialized(args)
    if args.command == 'check-data':
        print('RAW_SHA256_OK', source_check(config))
    elif args.command == 'prepare':
        from .integrity import verify_package, verify_splits
        verify_package()
        source_check(config)
        common = runtime(root)
        module = importlib.import_module('.workflow.' + ('prepare_ton' if args.dataset == 'ton_iot' else 'prepare_external'), __package__)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            module.prepare()
        print('FROZEN_SPLITS_VERIFIED', verify_splits(args.dataset, root))
    elif args.command == 'verify-splits':
        from .integrity import verify_splits
        verified = verify_splits(args.dataset, root)
        if not verified:
            raise ValueError('No prepared inputs; no split has been verified. Run prepare first.')
        print('FROZEN_SPLITS_VERIFIED', verified)
    elif args.command == 'run':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'status':
        print(json.dumps({'dataset': args.dataset, 'run_directory': str(root), 'seeds': [
            {'seed': seed, 'prepared': (root / 'inputs' / str(seed) / 'audit.json').exists(),
             'tree70_complete': (root / 'outputs' / str(seed) / 'tree70_complete.json').exists(),
             'matpv_complete': (root / 'outputs' / str(seed) / 'matpv_complete.json').exists()}
            for seed in config['seeds']]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        print('ERROR:', str(exc), file=sys.stderr)
        raise SystemExit(2)
