"""Check frozen algorithm ASTs without dependencies, training or raw datasets."""
from pathlib import Path
import ast, hashlib, json
ROOT = Path(__file__).resolve().parents[1]
proof = json.loads((ROOT/'provenance/algorithm_ast.json').read_text(encoding='utf-8'))
for item in proof['functions']:
    tree = ast.parse((ROOT/item['file']).read_text(encoding='utf-8'))
    node = next(f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == item['function'])
    actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if actual != item['ast_sha256']:
        raise SystemExit('Algorithm changed: '+item['file']+'/'+item['function'])
print('FROZEN_ALGORITHM_FUNCTIONS_OK', len(proof['functions']))
