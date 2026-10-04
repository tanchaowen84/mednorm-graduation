#!/usr/bin/env python3
"""Check official raw inputs and rebuild the fixed local data/KG artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROGRAM = Path(__file__).resolve().parents[1]
DATA = PROGRAM.parent / '数据'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((DATA / '原始数据校验清单.json').read_text(encoding='utf-8'))
    failures = []
    for item in manifest['files']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe raw-data manifest path')
        path = args.data_root / rel
        if not path.is_file():
            failures.append(f'{rel}: missing; obtain from the official source in 实验/数据/README.md')
        elif path.stat().st_size != item['size'] or digest(path) != item['sha256']:
            failures.append(f'{rel}: file does not match the locked official dataset')
    if failures:
        raise SystemExit('\n'.join(failures))
    print('PASS: all 5 official raw inputs match the locked SHA-256 values', flush=True)
    if args.check_only:
        return
    output = args.output_root or args.data_root / 'processed' / 'phase1'
    sys.path.insert(0, str(PROGRAM / 'src'))
    from mednorm.prepare import prepare_phase1_data
    from mednorm.kg import build_kg_index
    prepare_phase1_data(raw_dir=args.data_root / 'raw/chip-cdn', output_dir=output,
                        val_fraction=0.1, seed=2026)
    build_kg_index(kg_path=args.data_root / 'raw/cpubmedkg/CPubMed-KGv2_0.txt',
                   catalog_path=output / 'icd_catalog.jsonl', output_dir=output / 'kg_index')
    print(f'Prepared local data and KG index: {output}')


if __name__ == '__main__':
    main()
