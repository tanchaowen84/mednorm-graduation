#!/usr/bin/env python3
"""Verify saved predictions; optionally verify all source files in this package."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True

def digest(path: Path) -> str:
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-files',action='store_true')
    parser.add_argument('--models',action='store_true',help='Verify the 151 official model files in the complete package')
    args=parser.parse_args()
    if sys.version_info[:2]<(3,11):
        raise SystemExit('Use Python 3.11 or newer for the saved-result check.')
    root=Path(__file__).resolve().parent
    program=root/'实验/程序'
    sys.path.insert(0,str(program/'src'))
    source=program/'scripts/validate_official_dev_result_v3.py'
    spec=importlib.util.spec_from_file_location('mednorm_handoff_validator',source)
    if spec is None or spec.loader is None:raise RuntimeError('Validator could not be loaded')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result=module.validate_result(root/'实验/结果/phase1_v3/strict/official_fulltrain/official_dev_fusion')
    output={'saved_prediction_validation':result}
    for enabled,manifest_name,result_name in [
        (args.all_files,'GitHub资料文件清单.json','source_file_validation'),
        (args.models,'正式模型清单.json','model_file_validation'),
    ]:
        if not enabled:continue
        manifest=json.loads((root/manifest_name).read_text(encoding='utf-8'))
        failures=[]
        for index,item in enumerate(manifest['files'],1):
            rel=Path(item['local_path'])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe manifest path')
            path=root/rel
            if not path.is_file() or path.is_symlink():failures.append({'path':str(rel),'reason':'missing or symbolic link'})
            elif path.stat().st_size!=item['size']:failures.append({'path':str(rel),'reason':'size differs'})
            elif digest(path)!=item['sha256']:failures.append({'path':str(rel),'reason':'SHA-256 differs'})
            if index%50==0:print(f'Checked {index}/{len(manifest["files"])} files',file=sys.stderr,flush=True)
        output[result_name]={'status':'FAIL' if failures else 'PASS','files':len(manifest['files']),'failures':failures}
        if failures:
            print(json.dumps(output,ensure_ascii=False,indent=2));raise SystemExit(1)
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
