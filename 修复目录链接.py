#!/usr/bin/env python3
"""Restore only the two known relative project directory links after ZIP extraction."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LINKS={'实验/程序/data':'../数据','实验/程序/artifacts':'../结果'}

def main():
    for rel,target in LINKS.items():
        path=ROOT/rel
        if path.is_symlink():
            if path.readlink().as_posix()!=target:raise RuntimeError(f'Unexpected existing link: {rel}')
            print(f'OK {rel} -> {target}')
            continue
        destination=path.parent/target
        if not destination.is_dir():raise RuntimeError(f'Missing target directory: {target}')
        if path.exists():
            if not path.is_file() or path.stat().st_size>128 or path.read_text(encoding='utf-8').strip()!=target:
                raise RuntimeError(f'Will not overwrite existing content: {rel}')
            path.unlink()
        try:path.symlink_to(target,target_is_directory=True)
        except OSError as error:raise RuntimeError(f'Cannot create directory link {rel}; enable symbolic-link permission or use explicit paths in the instructions') from error
        print(f'Created {rel} -> {target}')

if __name__=='__main__':main()
