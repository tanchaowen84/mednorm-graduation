#!/usr/bin/env python3
"""Download the public MEDNORM package directly into one verified ZIP."""
from __future__ import annotations

import argparse
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE_URL = "https://github.com/tanchaowen84/mednorm-graduation/releases/download/research-handoff-2026-10-04/"
MANIFEST = json.loads('''{
  "schema": 1,
  "original_name": "MEDNORM_public_2026-10-04.zip",
  "original_size": 9977497576,
  "original_sha256": "560bc09549962225eeaceae3d38513170778655d103549070c7261b308b93815",
  "parts": [
    {
      "name": "MEDNORM_public_2026-10-04.zip.001",
      "size": 867509110,
      "sha256": "104298c0c38e2c31bc34af7aba14443e31e0cd3519ec8560efe3c06ff2f0db1c"
    },
    {
      "name": "MEDNORM_public_2026-10-04.zip.002",
      "size": 1073741824,
      "sha256": "2c6f5e423384a50125c5518f259278500ba5749d480cd17cd20963833a5bb67f"
    },
    {
      "name": "MEDNORM_public_2026-10-04.zip.003",
      "size": 1073741824,
      "sha256": "e08cac2a660d0e26d5a532d9f1515cab8336f5df363ee728a908da281cbc31a8"
    },
    {
      "name": "MEDNORM_public_2026-10-04.zip.004",
      "size": 1073741824,
      "sha256": "ae364d570b95151502ab5061c23c32f720222dda0d74326d8116085f08f24e11"
    },
    {
      "name": "MEDNORM_public_2026-10-04.zip.005",
      "size": 1073741824,
      "sha256": "ef0e78f43218e9ac690b4513686978cd8e605e4bc4b39159bd535b081430930d"
    },
    {
      "name": "MEDNORM_public_2026-10-04.zip.006",
      "size": 1073741824,
      "sha256": "a6ed799ab470ea2a0c3ad29488fb5831a21bcc4cb7b837b1d0c2b0077a679a29"
    },
    {
      "name": "MEDNORM_public_2026-10-04.zip.007",
      "size": 1073741824,
      "sha256": "bbe2516a3b10c3b51ac39d764c6d3ed62ae881738e09b17d2d282a4efe5ddb06"
    },
    {
      "name": "MEDNORM_public_2026-10-04.zip.008",
      "size": 1073741824,
      "sha256": "ad83a3a9c48eeb1ce95576fadd4b9204742d92d6745c3e5abf106c19a7598ed6"
    },
    {
      "name": "MEDNORM_public_2026-10-04.zip.009",
      "size": 1073741824,
      "sha256": "f277822927cf7ad84407f90dcd1b9b3ba18c5acc7cb6dfb3fb0271fdde079c89"
    },
    {
      "name": "MEDNORM_public_2026-10-04.zip.010",
      "size": 520053874,
      "sha256": "417ae4d36509076315a8ad894e491bb6459175916f00ff553f4657a51a0f3b0a"
    }
  ]
}''')
BLOCK = 1024 * 1024


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def download_complete(manifest: dict, base_url: str, directory: Path) -> Path:
    """Resume at verified boundaries; never retain separate volume files."""
    names = [manifest["original_name"], *(part["name"] for part in manifest["parts"])]
    if any(Path(name).name != name or name in ("", ".", "..") for name in names):
        raise ValueError("清单中的文件名不合法")
    if sum(part["size"] for part in manifest["parts"]) != manifest["original_size"]:
        raise ValueError("清单大小不符")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / manifest["original_name"]
    if target.exists():
        if target.stat().st_size == manifest["original_size"] and file_digest(target) == manifest["original_sha256"]:
            print("完整 ZIP 已存在且 SHA-256 通过：" + str(target), flush=True)
            return target
        raise FileExistsError("同名 ZIP 内容不同，未覆盖：" + str(target))
    partial = target.with_name(target.name + ".downloading")
    if partial.is_symlink():
        raise ValueError("下载临时文件不能是符号链接")
    whole = hashlib.sha256()
    offset = 0
    # The partial ZIP is the only intermediate data file. It becomes the final ZIP.
    with partial.open("r+b" if partial.exists() else "w+b") as output:
        # Lock the actual data file so two runs cannot write to it at once.
        if os.name == "nt":
            import msvcrt
            output.seek(0)
            msvcrt.locking(output.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(output.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        total = len(manifest["parts"])
        for index, part in enumerate(manifest["parts"], 1):
            output.seek(offset)
            remaining = part["size"]
            digest = hashlib.sha256()
            while remaining:
                block = output.read(min(BLOCK, remaining))
                if not block:
                    break
                digest.update(block)
                remaining -= len(block)
            if remaining == 0 and digest.hexdigest() == part["sha256"]:
                print(f"已验证，继续下载：{index}/{total}", flush=True)
            else:
                output.seek(offset)
                output.truncate()
                required = manifest["original_size"] - offset
                if shutil.disk_usage(directory).free < required + 64 * BLOCK:
                    raise OSError(f"可用空间不足；ZIP 还需约 {required / 2**30:.2f} GiB")
                for attempt in range(1, 4):
                    output.seek(offset)
                    output.truncate()
                    digest = hashlib.sha256()
                    remaining = part["size"]
                    print(f"下载进度：{index}/{total}（{offset / manifest['original_size']:.0%}）", flush=True)
                    request = Request(base_url + quote(part["name"]), headers={"User-Agent": "MEDNORM-complete-downloader/1"})
                    try:
                        with urlopen(request, timeout=60) as response:
                            while remaining:
                                block = response.read(min(BLOCK, remaining))
                                if not block:
                                    raise OSError("下载提前结束")
                                output.write(block)
                                digest.update(block)
                                remaining -= len(block)
                            if response.read(1):
                                raise OSError("下载内容超过清单大小")
                        if digest.hexdigest() != part["sha256"]:
                            raise OSError("下载内容 SHA-256 不符")
                        output.flush()
                        os.fsync(output.fileno())
                        break
                    except (OSError, URLError, HTTPException) as error:
                        output.seek(offset)
                        output.truncate()
                        output.flush()
                        if attempt == 3:
                            raise OSError("下载失败；重新运行会从已校验的位置继续") from error
                        print(f"本段重试 {attempt}/2：{error}", flush=True)
                        time.sleep(attempt)
            output.seek(offset)
            remaining = part["size"]
            while remaining:
                block = output.read(min(BLOCK, remaining))
                if not block:
                    raise OSError("已下载内容缺失")
                whole.update(block)
                remaining -= len(block)
            offset += part["size"]
        output.truncate(offset)
        output.flush()
        os.fsync(output.fileno())
        if offset != manifest["original_size"] or whole.hexdigest() != manifest["original_sha256"]:
            raise ValueError("完整 ZIP SHA-256 不符，未输出正式文件")
        if target.exists():
            raise FileExistsError("输出文件已存在，未覆盖：" + str(target))
    partial.rename(target)
    print("下载完成，完整 ZIP SHA-256 通过：" + str(target), flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="自动下载并输出一个完整 ZIP；不需要手动处理分卷。")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="ZIP 保存目录，默认为当前目录")
    args = parser.parse_args()
    print("完整公开资料与13个正式模型，约9.29 GiB。可中断后重新运行。", flush=True)
    download_complete(MANIFEST, BASE_URL, args.output_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n下载已中断；重新运行会保留并校验已经完成的内容。", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, HTTPException) as error:
        print("错误：" + str(error), file=sys.stderr)
        raise SystemExit(1)
