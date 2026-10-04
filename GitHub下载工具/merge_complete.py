#!/usr/bin/env python3
"""Join the downloaded parts and validate the original ZIP's SHA-256."""
from pathlib import Path
import hashlib,json,os

ROOT=Path(__file__).resolve().parent

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 manifest=json.loads((ROOT/'parts_manifest.json').read_text(encoding='utf-8'))
 target=ROOT/manifest['original_name']
 if target.name!=manifest['original_name']:raise ValueError('Unexpected output name')
 if target.exists():
  if target.stat().st_size==manifest['original_size'] and digest(target)==manifest['original_sha256']:
   print('完整 ZIP 已存在且 SHA-256 通过，可直接解压：'+target.name);return
  raise FileExistsError('已有同名文件内容不同，请先自行确认：'+target.name)
 for item in manifest['parts']:
  if Path(item['name']).name!=item['name']:raise ValueError('Unexpected part name')
  p=ROOT/item['name']
  if not p.is_file() or p.stat().st_size!=item['size']:raise ValueError('分卷缺失或大小不符：'+item['name'])
 temp=target.with_name(target.name+f'.merging-{os.getpid()}');whole=hashlib.sha256();count=0
 try:
  with temp.open('xb') as output:
   for item in manifest['parts']:
    h=hashlib.sha256();p=ROOT/item['name'];print('合并：'+p.name,flush=True)
    with p.open('rb') as source:
     for b in iter(lambda:source.read(8*1024*1024),b''):
      output.write(b);whole.update(b);h.update(b);count+=len(b)
    if h.hexdigest()!=item['sha256']:raise ValueError('分卷 SHA-256 不符，请重新下载：'+p.name)
  if count!=manifest['original_size'] or whole.hexdigest()!=manifest['original_sha256']:raise ValueError('合并后的 ZIP 校验不符')
  if target.exists():raise FileExistsError('输出文件已存在，未覆盖：'+target.name)
  temp.rename(target)
  print('合并成功，原始 ZIP 的 SHA-256 通过。请解压：'+target.name)
 finally:
  if temp.exists():temp.unlink()

if __name__=='__main__':main()
