#!/bin/zsh
set -eu
cd -- "${0:A:h}/程序"
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -m mednorm.cli verify
if [[ -t 0 ]]; then
    printf '\n复核结束，按回车关闭。\n'
    read -r
fi
