#!/usr/bin/env bash
# 在源码仓库内运行，输出差异时保留非零退出状态。
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root=$PWD
check_tmp=$(mktemp -d /tmp/atg-check.XXXXXXXX)
trap 'rm -rf "$check_tmp"' EXIT
export MUJOCO_GL=egl SOURCE_DATE_EPOCH=1700000000
export LP_NUM_THREADS=${LP_NUM_THREADS:-2}

uv run ruff check .
uv run ruff format --check .
uv run pyright

git status --porcelain > "$check_tmp/status"
find config meshes mjcf urdf srdf -type f -print0 | sort -z | xargs -0 sha256sum > "$check_tmp/assets"
uv run python tools/build_description.py
sha256sum --check --quiet "$check_tmp/assets"
git status --porcelain | diff -u "$check_tmp/status" -
echo '资产重建通过，文件摘要与 Git 状态一致。'

sha256sum docs/{figures,images}/* > "$check_tmp/images"
uv run python tools/render_previews.py
uv run python tools/render_figures.py
sha256sum --check --quiet "$check_tmp/images"
uv run python tools/render_figures.py --out-dir "$check_tmp/figures"
diff -qr docs/figures "$check_tmp/figures"
git status --porcelain | diff -u "$check_tmp/status" -
echo '预览与出版图重建通过，两次导出的三张出版图逐字节一致。'

uv run pytest

# 从源码树外验证安装目录，使用本轮锁定版本，仅由 wheel 声明决定安装哪些包。
uv build --wheel --out-dir "$check_tmp/dist"
uv export --locked --no-dev --no-emit-project --no-hashes -o "$check_tmp/constraints.txt" >/dev/null
(
    cd "$check_tmp"
    unset PYTHONPATH
    uv venv --python 3.12 wheel-env
    uv pip install --python wheel-env/bin/python --constraint constraints.txt dist/*.whl
    package=$(wheel-env/bin/python -c 'import adaptive_tactile_gripper_description as p; print(p.__path__[0])')
    cat > pyrightconfig.json <<'JSON'
{"typeCheckingMode":"basic","pythonVersion":"3.12","venvPath":".","venv":"wheel-env"}
JSON
    uv run --project "$repo_root" pyright --project pyrightconfig.json "$package"
    wheel-env/bin/python - <<'PYTHON'
from typing import get_type_hints

import mujoco
from adaptive_tactile_gripper_description import build_gripper_spec

assert get_type_hints(build_gripper_spec)["return"] is mujoco.MjSpec
PYTHON
    wheel-env/bin/atg-description render --out-dir wheel-figures
    diff -qr "$repo_root/docs/figures" wheel-figures
)
echo '独立安装的 wheel 通过整包类型检查、返回类型反射及三张出版图逐字节校验。'
if [[ -d /opt/ros/jazzy ]]; then
    source /opt/ros/jazzy/setup.bash
    cd "$check_tmp"
    colcon build --base-paths "$repo_root" --packages-select adaptive_tactile_gripper_description \
        --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
fi
