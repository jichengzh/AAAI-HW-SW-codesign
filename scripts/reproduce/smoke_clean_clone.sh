#!/usr/bin/env bash
# Clone one public ref into a new directory and run the CPU-only smoke workflow.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/reproduce/smoke_clean_clone.sh [options]

Clone a Git ref, build an isolated CPU Python environment, and run the public
smoke workflow plus its integration checks. It does not download datasets,
checkpoints, ONNX files, compiled engines, TVM/TensorRT artifacts, or hardware
measurements. It installs only the declared Python packages.

Options:
  --repo-url URL   Git URL to clone. Defaults to this checkout's origin remote.
  --ref REF        Branch or tag to clone. Defaults to this checkout's branch.
  --work-dir PATH  Empty directory to use. Defaults to a new /tmp directory.
  --python PATH    Python executable for the virtual environment (default: python3).
  --skip-install   Reuse the selected Python; dependencies must already exist.
  --help           Show this help text.

The work directory is retained so a failed run can be inspected safely.
EOF
}

die() {
  printf 'smoke_clean_clone: %s\n' "$*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL=""
REF=""
WORK_DIR=""
PYTHON_BIN="python3"
SKIP_INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url)
      [[ $# -ge 2 ]] || die '--repo-url requires a value'
      REPO_URL="$2"
      shift 2
      ;;
    --ref)
      [[ $# -ge 2 ]] || die '--ref requires a value'
      REF="$2"
      shift 2
      ;;
    --work-dir)
      [[ $# -ge 2 ]] || die '--work-dir requires a value'
      WORK_DIR="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die '--python requires a value'
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

command -v git >/dev/null 2>&1 || die 'git is required'
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 14), sys.version'
SOURCE_ROOT="$(git -C "$SCRIPT_DIR/../.." rev-parse --show-toplevel)"

if [[ -z "$REPO_URL" ]]; then
  REPO_URL="$(git -C "$SOURCE_ROOT" remote get-url origin)"
fi
if [[ -z "$REF" ]]; then
  REF="$(git -C "$SOURCE_ROOT" symbolic-ref --quiet --short HEAD || true)"
  [[ -n "$REF" ]] || die 'detached HEAD: pass --ref explicitly'
fi

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$(mktemp -d -p "${TMPDIR:-/tmp}" gear-clean-smoke.XXXXXX)"
else
  if [[ -e "$WORK_DIR" ]] && [[ -n "$(find "$WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be empty: $WORK_DIR"
  fi
  mkdir -p "$WORK_DIR"
fi

CLONE_ROOT="$WORK_DIR/repo"
printf 'smoke_clean_clone: cloning %s at %s\n' "$REPO_URL" "$REF"
git clone --depth 1 --filter=blob:none --single-branch --branch "$REF" "$REPO_URL" "$CLONE_ROOT"

if [[ "$SKIP_INSTALL" -eq 1 ]]; then
  RUN_PYTHON="$PYTHON_BIN"
else
  VENV_DIR="$WORK_DIR/venv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  RUN_PYTHON="$VENV_DIR/bin/python"
  "$RUN_PYTHON" -m pip install --upgrade pip
  "$RUN_PYTHON" -m pip install -r "$CLONE_ROOT/requirements.txt"
  "$RUN_PYTHON" -m pip install --no-deps -e "$CLONE_ROOT"
fi

"$RUN_PYTHON" "$CLONE_ROOT/scripts/reproduce/reproduce_all.py" \
  --mode smoke --output-root "$WORK_DIR/repro-smoke-output"
"$RUN_PYTHON" -m pytest -q \
  "$CLONE_ROOT/tests/integration/test_reproduce_all.py" \
  "$CLONE_ROOT/tests/integration/test_public_clean_clone.py"

printf 'smoke_clean_clone: PASS work_dir=%s commit=%s\n' \
  "$WORK_DIR" "$(git -C "$CLONE_ROOT" rev-parse HEAD)"
