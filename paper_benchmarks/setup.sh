#!/usr/bin/env bash
# Install the fixed paper environment. No experiment is run by this script.
set -euo pipefail
if [ "$#" -ne 0 ]; then
  echo 'setup.sh accepts no arguments.' >&2
  exit 2
fi
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
  echo 'The benchmark environment requires Linux x86_64.' >&2
  exit 2
fi
benchmark_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
for program in python3.12 git curl tar sha256sum cc cargo rustc; do
  command -v "$program" >/dev/null || { echo "Missing prerequisite: $program" >&2; exit 2; }
done
python3.12 - <<'PY'
import platform
import subprocess
import sys
if platform.python_implementation() != 'CPython' or sys.version_info[:2] != (3, 12):
    raise SystemExit('Setup requires CPython 3.12.')
libc, version = platform.libc_ver()
if libc != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 28):
    raise SystemExit('The pinned Linux wheels require glibc >= 2.28.')
rust = subprocess.check_output(['rustc', '--version'], text=True).split()[1]
if tuple(map(int, rust.split('.')[:2])) < (1, 85):
    raise SystemExit('FaultScope requires Rust >= 1.85.')
PY
python3.12 -m venv "$benchmark_root/.venv"
"$benchmark_root/.venv/bin/python" -m pip install pip==26.2.1 maturin==1.14.1
# FaultScope is the only source-built Python package. Keep its build backend
# fixed as well as the runtime dependency closure.
"$benchmark_root/.venv/bin/python" -m pip install --no-build-isolation -r "$benchmark_root/requirements-lock.txt"
"$benchmark_root/.venv/bin/python" -m pip check
mkdir -p "$benchmark_root/runtime"
julia_program="$benchmark_root/runtime/julia-1.12.6/bin/julia"
if [ ! -x "$julia_program" ]; then
  if command -v julia >/dev/null && [ "$(julia --startup-file=no --version)" = 'julia version 1.12.6' ]; then
    julia_program=$(command -v julia)
  else
    julia_archive='julia-1.12.6-linux-x86_64.tar.gz'
    download_dir=$(mktemp -d "$benchmark_root/runtime/download.XXXXXXXX")
    trap 'rm -rf -- "$download_dir"' EXIT
    curl --fail --location --retry 3 "https://julialang-s3.julialang.org/bin/linux/x64/1.12/$julia_archive" -o "$download_dir/$julia_archive"
    curl --fail --location --retry 3 'https://julialang-s3.julialang.org/bin/checksums/julia-1.12.6.sha256' -o "$download_dir/SHA256SUMS"
    (cd "$download_dir"; awk -v name="$julia_archive" '$2 == name { print }' SHA256SUMS > selected.sha256
      test -s selected.sha256; sha256sum --check selected.sha256)
    tar -xzf "$download_dir/$julia_archive" -C "$benchmark_root/runtime"
  fi
fi
JULIA_DEPOT_PATH="$benchmark_root/runtime/julia-depot" "$julia_program" \
  --project="$benchmark_root/_support/julia" --threads=1 --startup-file=no \
  -e 'using Pkg; Pkg.instantiate(); using QuantumClifford; @assert string(Base.pkgversion(QuantumClifford)) == "0.11.5"'
PYTHONPATH="$benchmark_root/_support" "$benchmark_root/.venv/bin/python" -c \
  'from paper_runtime import require_dependencies; require_dependencies(("faultscope", "stim", "qiskit-aer", "cirq", "symft", "quantumclifford")); print("Paper benchmark environment ready.")'
"$benchmark_root/.venv/bin/python" -m pip freeze > "$benchmark_root/runtime/installed-python.txt"
