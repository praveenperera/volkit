#!/usr/bin/env bash
# Fails when the committed PTX is stale: volkit/ptx/sources.sha256 records the hashes of the kernel crate's
# sources, manifest, lockfile and toolchain pin, the build script, the tool versions it built with
# (volkit/ptx/tools.txt), the libdevice functions the kernels link (volkit/ptx/libdevice.txt), the PTX it
# produced, one file per tier volkit/src/gpu.rs embeds, and the cubins build-cubin.sh compiled from it with
# the ptxas volkit/ptx/ptxas.txt names. Also fails when THIRD_PARTY_NOTICES names other libdevice
# functions than that list. Needs no CUDA toolkit or cuda-oxide
#
#   scripts/check-ptx.sh            # check
#   scripts/check-ptx.sh --write    # record the current files (build-cubin.sh does this)
set -euo pipefail

cd "$(dirname "$0")/.."
manifest=volkit/ptx/sources.sha256
crate=volkit-kernels
ptx=volkit/ptx
gpu=volkit/src/gpu.rs
mapfile -t sources < <(find "$crate/src" -name '*.rs' | LC_ALL=C sort)
files=("${sources[@]}" "$crate/Cargo.toml" "$crate/Cargo.lock" "$crate/rust-toolchain.toml"
  scripts/build-ptx.sh scripts/build-cubin.sh "$ptx/tools.txt" "$ptx/libdevice.txt" "$ptx"/*.ptx "$ptx/ptxas.txt"
  "$ptx"/*.cubin)

# the libdevice functions THIRD_PARTY_NOTICES names are exactly the ones build-ptx.sh found linked, so the
# notice covers what the PTX ships
if ! diff <(sed -n '/^2\. NVIDIA libdevice/,/^=====/p' THIRD_PARTY_NOTICES | grep -o '__nv_[A-Za-z0-9_]*' | LC_ALL=C sort -u) \
  "$ptx/libdevice.txt"; then
  echo "THIRD_PARTY_NOTICES section 2 and $ptx/libdevice.txt name different libdevice functions" >&2
  exit 1
fi

# the PTX files on disk are exactly the tiers volkit/src/gpu.rs embeds, so a tier can't be built but not shipped
# or shipped from a stale file the manifest no longer covers
if ! diff <(printf '%s\n' "$ptx"/*.ptx | LC_ALL=C sort) \
  <(grep -o 'include_str!("\.\./ptx/[^"]*")' "$gpu" | sed 's|include_str!("\.\./\(.*\)")|volkit/\1|' | LC_ALL=C sort); then
  echo "$ptx/*.ptx and the PTX $gpu embeds differ" >&2
  exit 1
fi

# likewise the cubins: each one on disk is embedded, and each embedded one is on disk
if ! diff <(printf '%s\n' "$ptx"/*.cubin | LC_ALL=C sort) \
  <(grep -o 'include_bytes!("\.\./ptx/[^"]*\.cubin")' "$gpu" | sed 's|include_bytes!("\.\./\(.*\)")|volkit/\1|' | LC_ALL=C sort); then
  echo "$ptx/*.cubin and the cubins $gpu embeds differ" >&2
  exit 1
fi

if [[ "${1:-}" == "--write" ]]; then
  sha256sum "${files[@]}" > "$manifest"
  exit 0
fi

if ! diff <(sha256sum "${files[@]}") "$manifest"; then
  echo "ptx is stale: run scripts/build-ptx.sh, or scripts/build-cubin.sh when only the cubins are" >&2
  exit 1
fi
echo "ptx matches its sources"
