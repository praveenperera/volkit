"""Cuts one AFV v2 column of real CT tiles and runs them through inkrs's `inkrs.AfvV2`, the reference volkit's
`examples/afv2_smoke.rs` is compared with byte for byte

Reads the band `ct[z0:z1, y0:y0+ny, x0:x0+nx]` from a uint8 zarr level, reflects it by the 16-voxel tile border,
cuts 128^3 tiles on the stride-96 lattice (the last tile on each axis shifted to end at the padded edge), drops
all-zero tiles as the air skip, and writes `tiles.u8` and `meta.json` for the column `(nz, ny, nx)` at band
`(0, 0, 0)` into OUT/input. Then runs inkrs.AfvV2 on them at each precision and writes OUT/inkrs-<precision>/
`semantic.u8`, `tracing.u8` and `report.json`. Needs numpy, zarr and an inkrs build with AfvV2 (0.4.0 + bf16)

    python -I scripts/afv2_inkrs_reference.py CT_LEVEL MODEL OUT --z 12032 12544 --y 4000 --x 3900 --edge 384
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zarr

import inkrs

PATCH, BORDER, STRIDE = 128, 16, 96


def starts(extent: int) -> list[int]:
    """Lattice origins on one padded axis: every STRIDE, the last shifted to end at the axis's end"""
    if extent <= PATCH:
        return [0]
    out = list(range(0, extent - PATCH, STRIDE))
    return out + [extent - PATCH] if out[-1] != extent - PATCH else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ct", type=Path)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--z", type=int, nargs=2, required=True)
    ap.add_argument("--y", type=int, required=True)
    ap.add_argument("--x", type=int, required=True)
    ap.add_argument("--edge", type=int, default=384)
    ap.add_argument("--precisions", nargs="+", default=["fp32", "bf16"])
    args = ap.parse_args()

    ct = zarr.open(str(args.ct), mode="r")
    z0, z1 = args.z
    band = np.asarray(ct[z0:z1, args.y : args.y + args.edge, args.x : args.x + args.edge])
    padded = np.pad(band, BORDER, mode="reflect")
    tiles, origins = [], []
    for pz in starts(padded.shape[0]):
        for py in starts(padded.shape[1]):
            for px in starts(padded.shape[2]):
                tile = padded[pz : pz + PATCH, py : py + PATCH, px : px + PATCH]
                if not tile.any():
                    continue
                tiles.append(np.ascontiguousarray(tile))
                origins.append([pz, py, px])
    stack = np.stack(tiles)

    inp = args.out / "input"
    inp.mkdir(parents=True, exist_ok=True)
    stack.tofile(inp / "tiles.u8")
    meta = {"shape": list(band.shape), "y0": 0, "x0": 0, "origins": origins,
            "region": {"ct": str(args.ct), "z": [z0, z1], "y": args.y, "x": args.x, "edge": args.edge}}
    (inp / "meta.json").write_text(json.dumps(meta))
    print(f"{len(tiles)} tiles for column {band.shape}, band mean {band.mean():.1f}")

    for precision in args.precisions:
        net = inkrs.AfvV2(args.model, device=0, algo="find", precision=precision)
        start = time.perf_counter()
        net.begin(band.shape, 0, 0)
        net.add(stack, origins)
        semantic, tracing, uncovered = net.finish(True)
        run_s = time.perf_counter() - start
        out = args.out / f"inkrs-{precision}"
        out.mkdir(parents=True, exist_ok=True)
        np.ascontiguousarray(semantic).tofile(out / "semantic.u8")
        np.ascontiguousarray(tracing).tofile(out / "tracing.u8")
        report = {"load": net.load_report, "tiles": len(tiles), "uncovered": int(uncovered), "run_s": run_s}
        (out / "report.json").write_text(json.dumps(report, indent=1))
        print(f"inkrs {precision}: {len(tiles)} tiles in {run_s:.2f} s, {uncovered} voxels uncovered")


if __name__ == "__main__":
    main()
