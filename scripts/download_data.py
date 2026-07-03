"""Download the Stanford 3D Scanning Repository meshes used as GT models.

These are the canonical, freely-downloadable benchmark models (Bunny, Armadillo,
Dragon, Happy Buddha).  Synthetic sphere/torus are generated in prepare_data.py
and need no download.  Files land in data/meshes_raw/.

Note: graphics.stanford.edu is reachable directly but can be slow from some
regions; each file is retried a few times.  If a download keeps failing, fetch
the URL manually into data/meshes_raw/ and re-run prepare_data.py.
"""

from __future__ import annotations

import os
import subprocess
import tarfile
import gzip
import shutil

RAW = "data/meshes_raw"
FILES = {
    "bunny.tar.gz": "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz",
    "Armadillo.ply.gz": "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz",
    "dragon_recon.tar.gz": "http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz",
    "happy_recon.tar.gz": "http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz",
}


def fetch(fname, url, retries=4):
    dst = os.path.join(RAW, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[have] {fname}")
        return dst
    for i in range(retries):
        print(f"[get ] {fname} (try {i+1}/{retries}) ...")
        rc = subprocess.call(["curl", "-fsSL", "--max-time", "900", "-o", dst, url])
        if rc == 0 and os.path.getsize(dst) > 0:
            print(f"[ok  ] {fname}  ({os.path.getsize(dst)//1024} KB)")
            return dst
    print(f"[FAIL] {fname} -- fetch manually from {url}")
    return None


def extract(fname):
    path = os.path.join(RAW, fname)
    if not os.path.exists(path):
        return
    if fname.endswith(".tar.gz"):
        with tarfile.open(path) as t:
            t.extractall(RAW)
    elif fname.endswith(".ply.gz"):
        out = path[:-3]
        with gzip.open(path, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)


def main():
    os.makedirs(RAW, exist_ok=True)
    for fname, url in FILES.items():
        if fetch(fname, url):
            try:
                extract(fname)
            except Exception as e:
                print(f"[warn] extract {fname}: {e!r}")
    # report which GT plys are available
    keys = {
        "bunny": f"{RAW}/bunny/reconstruction/bun_zipper.ply",
        "armadillo": f"{RAW}/Armadillo.ply",
        "dragon": f"{RAW}/dragon_recon/dragon_vrip.ply",
        "happy": f"{RAW}/happy_recon/happy_vrip.ply",
    }
    print("\navailable meshes:")
    for k, p in keys.items():
        print(f"  {k:10s}: {'OK' if os.path.exists(p) else 'MISSING'}  ({p})")


if __name__ == "__main__":
    main()
