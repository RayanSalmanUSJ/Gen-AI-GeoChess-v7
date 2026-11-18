"""Download the pinned official Fishtest archive; verify before committing."""

import argparse
from pathlib import Path
import urllib.request
from chess_policy.data import sha256

URL = "https://huggingface.co/datasets/official-stockfish/fishtest_pgns/resolve/main/25-01-01/677492ae1a2f267f20548afc/677492ae1a2f267f20548afc.pgn.gz"
SHA256 = "bef40277bf1996beb8a025258922761ad4c52a47a294b2b8869ed65b78189e09"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/source.pgn.gz"))
    a = p.parse_args()
    if a.out.exists():
        if sha256(a.out) != SHA256:
            raise ValueError("existing archive checksum mismatch; choose a different path")
        print("Existing source verified.")
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.out.with_suffix(".part")
    with urllib.request.urlopen(URL, timeout=60) as source, temporary.open("wb") as target:
        while block := source.read(1024 * 1024):
            target.write(block)
    if sha256(temporary) != SHA256:
        raise ValueError("download checksum mismatch")
    temporary.replace(a.out)
    print(f"Verified {a.out.stat().st_size} bytes.")


if __name__ == "__main__":
    main()
