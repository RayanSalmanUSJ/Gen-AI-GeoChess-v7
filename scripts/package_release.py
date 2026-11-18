"""Create a compact release with per-file hashes and verify the resulting archive."""

import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from chess_policy.data import sha256


def main():
    root = Path(__file__).resolve().parents[1]
    skipped = {
        "corpus",
        "holdout",
        "runs",
        "v4_holdout",
        "v4_runs",
        "v5_holdout",
        "v5_runs",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "build",
        "dist",
    }
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(
            part in skipped or part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        if path.name in {"source.pgn.gz", "release_manifest.json", "smoke.pt", "smoke.pt.sha256"}:
            continue
        files.append(path)
    manifest = {
        str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files
    }
    (root / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = root.parent / "Chess_Policy_System_v5.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=6) as output:
        for path in files + [root / "release_manifest.json"]:
            output.write(path, Path(root.name) / path.relative_to(root))
    with ZipFile(archive) as output:
        assert output.testzip() is None
        import hashlib

        for name, item in manifest.items():
            data = output.read(str(Path(root.name) / name))
            assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"]
    print(
        json.dumps(
            {
                "archive": str(archive),
                "files": len(files) + 1,
                "bytes": archive.stat().st_size,
                "sha256": sha256(archive),
                "all_archive_entries_verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
