"""Streaming PGNs, opening-group partitions, disk deduplication, mmap training shards."""

from bisect import bisect_right
from collections import OrderedDict
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import chess.pgn
import numpy as np
from torch.utils.data import Dataset, Sampler
from .encoding import ENCODING_VERSION, encode, fingerprint, move_id, position_key


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_manifest(root, verify=False):
    root = Path(root)
    obj = json.loads((root / "manifest.json").read_text())
    if obj["schema_version"] != 1 or obj["encoding_version"] != ENCODING_VERSION:
        raise ValueError("incompatible dataset format")
    for shard in obj["shards"]:
        for entry in shard["files"].values():
            file = root / entry["path"]
            if not file.resolve().is_relative_to(root.resolve()):
                raise ValueError("dataset path escapes root")
            if verify and sha256(file) != entry["sha256"]:
                raise ValueError(f"dataset checksum mismatch: {file}")
    return obj


def build_corpus(
    paths,
    output,
    *,
    max_games=0,
    shard_size=4096,
    seed="fishtest-v2",
    force_test=False,
    exclude=None,
):
    """Manifest is written last and signifies completion. Invalid input fails closed.

    No moves or labels from another split enter a model input. First occurrence
    wins global deduplication (clocks ignored); opening FEN groups choose splits.
    """
    if shard_size < 1 or max_games < 0:
        raise ValueError("invalid corpus limits")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("corpus output must be empty")
    database = sqlite3.connect(output / "dedup.sqlite3")
    database.execute("CREATE TABLE positions (key BLOB PRIMARY KEY) WITHOUT ROWID")
    excluded_groups = set()
    if exclude is not None:
        old = read_manifest(exclude, verify=True)
        for shard in old["shards"]:
            keys = np.load(Path(exclude) / shard["files"]["keys"]["path"], mmap_mode="r")
            database.executemany(
                "INSERT OR IGNORE INTO positions VALUES (?)", [(bytes(k),) for k in keys]
            )
        excluded_groups = {g["group"] for g in old["games"]}
    buffers = {split: [] for split in ("train", "dev", "test")}
    counts = dict.fromkeys(buffers, 0)
    shards, games = [], []
    duplicates = excluded = read_games = 0
    started = time.monotonic()

    def flush(split):
        rows = buffers[split]
        if not rows:
            return
        directory = output / f"{split}_{sum(x['split'] == split for x in shards):05d}"
        directory.mkdir()
        legal = np.full((len(rows), max(len(x["legal"]) for x in rows)), -1, dtype=np.int16)
        for i, row in enumerate(rows):
            legal[i, : len(row["legal"])] = row["legal"]
        arrays = {k: np.stack([x[k] for x in rows]) for k in ("pieces", "geometry", "attacks")}
        arrays.update(
            legal=legal,
            target=np.array([x["target"] for x in rows], dtype=np.int16),
            keys=np.array([x["key"] for x in rows], dtype="V16"),
            groups=np.array([x["group"] for x in rows], dtype="V16"),
        )
        files = {}
        for key, array in arrays.items():
            file = directory / f"{key}.npy"
            np.save(file, array, allow_pickle=False)
            files[key] = {"path": str(file.relative_to(output)), "sha256": sha256(file)}
        shards.append({"split": split, "rows": len(rows), "files": files})
        buffers[split] = []

    try:
        for source in map(Path, paths):
            opener = gzip.open if str(source).endswith(".gz") else open
            with opener(source, "rt", encoding="utf-8", errors="strict") as stream:
                while not max_games or read_games < max_games:
                    game = chess.pgn.read_game(stream)
                    if game is None:
                        break
                    read_games += 1
                    board = game.board()
                    if game.errors or board.chess960 or not board.is_valid():
                        raise ValueError(f"invalid PGN game {read_games}; corpus not committed")
                    if game.headers.get("Variant", "Standard") not in ("Standard", "Chess"):
                        raise ValueError("only orthodox chess is supported")
                    opening = position_key(board)
                    group = hashlib.blake2b(opening.encode(), digest_size=16).digest()
                    if group.hex() in excluded_groups:
                        excluded += 1
                        continue
                    bucket = (
                        int.from_bytes(
                            hashlib.sha256((seed + opening).encode()).digest()[:4], "big"
                        )
                        % 100
                    )
                    split = (
                        "test"
                        if force_test
                        else ("train" if bucket < 85 else "dev" if bucket < 92 else "test")
                    )
                    games.append(
                        {
                            "source": source.name,
                            "ordinal": read_games,
                            "group": group.hex(),
                            "split": split,
                            "opening": opening,
                            "white": game.headers.get("White", ""),
                            "black": game.headers.get("Black", ""),
                        }
                    )
                    for move in game.mainline_moves():
                        if move not in board.legal_moves:
                            raise ValueError("illegal PGN mainline move")
                        key = fingerprint(board)
                        inserted = database.execute(
                            "INSERT OR IGNORE INTO positions VALUES (?)", (key,)
                        ).rowcount
                        if inserted:
                            item = encode(board, validate=False)
                            item.update(target=move_id(move), key=key, group=group)
                            buffers[split].append(item)
                            counts[split] += 1
                            if len(buffers[split]) >= shard_size:
                                flush(split)
                        else:
                            duplicates += 1
                        board.push(move)
                    if read_games % 100 == 0:
                        database.commit()
                        print(
                            json.dumps(
                                {
                                    "games": read_games,
                                    "counts": counts,
                                    "seconds": round(time.monotonic() - started, 1),
                                }
                            ),
                            flush=True,
                        )
            if max_games and read_games >= max_games:
                break
        for split in buffers:
            flush(split)
        database.commit()
    finally:
        database.close()
    if not sum(counts.values()):
        raise ValueError("no accepted labeled positions")
    obj = {
        "schema_version": 1,
        "encoding_version": ENCODING_VERSION,
        "sources": [{"name": Path(p).name, "sha256": sha256(p)} for p in paths],
        "config": {
            "seed": seed,
            "max_games": max_games,
            "force_test": force_test,
            "exclude_sha256": sha256(Path(exclude) / "manifest.json") if exclude else None,
        },
        "counts": counts,
        "games": games,
        "read_games": read_games,
        "duplicates_removed": duplicates,
        "excluded_opening_games": excluded,
        "shards": shards,
        "seconds": time.monotonic() - started,
    }
    atomic_json(output / "manifest.json", obj)
    (output / "dedup.sqlite3").unlink()
    return obj


class ShardDataset(Dataset):
    def __init__(self, root, split, verify=False):
        self.root = Path(root)
        self.manifest = read_manifest(self.root, verify)
        self.shards = [x for x in self.manifest["shards"] if x["split"] == split]
        self.ends = np.cumsum([s["rows"] for s in self.shards]).tolist()
        self.cache = OrderedDict()
        if not self.ends:
            raise ValueError(f"empty split {split}")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        offset = index - (self.ends[shard - 1] if shard else 0)
        if shard not in self.cache:
            self.cache[shard] = {
                k: np.load(self.root / v["path"], mmap_mode="r")
                for k, v in self.shards[shard]["files"].items()
            }
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        arrays = self.cache[shard]
        item = {k: arrays[k][offset] for k in ("pieces", "geometry", "attacks", "target")}
        legal = arrays["legal"][offset]
        item.update(legal=legal[legal >= 0], group=bytes(arrays["groups"][offset]).hex())
        return item


class EpochSampler(Sampler):
    def __init__(self, dataset, seed, epoch, max_rows=0):
        self.dataset, self.seed, self.epoch = dataset, seed, epoch
        self.count = min(max_rows or len(dataset), len(dataset))

    def __len__(self):
        return self.count

    def __iter__(self):
        # Same order as concatenating shard permutations, without an O(rows) array.
        rng = np.random.default_rng(self.seed + self.epoch)
        starts = [0] + self.dataset.ends[:-1]
        remaining = self.count
        for shard in rng.permutation(len(self.dataset.shards)):
            for offset in rng.permutation(self.dataset.shards[shard]["rows"]):
                if remaining == 0:
                    return
                yield int(starts[shard] + offset)
                remaining -= 1
