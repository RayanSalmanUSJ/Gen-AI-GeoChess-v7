"""Exact terminal-action curriculum. Proofs are computed offline, never at inference."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import chess
from torch.utils.data import Dataset

from .corrections import read_keys
from .data import atomic_json, sha256
from .encoding import encode, fingerprint, move_id
from .geometry import D4


def mate_moves(board):
    result = []
    for move in list(board.legal_moves):
        if not board.gives_check(move):
            continue
        board.push(move)
        if board.is_checkmate():
            result.append(move.uci())
        board.pop()
    return sorted(result)


def orbit_group(board):
    """For pawnless/castling-free examples: group color-normalized D4 equivalents."""
    pieces = list(board.piece_map().items())
    forms = []
    for permutation in D4:
        forms.append(
            tuple(
                sorted(
                    (p.piece_type, p.color == board.turn, int(permutation[s])) for s, p in pieces
                )
            )
        )
    return hashlib.sha256(repr(min(forms)).encode()).hexdigest()


def generate(
    output,
    protected_roots,
    *,
    train_rows=2500,
    dev_rows=500,
    seed=5221,
    test_rows=0,
    exclude_curricula=(),
):
    if min(train_rows, dev_rows, test_rows) < 0 or not (test_rows or train_rows or dev_rows):
        raise ValueError("nonnegative partition sizes with at least one example required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("mate curriculum output must be empty")
    protected = set()
    for root in protected_roots:
        protected |= read_keys(root, {"train", "dev", "test"})
    rng = random.Random(seed)
    rows = {"test": []} if test_rows else {"train": [], "dev": []}
    limits = {"test": test_rows} if test_rows else {"train": train_rows, "dev": dev_rows}
    seen = set()
    for root in exclude_curricula:
        for path in Path(root).glob("*.jsonl"):
            for line in path.read_text().splitlines():
                seen.add(json.loads(line)["group"])
    initially_excluded_orbits = len(seen)
    attempts = 0
    started = time.monotonic()
    while any(len(rows[split]) < limit for split, limit in limits.items()):
        attempts += 1
        if attempts > 600000:
            raise RuntimeError("curriculum sampling budget exhausted")
        board = chess.Board(None)
        color = rng.choice([chess.WHITE, chess.BLACK])
        board.turn = color
        # All material is pawnless: full D4 orbit grouping is valid here.
        extra = rng.choice([chess.QUEEN, chess.QUEEN, chess.ROOK])
        specs = [(chess.KING, color), (extra, color), (chess.KING, not color)]
        if rng.random() < 0.35:
            specs.append((rng.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK]), color))
        if rng.random() < 0.15:
            specs.append((rng.choice([chess.KNIGHT, chess.BISHOP]), not color))
        for square, (kind, c) in zip(rng.sample(range(64), len(specs)), specs):
            board.set_piece_at(square, chess.Piece(kind, c))
        if not board.is_valid() or board.is_game_over() or fingerprint(board) in protected:
            continue
        mates = mate_moves(board)
        if not mates:
            continue
        group = orbit_group(board)
        if group in seen:
            continue
        split = "test" if test_rows else ("dev" if int(group[:8], 16) % 6 == 0 else "train")
        limit = limits[split]
        if len(rows[split]) >= limit:
            continue
        seen.add(group)
        rows[split].append(
            {
                "fen": board.fen(),
                "targets": mates,
                "key": fingerprint(board).hex(),
                "group": group,
                "pieces": len(specs),
            }
        )
        if sum(map(len, rows.values())) % 500 == 0:
            print(
                json.dumps(
                    {
                        **{s: len(values) for s, values in rows.items()},
                        "attempts": attempts,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    partitions = {}
    for split, values in rows.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in values))
        partitions[split] = {"rows": len(values), "sha256": sha256(path)}
    result = {
        "schema_version": 1,
        "seed": seed,
        "partitions": partitions,
        "attempts": attempts,
        "excluded_orbits": initially_excluded_orbits,
        "seconds": time.monotonic() - started,
        "label": "all legal moves delivering immediate checkmate",
        "split": "whole color-normalized D4 orbits, with no pawns or castling",
        "limitations": "Synthetic sparse-board stress curriculum; success is not full-game tactical strength.",
    }
    atomic_json(output / "manifest.json", result)
    return result


class MateDataset(Dataset):
    def __init__(self, root, split):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        path = root / f"{split}.jsonl"
        if (
            manifest.get("schema_version") != 1
            or sha256(path) != manifest["partitions"][split]["sha256"]
        ):
            raise ValueError("mate dataset integrity failure")
        self.rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(self.rows) != manifest["partitions"][split]["rows"] or not self.rows:
            raise ValueError("mate row count mismatch")
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index not in self.cache:
            row = self.rows[index]
            board = chess.Board(row["fen"])
            if not board.is_valid() or board.pawns or board.castling_rights:
                raise ValueError("mate curriculum requires valid pawnless, castling-free positions")
            if fingerprint(board).hex() != row["key"]:
                raise ValueError("position fingerprint mismatch")
            if (
                not row["targets"]
                or mate_moves(board) != row["targets"]
                or orbit_group(board) != row["group"]
            ):
                raise ValueError("incorrect mate labels or orbit group")
            item = encode(board)
            targets = [move_id(chess.Move.from_uci(text)) for text in row["targets"]]
            item.update(target=targets[0], acceptable=targets, group=row["group"])
            self.cache[index] = item
        return self.cache[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("v4_data/mates"))
    parser.add_argument("--protected-roots", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--test-rows",
        type=int,
        default=0,
        help="Generate only a final test partition, never used by refinement.",
    )
    parser.add_argument("--exclude-curricula", type=Path, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=5221)
    args = parser.parse_args()
    print(json.dumps(generate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
