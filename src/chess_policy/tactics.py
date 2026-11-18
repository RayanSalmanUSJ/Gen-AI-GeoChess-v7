"""Offline pawnful tactical supervision. Deployment never imports this module."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import chess
from torch.utils.data import Dataset

from .corrections import read_keys
from .data import atomic_json, sha256
from .encoding import encode, fingerprint, move_id
from .mate_curriculum import mate_moves


def safe_moves(board):
    """All moves that do not permit immediate opposing checkmate; not optimal moves."""
    good, bad = [], []
    for move in list(board.legal_moves):
        board.push(move)
        try:
            unsafe = not board.is_game_over(claim_draw=False) and bool(mate_moves(board))
        finally:
            board.pop()
        (bad if unsafe else good).append(move.uci())
    return sorted(good), sorted(bad)


def augment(seed_board, rng):
    """Add pawnful context while retaining a valid position. No invented D4 symmetry."""
    board = seed_board.copy(stack=False)
    target = rng.randint(8, 22)
    attempts = 0
    while len(board.piece_map()) < target and attempts < 150:
        attempts += 1
        square = rng.randrange(64)
        if board.piece_at(square):
            continue
        kind = rng.choices(
            [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN], [65, 12, 12, 8, 3]
        )[0]
        if kind == chess.PAWN and chess.square_rank(square) in (0, 7):
            continue
        color = rng.choice([chess.WHITE, chess.BLACK])
        board.set_piece_at(square, chess.Piece(kind, color))
        if not board.is_valid():
            board.remove_piece_at(square)
    if len(board.piece_map()) != target or len(chess.SquareSet(board.pawns)) < 2:
        return None
    return board


def generate(
    seeds,
    output,
    protected_roots,
    *,
    partitions=("train", "dev"),
    mate_rows=(6000, 1000),
    defense_rows=(3000, 600),
    seed=16031,
    excluded_json_roots=(),
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("use a fresh tactical dataset directory")
    if (
        not (len(partitions) == len(mate_rows) == len(defense_rows))
        or min(*mate_rows, *defense_rows) < 1
    ):
        raise ValueError("positive task sizes required for every partition")
    protected = set()
    for root in protected_roots:
        protected.update(read_keys(root, {"train", "dev", "test"}))
    for root in excluded_json_roots:
        for file in Path(root).glob("*.jsonl"):
            protected.update(
                bytes.fromhex(json.loads(line)["key"]) for line in file.read_text().splitlines()
            )
    sources = {
        split: [
            json.loads(line) for line in (Path(seeds) / (split + ".jsonl")).read_text().splitlines()
        ]
        for split in partitions
    }
    owners = {}
    for split, rows in sources.items():
        for row in rows:
            if row["group"] in owners and owners[row["group"]] != split:
                raise ValueError("seed archetype crosses partitions")
            owners[row["group"]] = split
    rng = random.Random(seed)
    seen = set(protected)
    summaries = {}
    started = time.monotonic()
    for split, mates_needed, defense_needed in zip(partitions, mate_rows, defense_rows):
        rows = []
        counts = Counter()
        attempts = 0
        while counts["mate"] < mates_needed or counts["defense"] < defense_needed:
            attempts += 1
            if attempts > 100000:
                raise RuntimeError("tactical sampling budget exhausted")
            source = rng.choice(sources[split])
            board = augment(chess.Board(source["fen"]), rng)
            if board is None:
                continue
            targets = mate_moves(board)
            if not targets:
                continue
            for task in ["mate", "defense"]:
                if task == "defense":
                    board = board.copy(stack=False)
                    board.turn = not board.turn
                needed = mates_needed if task == "mate" else defense_needed
                if counts[task] >= needed or not board.is_valid() or board.is_game_over():
                    continue
                key = fingerprint(board)
                if key in seen:
                    continue
                bad = []
                if task == "defense":
                    targets, bad = safe_moves(board)
                    if not targets or not bad:
                        continue
                rows.append(
                    {
                        "fen": board.fen(),
                        "key": key.hex(),
                        "targets": targets,
                        "unsafe": bad,
                        "task": task,
                        "group": source["group"],
                        "seed_fen": source["fen"],
                        "pieces": len(board.piece_map()),
                        "pawns": len(chess.SquareSet(board.pawns)),
                    }
                )
                seen.add(key)
                counts[task] += 1
                if len(rows) % 1000 == 0:
                    print(
                        json.dumps(
                            {
                                "split": split,
                                **counts,
                                "attempts": attempts,
                                "seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
        path = output / (split + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        summaries[split] = {
            "rows": len(rows),
            "tasks": dict(counts),
            "sha256": sha256(path),
            "seed_sha256": sha256(Path(seeds) / (split + ".jsonl")),
            "groups": sorted({r["group"] for r in rows}),
            "attempts": attempts,
            "pieces_histogram": dict(sorted(Counter(r["pieces"] for r in rows).items())),
            "pawns_histogram": dict(sorted(Counter(r["pawns"] for r in rows).items())),
        }
    result = {
        "schema_version": 1,
        "seed": seed,
        "partitions": summaries,
        "seconds": time.monotonic() - started,
        "protected_positions": len(protected),
        "split": "Whole sparse-mate archetypes retained in their source partition; no full-board D4 augmentation.",
        "labels": "All immediate mating moves, or all moves preventing immediate opposing checkmate.",
        "limits": "Constructed legal pawnful positions, not natural-game distribution or proof of historical reachability; defensive labels establish only one-reply safety.",
    }
    atomic_json(output / "manifest.json", result)
    return result


class TacticDataset(Dataset):
    def __init__(self, root, split, task):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        file = root / (split + ".jsonl")
        if (
            manifest.get("schema_version") != 1
            or sha256(file) != manifest["partitions"][split]["sha256"]
        ):
            raise ValueError("tactic dataset integrity failure")
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        if len(rows) != manifest["partitions"][split]["rows"] or task not in ("mate", "defense"):
            raise ValueError("invalid tactical partition or task")
        self.rows = [r for r in rows if r["task"] == task]
        if len(self.rows) != manifest["partitions"][split]["tasks"][task] or not self.rows:
            raise ValueError("tactic task count mismatch")
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index not in self.cache:
            row = self.rows[index]
            board = chess.Board(row["fen"])
            if not board.is_valid() or fingerprint(board).hex() != row["key"]:
                raise ValueError("invalid tactical position or fingerprint")
            if row["task"] == "mate":
                targets, bad = mate_moves(board), []
            else:
                targets, bad = safe_moves(board)
            if not targets or targets != row["targets"] or bad != row["unsafe"]:
                raise ValueError("incorrect tactical labels")
            item = encode(board)
            accepted = [move_id(chess.Move.from_uci(x)) for x in targets]
            item.update(target=accepted[0], acceptable=accepted, group=row["group"])
            self.cache[index] = item
        return self.cache[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, default=Path("v4_data/mates"))
    parser.add_argument("--output", type=Path, default=Path("v5_data/tactics"))
    parser.add_argument("--protected-roots", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--excluded-json-roots", type=Path, nargs="*", default=[Path("v4_data/corrections")]
    )
    parser.add_argument("--test-only", action="store_true")
    args = vars(parser.parse_args())
    test = args.pop("test_only")
    if test:
        args.update(partitions=("test",), mate_rows=(1000,), defense_rows=(600,), seed=26031)
    print(json.dumps(generate(**args), indent=2))


if __name__ == "__main__":
    main()
