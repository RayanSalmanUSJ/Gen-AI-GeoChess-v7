"""Teacher-labeled learner rollouts for direct policy correction.

The teacher runs only during dataset creation. Deployment uses the same
one-forward-pass policy architecture as v3. No teacher scores are input features.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import chess
import chess.engine
import numpy as np
from torch.utils.data import Dataset

from .data import atomic_json, read_manifest, sha256
from .encoding import encode, fingerprint, move_id
from .inference import Policy


def read_keys(root, splits):
    root = Path(root)
    manifest = read_manifest(root, verify=True)
    keys = set()
    for shard in manifest["shards"]:
        if shard["split"] in splits:
            keys.update(
                bytes(x) for x in np.load(root / shard["files"]["keys"]["path"], mmap_mode="r")
            )
    return keys


def create_corrections(
    checkpoint,
    stockfish,
    base_data,
    heldout_data,
    output,
    *,
    seed=4071,
    nodes=6000,
    teacher_probability=0.3,
    train_games=192,
    dev_games=40,
    max_plies=96,
):
    """Two fixed partitions, distinct initial opening groups, exact-position exclusions."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("correction output must be empty")
    if min(nodes, train_games, dev_games, max_plies) < 1 or not 0 <= teacher_probability <= 1:
        raise ValueError("invalid rollout parameters")
    policy = Policy(checkpoint, threads=2)
    base = read_manifest(base_data)
    protected_test = read_keys(heldout_data, {"test"}) | read_keys(base_data, {"test"})
    train_blocked = protected_test | read_keys(base_data, {"dev"})
    dev_blocked = protected_test | read_keys(base_data, {"train"})
    seen = set()
    rng = random.Random(seed)
    engine = chess.engine.SimpleEngine.popen_uci(str(Path(stockfish).resolve()))
    engine.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": False})
    summaries = {}
    started = time.monotonic()
    try:
        for split, game_count, blocked in [
            ("train", train_games, train_blocked),
            ("dev", dev_games, dev_blocked),
        ]:
            openings = sorted({g["opening"] for g in base["games"] if g["split"] == split})
            rng.shuffle(openings)
            if game_count > len(openings):
                raise ValueError("require distinct opening groups for every rollout")
            records = duplicates = disagreements = 0
            events = []
            file = output / f"{split}.jsonl"
            with file.open("w") as stream:
                for episode, fen4 in enumerate(openings[:game_count]):
                    board = chess.Board(fen4 + " 0 1")
                    group = hashlib.blake2b(fen4.encode(), digest_size=16).hexdigest()
                    moves = []
                    game_id = object()
                    for ply in range(max_plies):
                        if board.is_game_over(claim_draw=True):
                            break
                        student = policy.predict(board)["move"]
                        info = engine.analyse(board, chess.engine.Limit(nodes=nodes), game=game_id)
                        teacher = info["pv"][0]
                        if teacher not in board.legal_moves:
                            raise ValueError("illegal teacher label")
                        key = fingerprint(board)
                        execute_teacher = rng.random() < teacher_probability
                        chosen = teacher if execute_teacher else chess.Move.from_uci(student)
                        if key not in blocked and key not in seen:
                            row = {
                                "fen": board.fen(en_passant="fen"),
                                "target": teacher.uci(),
                                "student": student,
                                "played": chosen.uci(),
                                "teacher_executed": execute_teacher,
                                "group": group,
                                "episode": episode,
                                "ply": ply,
                                "key": key.hex(),
                                "teacher_nodes": info.get("nodes"),
                                "teacher_depth": info.get("depth"),
                            }
                            stream.write(json.dumps(row) + "\n")
                            seen.add(key)
                            records += 1
                            disagreements += student != teacher.uci()
                        else:
                            duplicates += 1
                        moves.append(chosen.uci())
                        board.push(chosen)
                    outcome = board.outcome(claim_draw=True)
                    events.append(
                        {
                            "opening": fen4,
                            "group": group,
                            "moves": moves,
                            "result": outcome.result() if outcome else "*",
                            "reason": outcome.termination.name if outcome else "ply_cap",
                        }
                    )
                    if (episode + 1) % 8 == 0:
                        stream.flush()
                        print(
                            json.dumps(
                                {
                                    "split": split,
                                    "episodes": episode + 1,
                                    "records": records,
                                    "seconds": round(time.monotonic() - started, 1),
                                }
                            ),
                            flush=True,
                        )
            summaries[split] = {
                "rows": records,
                "episodes": len(events),
                "excluded_or_duplicate": duplicates,
                "teacher_student_disagreements": disagreements,
                "sha256": sha256(file),
                "groups": [e["group"] for e in events],
            }
            atomic_json(output / f"{split}_episodes.json", {"episodes": events})
        result = {
            "schema_version": 1,
            "checkpoint_sha256": sha256(checkpoint),
            "base_manifest_sha256": sha256(Path(base_data) / "manifest.json"),
            "heldout_manifest_sha256": sha256(Path(heldout_data) / "manifest.json"),
            "teacher": dict(engine.id),
            "teacher_binary_sha256": sha256(stockfish),
            "config": {
                "seed": seed,
                "nodes": nodes,
                "teacher_probability": teacher_probability,
                "train_games": train_games,
                "dev_games": dev_games,
                "max_plies": max_plies,
            },
            "partitions": summaries,
            "seconds": time.monotonic() - started,
            "limitations": "Single aggregation round with a frozen student; finite-node teacher labels. Not a full convergence demonstration of DAgger.",
        }
        atomic_json(output / "manifest.json", result)
        return result
    finally:
        engine.quit()


class CorrectionDataset(Dataset):
    def __init__(self, root, split):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        path = root / f"{split}.jsonl"
        if (
            manifest["schema_version"] != 1
            or sha256(path) != manifest["partitions"][split]["sha256"]
        ):
            raise ValueError("correction dataset integrity failure")
        self.rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(self.rows) != manifest["partitions"][split]["rows"] or not self.rows:
            raise ValueError("correction row count mismatch")
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index not in self.cache:
            row = self.rows[index]
            board = chess.Board(row["fen"])
            if fingerprint(board).hex() != row["key"]:
                raise ValueError("position fingerprint mismatch")
            target = chess.Move.from_uci(row["target"])
            if target not in board.legal_moves:
                raise ValueError("illegal correction target")
            item = encode(board)
            item.update(target=move_id(target), group=row["group"])
            self.cache[index] = item
        return self.cache[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/pilot.pt"))
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--heldout-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("v4_data/corrections"))
    parser.add_argument("--train-games", type=int, default=192)
    parser.add_argument("--dev-games", type=int, default=40)
    parser.add_argument("--nodes", type=int, default=6000)
    args = parser.parse_args()
    print(json.dumps(create_corrections(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
