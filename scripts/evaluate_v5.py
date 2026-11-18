"""Fresh frozen-model game and tactical tests; no checkpoint selection occurs here."""

import argparse
import json
from pathlib import Path

import chess
import torch
from torch.utils.data import DataLoader

from chess_policy.checkpoint import load_checkpoint
from chess_policy.data import atomic_json, sha256
from chess_policy.evaluation import evaluate
from chess_policy.inference import Policy
from chess_policy.mate_curriculum import mate_moves, orbit_group
from chess_policy.model import model_inputs
from chess_policy.refinement import collate_targets, set_nll
from chess_policy.tactics import TacticDataset


@torch.inference_mode()
def tactical_metrics(model, data):
    n = hit = top3 = 0
    nll = uniform = 0.0
    for batch in DataLoader(data, batch_size=64, collate_fn=collate_targets):
        logits = model(**model_inputs(batch))["logits"]
        accepted = batch["accepted"]
        choices = logits.topk(min(3, logits.shape[-1]), dim=-1).indices
        n += len(choices)
        hit += accepted.gather(1, choices[:, :1]).sum().item()
        top3 += accepted.gather(1, choices).any(-1).sum().item()
        nll += set_nll(logits, accepted).sum().item()
        uniform += (accepted.sum(-1) / (batch["legal"] >= 0).sum(-1)).sum().item()
    return {
        "positions": n,
        "acceptable_top1": hit / n,
        "acceptable_top3": top3 / n,
        "set_nll": nll / n,
        "uniform_acceptable_probability": uniform / n,
        "seed_families": len({row["group"] for row in data.rows}),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("v5_holdout"))
    p.add_argument("--tactics", type=Path, default=Path("v5_data/tactic_test"))
    p.add_argument("--out", type=Path, default=Path("v5_results"))
    p.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    args = p.parse_args()
    selection = json.loads((args.out / "selection.json").read_text())
    if selection["selected_sha256"] != sha256(args.checkpoints / "policy_v5.pt"):
        raise ValueError("selected model identity changed")
    datasets = {task: TacticDataset(args.tactics, "test", task) for task in ["mate", "defense"]}
    forbidden = set()
    for root in [Path("v4_data/mates"), Path("v4_data/mate_test"), Path("v5_data/tactics")]:
        for file in root.glob("*.jsonl"):
            forbidden.update(json.loads(line)["group"] for line in file.read_text().splitlines())
    for data in datasets.values():
        if any(
            row["group"] in forbidden or orbit_group(chess.Board(row["seed_fen"])) != row["group"]
            for row in data.rows
        ):
            raise ValueError("tactical test seed-family leakage")
    paths = {
        "v4": args.checkpoints / "policy_v4.pt",
        "control": args.checkpoints / "v5_control_best.pt",
        "targeted": args.checkpoints / "v5_targeted_best.pt",
    }
    diagnostics = json.loads(Path("v4_results/v3_loss_audit.json").read_text())
    diagnostic_rows = [r for r in diagnostics["rows"] if r["missed_mate_in_one"]]
    results = {}
    for name, path in paths.items():
        torch.set_num_threads(2)
        model, _ = load_checkpoint(path)
        tactic = {task: tactical_metrics(model, data) for task, data in datasets.items()}
        game = evaluate(path, args.data, args.out / f"fresh_{name}.json", threads=2)
        policy = Policy(path, threads=2)
        decisions = []
        for row in diagnostic_rows:
            board = chess.Board(row["fen"])
            move = policy.predict(board)["move"]
            decisions.append(
                {"fen": row["fen"], "move": move, "correct": move in mate_moves(board)}
            )
        results[name] = {
            "checkpoint_sha256": sha256(path),
            "tactics": tactic,
            "games": game,
            "old_diagnostic": {
                "correct": sum(r["correct"] for r in decisions),
                "opportunities": len(decisions),
                "rows": decisions,
            },
        }
        atomic_json(args.out / f"evaluation_{name}.json", results[name])
        print(
            json.dumps(
                {
                    "model": name,
                    "tactics": tactic,
                    "fresh_top1": game["top1"],
                    "diagnostic_mates": results[name]["old_diagnostic"]["correct"],
                }
            ),
            flush=True,
        )
    report = {
        "models": results,
        "selection_sha256": sha256(args.out / "selection.json"),
        "tactical_test_seed_overlap": 0,
        "limits": "Tactical boards are constructed, with correlated variants of held-out seed families. Acceptable defensive moves prevent only immediate mate, not all losses. Same-run game imitation and a prior inspected diagnostic do not establish Elo.",
    }
    atomic_json(args.out / "final_evaluation.json", report)


if __name__ == "__main__":
    main()
