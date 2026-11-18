"""Unified CLI. No expensive work occurs at import time."""

import argparse
import json
from pathlib import Path
import chess


def main():
    root = argparse.ArgumentParser(description="Geometry-aware legal next-move transformer")
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-data")
    build.add_argument("--pgn", type=Path, action="append", required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--max-games", type=int, default=0)
    build.add_argument("--force-test", action="store_true")
    build.add_argument("--exclude", type=Path)
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    inspect = commands.add_parser("inspect-config")
    inspect.add_argument("--config", type=Path, required=True)
    for name in ("predict", "uci", "serve", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", type=Path, default=Path("checkpoints/policy_v5.pt"))
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=4)
        if name == "predict":
            command.add_argument("--fen", default=chess.STARTING_FEN)
            command.add_argument("--top-k", type=int, default=5)
            command.add_argument("--temperature", type=float, default=0)
            command.add_argument("--seed", type=int)
            command.add_argument("--show-tokens", action="store_true")
        elif name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8000)
        elif name == "evaluate":
            command.add_argument("--data", type=Path, required=True)
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--split", default="test", choices=["train", "dev", "test"])
    a = root.parse_args()
    if a.command == "build-data":
        from .data import build_corpus

        r = build_corpus(
            a.pgn, a.out, max_games=a.max_games, force_test=a.force_test, exclude=a.exclude
        )
        result = {k: r[k] for k in ("counts", "seconds", "read_games", "duplicates_removed")}
    elif a.command == "train":
        from .model import ModelConfig
        from .training import TrainConfig, train

        spec = json.loads(a.config.read_text())
        result = train(
            a.data, a.out, ModelConfig(**spec["model"]), TrainConfig(**spec["train"]), a.resume
        )
    elif a.command == "evaluate":
        from .evaluation import evaluate

        result = evaluate(
            a.checkpoint, a.data, a.out, split=a.split, device=a.device, threads=a.threads
        )
    elif a.command == "inspect-config":
        import torch
        from .model import ModelConfig, ChessPolicy

        spec = ModelConfig(**json.loads(a.config.read_text())["model"])
        with torch.device("meta"):
            model = ChessPolicy(spec)
        n = sum(p.numel() for p in model.parameters())
        result = {
            "parameters": n,
            "fp32_weights_MiB": n * 4 / 2**20,
            "note": "Weights only; excludes gradients, optimizer, activations. Not measured VRAM.",
        }
    elif a.command == "serve":
        import uvicorn
        from .api import create_app

        uvicorn.run(create_app(a.checkpoint, a.device, a.threads), host=a.host, port=a.port)
        return
    else:
        from .inference import Policy

        policy = Policy(a.checkpoint, a.device, a.threads)
        if a.command == "uci":
            from .uci import serve_uci

            serve_uci(policy)
            return
        result = policy.predict(a.fen, top_k=a.top_k, temperature=a.temperature, seed=a.seed)
        if a.show_tokens:
            from .encoding import text_tokens

            result["input_tokens"] = text_tokens(chess.Board(a.fen))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
