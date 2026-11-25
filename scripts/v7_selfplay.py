"""Oracle-Labelled Self-Play with Batched PUCT MCTS for GeoChess (Chunk 2, Step 3).

1. Generates self-play games using V6Policy with PUCT MCTS (stochastic opening exploration,
   deterministic endgame play, resignation handling).
2. Saves raw positions and AlphaZero-style visit distributions (visits, policy targets).
3. Labels positions with Stockfish 15.1 depth 10 using v6_data.label (MultiPV all legal moves).
4. Assembles dataset tensors into data/v7/selfplay_d10 using v6_data.assemble.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple

import chess
import numpy as np
import torch

from chess_policy.encoding import fingerprint, move_id
from chess_policy.mcts import MCTS, MCTSConfig
from chess_policy.v6_data import assemble, label
from chess_policy.v6_model import load_v6


def play_game(
    mcts: MCTS,
    game_idx: int,
    max_plies: int = 150,
    temp_plies: int = 15,
    resign_threshold: float = -0.85,
    resign_streak_required: int = 3,
) -> Tuple[List[Dict], str, str]:
    """Play a self-play game using MCTS.

    Returns:
        positions: list of recorded position dicts
        result_str: "1-0", "0-1", or "1/2-1/2"
        termination_reason: reason the game ended
    """
    board = chess.Board()
    positions = []
    consecutive_low_val = {chess.WHITE: 0, chess.BLACK: 0}
    resigned_color = None
    termination = "unknown"

    t0 = time.monotonic()

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        # Temperature & Dirichlet noise schedule:
        # Early plies: explore with T=1.0 and Dirichlet noise at root
        # Later plies: competitive play with T=0.0 (argmax) and no noise
        if ply < temp_plies:
            mcts.config.temperature = 1.0
            mcts.config.dirichlet_epsilon = 0.25
        else:
            mcts.config.temperature = 0.0
            mcts.config.dirichlet_epsilon = 0.0

        fen_before = board.fen()
        turn_before = board.turn
        key = fingerprint(board)

        res = mcts.search(board)
        move = res["best_move"]
        if move is None:
            break

        root_val = float(res.get("root_value", 0.0))

        # Check resignation condition (perspective of player to move)
        if root_val < resign_threshold:
            consecutive_low_val[turn_before] += 1
            if consecutive_low_val[turn_before] >= resign_streak_required:
                resigned_color = turn_before
                termination = f"resignation by {'White' if turn_before else 'Black'}"
                break
        else:
            consecutive_low_val[turn_before] = 0

        # Record position data
        positions.append({
            "fen": fen_before,
            "move": move.uci(),
            "target": move_id(move),
            "game": game_idx,
            "ply": ply,
            "turn": turn_before,
            "key": key,
            "policy_target": res.get("policy_target", {}),
            "visits": res.get("visits", {}),
            "root_val": root_val,
        })

        board.push(move)

    elapsed = time.monotonic() - t0

    # Determine game outcome
    if resigned_color is not None:
        result_str = "0-1" if resigned_color == chess.WHITE else "1-0"
    elif board.is_checkmate():
        # The player whose turn it is now has been checkmated
        result_str = "0-1" if board.turn == chess.WHITE else "1-0"
        termination = "checkmate"
    elif board.is_stalemate():
        result_str = "1/2-1/2"
        termination = "stalemate"
    elif board.is_insufficient_material():
        result_str = "1/2-1/2"
        termination = "insufficient_material"
    elif board.can_claim_draw():
        result_str = "1/2-1/2"
        termination = "draw_claim"
    elif len(positions) >= max_plies:
        result_str = "1/2-1/2"
        termination = f"max_plies_{max_plies}"
    else:
        result_str = "1/2-1/2"
        termination = "draw"

    # Map game outcome to numeric result for White
    white_score = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[result_str]

    # Assign relative outcome (+1 if side to move won, -1 if side to move lost, 0 for draw)
    for p in positions:
        p["result"] = white_score if p["turn"] == chess.WHITE else -white_score

    moves_per_sec = len(positions) / max(elapsed, 1e-6)
    print(f"Game {game_idx + 1:2d} finished: {result_str:7s} ({len(positions):3d} plies, "
          f"{elapsed:4.1f}s, {moves_per_sec:4.1f} moves/s) — {termination}")

    return positions, result_str, termination


def generate_selfplay_positions(
    model: torch.nn.Module,
    device: torch.device,
    n_games: int,
    simulations: int = 32,
    c_puct: float = 1.5,
    batch_size: int = 4,
    temp_plies: int = 15,
    max_plies: int = 150,
) -> Tuple[List[Dict], Dict]:
    """Generate self-play games and return list of all position records."""
    cfg = MCTSConfig(
        simulations=simulations,
        c_puct=c_puct,
        batch_size=batch_size,
        virtual_loss_weight=0.1,
        fpu_reduction=0.0,
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )
    mcts = MCTS(model, device, cfg)

    all_positions = []
    results_count = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    print(f"\nStarting Self-Play: {n_games} games with MCTS (N={simulations}, c_puct={c_puct})")
    print(f"Device: {device}, Leaf batch size: {batch_size}, Temp plies: {temp_plies}")
    t_start = time.monotonic()

    for g in range(n_games):
        game_pos, res_str, _ = play_game(
            mcts=mcts,
            game_idx=g,
            max_plies=max_plies,
            temp_plies=temp_plies,
        )
        all_positions.extend(game_pos)
        results_count[res_str] += 1

        if (g + 1) % 10 == 0 or g + 1 == n_games:
            elapsed_so_far = time.monotonic() - t_start
            m_s = len(all_positions) / max(elapsed_so_far, 1e-6)
            eta = (elapsed_so_far / (g + 1)) * (n_games - (g + 1))
            print(f"--> [{g + 1:2d}/{n_games} games complete | {len(all_positions)} positions | "
                  f"{elapsed_so_far:.1f}s elapsed | {m_s:.1f} moves/s | ETA: {eta / 60:.1f} min]")

    total_time = time.monotonic() - t_start
    total_plies = len(all_positions)
    avg_speed = total_plies / max(total_time, 1e-6)

    stats = {
        "games": n_games,
        "total_positions": total_plies,
        "total_seconds": round(total_time, 2),
        "moves_per_sec": round(avg_speed, 2),
        "sec_per_game": round(total_time / n_games, 2),
        "results": results_count,
    }

    print("\n--- Self-Play Summary ---")
    print(f"Total Games: {n_games} | Total Plies: {total_plies}")
    print(f"Results: White wins: {results_count['1-0']}, Black wins: {results_count['0-1']}, Draws: {results_count['1/2-1/2']}")
    print(f"Total Duration: {total_time:.1f}s ({stats['sec_per_game']:.1f}s/game, {avg_speed:.1f} moves/s)")

    return all_positions, stats


def save_positions_npz(positions: List[Dict], out_file: Path) -> Path:
    """Format and save positions to an .npz file matching v6_data specifications."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n = len(positions)
    fens = np.array([p["fen"].encode("ascii") for p in positions], dtype="S100")
    moves = np.array([p["move"].encode("ascii") for p in positions], dtype="S5")
    targets = np.array([p["target"] for p in positions], dtype=np.int16)
    splits = np.zeros(n, dtype=np.int8)  # SPLIT_CODES["train"] = 0
    games = np.array([p["game"] for p in positions], dtype=np.int32)
    plies = np.array([p["ply"] for p in positions], dtype=np.int16)
    results = np.array([p["result"] for p in positions], dtype=np.int8)
    keys = np.array([p["key"] for p in positions], dtype="V16")

    # Group hash by game
    groups = []
    for p in positions:
        g_str = f"selfplay_game_{p['game']}"
        ghash = hashlib.blake2b(g_str.encode(), digest_size=16).hexdigest()
        groups.append(ghash.encode("ascii"))
    groups_arr = np.array(groups, dtype="S32")

    out_dict = dict(
        fen=fens,
        move=moves,
        target=targets,
        split=splits,
        game=games,
        ply=plies,
        result=results,
        key=keys,
        group=groups_arr,
    )

    np.savez(out_file, **out_dict)

    # Write summary metadata
    meta = {
        "positions": n,
        "games": int(len(np.unique(games))),
        "file": str(out_file),
    }
    Path(str(out_file) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved {n} self-play positions to {out_file}")

    # Also save visit distributions for Step 4 (AlphaZero targets)
    visits_file = out_file.parent / "selfplay_visits.jsonl"
    with visits_file.open("w", encoding="utf-8") as f:
        for p in positions:
            record = {
                "fen": p["fen"],
                "move": p["move"],
                "target": int(p["target"]),
                "game": int(p["game"]),
                "ply": int(p["ply"]),
                "result": int(p["result"]),
                "policy_target": p["policy_target"],
                "root_val": round(float(p["root_val"]), 4),
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved MCTS visit targets to {visits_file}")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Self-play with PUCT MCTS and Oracle Stockfish labelling.")
    parser.add_argument("--model", default="runs/v6/scale_L_1M/best.pt", help="Path to checkpoint.")
    parser.add_argument("--games", type=int, default=5, help="Number of self-play games to generate.")
    parser.add_argument("--simulations", type=int, default=32, help="MCTS simulation budget per move.")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration factor.")
    parser.add_argument("--batch-size", type=int, default=4, help="MCTS leaf evaluation batch size.")
    parser.add_argument("--temp-plies", type=int, default=15, help="Plies with T=1.0 exploration.")
    parser.add_argument("--max-plies", type=int, default=150, help="Max plies per game.")
    parser.add_argument("--engine", default="tools/stockfish/stockfish.exe", help="Stockfish engine binary.")
    parser.add_argument("--depth", type=int, default=10, help="Stockfish labelling depth.")
    parser.add_argument("--workers", type=int, default=0, help="Stockfish labelling workers.")
    parser.add_argument("--device", default="cpu", help="PyTorch device.")
    parser.add_argument("--out-dir", default="data/v7/selfplay_d10", help="Output directory for assembled tensors.")
    parser.add_argument("--skip-label", action="store_true", help="Skip Stockfish labelling and tensor assembly.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load model
    print(f"Loading checkpoint: {args.model}")
    model, config = load_v6(args.model)
    model.to(device)
    model.eval()

    # 2. Run self-play games
    positions, stats = generate_selfplay_positions(
        model=model,
        device=device,
        n_games=args.games,
        simulations=args.simulations,
        c_puct=args.c_puct,
        batch_size=args.batch_size,
        temp_plies=args.temp_plies,
        max_plies=args.max_plies,
    )

    # 3. Save raw positions npz
    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    positions_file = out_dir.parent / "positions_selfplay.npz"
    save_positions_npz(positions, positions_file)

    if args.skip_label:
        print("\n--skip-label specified: Skipping Stockfish labelling and tensor assembly.")
        return

    # 4. Stockfish Depth-10 Labelling
    label_dir = out_dir.parent / f"labels_selfplay_d{args.depth}"
    label_meta_file = label_dir / "label_meta.json"
    if label_meta_file.exists():
        try:
            import shutil
            old_meta = json.loads(label_meta_file.read_text())
            if old_meta.get("n_rows") != len(positions):
                print(f"Clearing old label dir (previous rows: {old_meta.get('n_rows')}, new rows: {len(positions)})")
                shutil.rmtree(label_dir)
        except Exception as e:
            print(f"Warning checking label dir: {e}")

    # Also check if target assemble dir needs clearing
    if out_dir.exists():
        meta_file = out_dir / "meta.json"
        if meta_file.exists():
            try:
                import shutil
                old_meta = json.loads(meta_file.read_text())
                if old_meta.get("rows") != len(positions):
                    print(f"Clearing old assemble dir (previous rows: {old_meta.get('rows')}, new rows: {len(positions)})")
                    shutil.rmtree(out_dir)
            except Exception as e:
                print(f"Warning checking assemble dir: {e}")

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"\n--- Labelling {len(positions)} positions with Stockfish 15.1 at depth {args.depth} ---")
    print(f"Engine: {args.engine}, Workers: {workers}, Output: {label_dir}")

    t_label_start = time.monotonic()
    label(
        positions_file=str(positions_file),
        out_dir=str(label_dir),
        engine_path=args.engine,
        depth=args.depth,
        workers=workers,
        splits=("train",),
        chunk=500,
    )
    t_label = time.monotonic() - t_label_start
    print(f"Labelling finished in {t_label:.1f}s ({len(positions) / max(t_label, 1e-6):.1f} pos/s)")

    # 5. Assemble Dataset Tensors
    print(f"\n--- Assembling dataset tensors into {out_dir} ---")
    meta = assemble(
        positions_file=str(positions_file),
        out_dir=str(out_dir),
        split="train",
        label_dir=str(label_dir),
        workers=workers,
    )
    print(f"Successfully assembled {meta['rows']} rows into {out_dir}!")


if __name__ == "__main__":
    main()
