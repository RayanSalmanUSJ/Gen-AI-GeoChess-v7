"""v6 data: positions -> Stockfish scores for every legal move -> pre-encoded tensors.

Three steps, each resumable and each with its own CLI (python -m chess_policy.v6_data ...):

  positions  walk PGNs with v5's exact split and de-duplication rules
             (opening-group split 85/7/8, first occurrence wins) and save, per position:
             FEN, the move played, the split, the game, the ply and the game result
  label      Stockfish scores every legal move (MultiPV = number of legal moves) at a
             fixed depth. Work is cut into chunks; --shard i --of n lets several machines
             share one job; finished chunks are skipped on restart
  assemble   encode positions with v5's encoder and write one directory per split:
             pieces, geometry, attacks (packed), ragged legal-move ids, the played move,
             the game result and, when labels exist, the score of every legal move

Score convention (same as Prototype 2): centipawns from the side to move after
playing the move; mate in n = 10000 - n (getting mated: -(10000 - n)).
"""

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import chess
import chess.engine
import chess.pgn
import numpy as np

from .encoding import encode, fingerprint, move_id, position_key

MATE = 10000
SPLIT_CODES = {"train": 0, "dev": 1, "test": 2}


# ---------------------------------------------------------------------------
# 1. positions
# ---------------------------------------------------------------------------

def _split_for(opening, seed, force_test):
    bucket = int.from_bytes(hashlib.sha256((seed + opening).encode()).digest()[:4], "big") % 100
    if force_test:
        return "test"
    return "train" if bucket < 85 else "dev" if bucket < 92 else "test"


def _result_for_white(game):
    return {"1-0": 1, "0-1": -1, "1/2-1/2": 0}.get(game.headers.get("Result", "*"), 0)


def build_positions(pgn_paths, out_file, seed="fishtest-v2", force_test=False, exclude=None,
                    max_games=0, skip_games=0):
    """Mirror of v5 data.build_corpus (same splits, same de-duplication), keeping FENs.

    exclude: earlier positions file(s); their positions and opening groups are skipped
    (how v5 built its reserved holdout). skip_games: ignore the first N games of the
    PGN stream (e.g. the games already used for training and holdouts).
    """
    seen, excluded_groups = set(), set()
    for path in ([exclude] if isinstance(exclude, str) else exclude or []):
        old = np.load(path)
        seen.update(bytes(k) for k in old["key"])
        excluded_groups.update(g.decode() for g in old["group"])
    rows = {k: [] for k in ("fen", "move", "target", "split", "game", "ply", "result", "key",
                            "group")}
    read = duplicates = skipped = 0
    started = time.monotonic()
    for source in map(Path, pgn_paths):
        opener = gzip.open if str(source).endswith(".gz") else open
        with opener(source, "rt", encoding="utf-8", errors="strict") as stream:
            while not max_games or read < max_games:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                read += 1
                if read <= skip_games:
                    continue
                board = game.board()
                if game.errors or board.chess960 or not board.is_valid():
                    raise ValueError(f"invalid PGN game {read}")
                opening = position_key(board)
                group = hashlib.blake2b(opening.encode(), digest_size=16).hexdigest()
                if group in excluded_groups:
                    skipped += 1
                    continue
                split = _split_for(opening, seed, force_test)
                white_result = _result_for_white(game)
                for ply, move in enumerate(game.mainline_moves()):
                    key = fingerprint(board)
                    if key in seen:
                        duplicates += 1
                    else:
                        seen.add(key)
                        rows["fen"].append(board.fen())
                        rows["move"].append(move.uci())
                        rows["target"].append(move_id(move))
                        rows["split"].append(SPLIT_CODES[split])
                        rows["game"].append(read)
                        rows["ply"].append(ply)
                        rows["result"].append(white_result if board.turn else -white_result)
                        rows["key"].append(key)
                        rows["group"].append(group)
                    board.push(move)
                if read % 500 == 0:
                    print(json.dumps({"games": read, "positions": len(rows["fen"]),
                                      "seconds": round(time.monotonic() - started, 1)}),
                          flush=True)
    out = dict(
        fen=np.array(rows["fen"], dtype="S100"),
        move=np.array(rows["move"], dtype="S5"),
        target=np.array(rows["target"], dtype=np.int16),
        split=np.array(rows["split"], dtype=np.int8),
        game=np.array(rows["game"], dtype=np.int32),
        ply=np.array(rows["ply"], dtype=np.int16),
        result=np.array(rows["result"], dtype=np.int8),
        key=np.array(rows["key"], dtype="V16"),
        group=np.array(rows["group"], dtype="S32"),
    )
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_file, **out)
    counts = {s: int((out["split"] == c).sum()) for s, c in SPLIT_CODES.items()}
    summary = dict(games=read, positions=len(out["fen"]), counts=counts,
                   duplicates_removed=duplicates, excluded_opening_games=skipped)
    Path(str(out_file) + ".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ---------------------------------------------------------------------------
# 2. label
# ---------------------------------------------------------------------------

_ENGINE = None
_LIMIT = None


def _start_engine(path, depth, hash_mb):
    global _ENGINE, _LIMIT
    _ENGINE = chess.engine.SimpleEngine.popen_uci(path)
    _ENGINE.configure({"Threads": 1, "Hash": hash_mb})
    _LIMIT = chess.engine.Limit(depth=depth)


def score_all_moves(engine, board, limit):
    """Score of every legal move, in v5's legal-move order (sorted by move id)."""
    legal = sorted(board.legal_moves, key=move_id)
    infos = engine.analyse(board, limit, multipv=len(legal))
    scores = {}
    for info in infos:
        if info.get("pv"):
            scores[info["pv"][0]] = info["score"].pov(board.turn).score(mate_score=MATE)
    missing = 0
    for move in legal:  # MultiPV very rarely returns fewer lines; score the rest singly
        if move not in scores:
            missing += 1
            info = engine.analyse(board, limit, root_moves=[move])
            scores[move] = info["score"].pov(board.turn).score(mate_score=MATE)
    return [int(scores[m]) for m in legal], missing


def _label_chunk(job):
    chunk_id, fens, out_path = job
    t0 = time.monotonic()
    flat, lengths, missing = [], [], 0
    for fen in fens:
        board = chess.Board(fen.decode() if isinstance(fen, bytes) else fen)
        scores, miss = score_all_moves(_ENGINE, board, _LIMIT)
        flat.extend(scores)
        lengths.append(len(scores))
        missing += miss
    tmp = Path(str(out_path) + ".tmp.npz")
    with open(tmp, "wb") as f:  # fsync before the rename: a reboot once left a corrupt chunk
        np.savez(f, cp=np.clip(np.array(flat, dtype=np.int32), -MATE, MATE).astype(np.int16),
                 lengths=np.array(lengths, dtype=np.int16), missing=np.int32(missing))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)
    return chunk_id, len(fens), time.monotonic() - t0


def _chunk_ok(path):
    try:
        with np.load(path) as z:
            return len(z["cp"]) == int(z["lengths"].astype(np.int64).sum())
    except Exception:
        return False


def label(positions_file, out_dir, engine_path, depth=10, workers=0, chunk=1000, shard=0,
          of=1, splits=("train", "dev", "test"), limit=0, hash_mb=16, stride=1):
    # Windows: CreateProcess cannot start a relative path written with "/" (e.g. the
    # tools/stockfish/stockfish.exe in CLAUDE.md), and a pool initializer that fails just
    # respawns forever. So use an absolute path and start one engine here to fail fast.
    engine_path = os.path.abspath(engine_path)
    chess.engine.SimpleEngine.popen_uci(engine_path).quit()
    pos = np.load(positions_file)
    keep = np.isin(pos["split"], [SPLIT_CODES[s] for s in splits])
    index = np.flatnonzero(keep)[::stride]  # stride > 1: an even sample across all games
    if limit:
        index = index[:limit]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(positions_file=str(positions_file), depth=depth, chunk=chunk, splits=list(splits),
                engine=str(engine_path), stride=stride,
                n_rows=int(len(index)), limit=limit)
    meta_path = out_dir / "label_meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for k in ("depth", "chunk", "splits", "n_rows", "limit", "stride"):
            if old.get(k, 1 if k == "stride" else None) != meta[k]:
                raise ValueError(f"label dir was started with {k}={old[k]}, not {meta[k]}")
    else:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    np.save(out_dir / "rows.npy", index)
    n_chunks = (len(index) + chunk - 1) // chunk
    jobs = []
    for c in range(n_chunks):
        if c % of != shard:
            continue
        path = out_dir / f"chunk_{c:05d}.npz"
        if path.exists():
            if _chunk_ok(path):
                continue
            print(json.dumps({"relabel_corrupt_chunk": c}), flush=True)
            path.unlink()
        jobs.append((c, pos["fen"][index[c * chunk:(c + 1) * chunk]], path))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(json.dumps({"chunks_total": n_chunks, "chunks_todo": len(jobs), "workers": workers,
                      "depth": depth, "positions": int(len(index))}), flush=True)
    if not jobs:
        return
    started, done = time.monotonic(), 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_start_engine,
                  initargs=(engine_path, depth, hash_mb)) as pool:
        for chunk_id, n, sec in pool.imap_unordered(_label_chunk, jobs):
            done += n
            rate = done / (time.monotonic() - started)
            left = sum(len(j[1]) for j in jobs) - done
            print(json.dumps({"chunk": chunk_id, "positions_done": done,
                              "per_second": round(rate, 1),
                              "eta_minutes": round(left / max(rate, 1e-9) / 60, 1)}), flush=True)


def load_labels(label_dir):
    """Concatenate finished chunks -> (rows, cp_flat, lengths, missing); fails if incomplete."""
    label_dir = Path(label_dir)
    meta = json.loads((label_dir / "label_meta.json").read_text())
    rows = np.load(label_dir / "rows.npy")
    n_chunks = (len(rows) + meta["chunk"] - 1) // meta["chunk"]
    cps, lens, missing = [], [], 0
    for c in range(n_chunks):
        path = label_dir / f"chunk_{c:05d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"labelling unfinished: {path} missing")
        z = np.load(path)
        cps.append(z["cp"])
        lens.append(z["lengths"])
        missing += int(z["missing"])
    return rows, np.concatenate(cps), np.concatenate(lens), missing, meta


# ---------------------------------------------------------------------------
# 3. assemble
# ---------------------------------------------------------------------------

def _encode_rows(fens):
    out = []
    for fen in fens:
        e = encode(chess.Board(fen.decode()), validate=False)
        out.append((e["pieces"], e["geometry"], e["attacks"], e["legal"]))
    return out


def assemble(positions_file, out_dir, split, label_dir=None, workers=0, max_rows=0):
    """Write pre-encoded arrays for one split; scores attached when label_dir is given."""
    pos = np.load(positions_file)
    rows = np.flatnonzero(pos["split"] == SPLIT_CODES[split])
    cp_by_row = None
    if label_dir is not None:
        lrows, cp_flat, lengths, missing, meta = load_labels(label_dir)
        offsets = np.concatenate([[0], np.cumsum(lengths.astype(np.int64))])
        where = {int(r): i for i, r in enumerate(lrows)}
        rows = np.array([r for r in rows if int(r) in where], dtype=np.int64)
        cp_by_row = (where, cp_flat, offsets, meta)
    if max_rows:
        rows = rows[:max_rows]
    fens = pos["fen"][rows]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    pieces = np.zeros((len(rows), 64), np.uint8)
    geometry = np.zeros((len(rows), 64, 8), np.uint8)
    attacks = np.zeros((len(rows), 64, 8), np.uint8)
    legal_parts, lengths = [], np.zeros(len(rows), np.int16)
    step = 2000
    batches = [fens[s:s + step] for s in range(0, len(fens), step)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        i = 0
        for part in pool.imap(_encode_rows, batches):
            for p, g, a, l in part:
                pieces[i], geometry[i], attacks[i] = p, g, a
                legal_parts.append(l)
                lengths[i] = len(l)
                i += 1
    legal_flat = np.concatenate(legal_parts).astype(np.int16)
    offsets = np.concatenate([[0], np.cumsum(lengths.astype(np.int64))])
    target_index = np.zeros(len(rows), np.int16)
    for i, r in enumerate(rows):
        hit = np.flatnonzero(legal_flat[offsets[i]:offsets[i + 1]] == pos["target"][r])
        if len(hit) != 1:
            raise ValueError(f"played move not found among legal moves (row {r})")
        target_index[i] = hit[0]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = dict(pieces=pieces, geometry=geometry, attacks=attacks, legal=legal_flat,
                  offsets=offsets, target=target_index, result=pos["result"][rows],
                  game=pos["game"][rows], ply=pos["ply"][rows], source_row=rows)
    meta = dict(split=split, rows=int(len(rows)), positions_file=str(positions_file),
                labelled=label_dir is not None)
    if cp_by_row is not None:
        where, cp_flat, loff, lmeta = cp_by_row
        cp = np.zeros(len(legal_flat), np.int16)
        for i, r in enumerate(rows):
            j = where[int(r)]
            seg = cp_flat[loff[j]:loff[j + 1]]
            if len(seg) != offsets[i + 1] - offsets[i]:
                raise ValueError(f"label/legal length mismatch at row {r}")
            cp[offsets[i]:offsets[i + 1]] = seg
        arrays["cp"] = cp
        meta.update(depth=lmeta["depth"], label_dir=str(label_dir))
    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v, allow_pickle=False)
    (out / "fen.txt").write_text("\n".join(f.decode() for f in fens) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def concat(out_dir, *dirs):
    """Concatenate assembled dataset directories (e.g. D1 train + D2 extra -> big)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metas = [json.loads((Path(d) / "meta.json").read_text()) for d in dirs]
    labelled = all((Path(d) / "cp.npy").exists() for d in dirs)
    keys = ["pieces", "geometry", "attacks", "legal", "target", "result", "game", "ply",
            "source_row"] + (["cp"] if labelled else [])
    game_offset, offsets, fens = 0, [np.zeros(1, np.int64)], []
    parts = {k: [] for k in keys}
    for d in dirs:
        d = Path(d)
        for k in keys:
            a = np.load(d / f"{k}.npy")
            if k == "game":
                a = a.astype(np.int64) + game_offset  # keep games distinct across sources
            parts[k].append(a)
        game_offset = int(max(int(p.max()) for p in parts["game"])) + 1
        o = np.load(d / "offsets.npy")
        offsets.append(o[1:] + offsets[-1][-1])
        fens.append((d / "fen.txt").read_text().rstrip("\n"))
    for k in keys:
        np.save(out / f"{k}.npy", np.concatenate(parts[k]), allow_pickle=False)
    np.save(out / "offsets.npy", np.concatenate(offsets), allow_pickle=False)
    (out / "fen.txt").write_text("\n".join(fens) + "\n")
    meta = dict(split="train", rows=int(sum(m["rows"] for m in metas)), labelled=labelled,
                sources=[str(d) for d in dirs], depth=sorted({m.get("depth") for m in metas},
                                                             key=str))
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# ---------------------------------------------------------------------------
# Training-time access: memory-mapped arrays + vectorised padding
# ---------------------------------------------------------------------------

def cp_to_value(cp):
    """Centipawns (side to move) -> expected score in [-1, 1] (Lichess logistic curve)."""
    cp = np.clip(np.asarray(cp, dtype=np.float64), -2000, 2000)
    return 2.0 / (1.0 + np.exp(-0.00368208 * cp)) - 1.0


class V6Dataset:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        load = lambda k: np.load(self.dir / f"{k}.npy", mmap_mode="r")  # noqa: E731
        self.pieces, self.geometry, self.attacks = load("pieces"), load("geometry"), load("attacks")
        self.legal, self.offsets = load("legal"), np.load(self.dir / "offsets.npy")
        self.target, self.result = np.load(self.dir / "target.npy"), np.load(self.dir / "result.npy")
        self.game = np.load(self.dir / "game.npy")
        self.cp = load("cp") if (self.dir / "cp.npy").exists() else None
        self.lengths = np.diff(self.offsets).astype(np.int64)
        if self.cp is not None:
            best = np.maximum.reduceat(np.asarray(self.cp), self.offsets[:-1])
            self.best_cp = best.astype(np.int64)

    def __len__(self):
        return len(self.target)

    def _ragged(self, flat, idx, fill):
        starts, lens = self.offsets[idx], self.lengths[idx]
        width = int(lens.max())
        grid = starts[:, None] + np.arange(width)[None]
        valid = np.arange(width)[None] < lens[:, None]
        out = np.asarray(flat)[np.where(valid, grid, 0)]
        out[~valid] = fill
        return out, valid

    def batch(self, idx):
        idx = np.sort(np.asarray(idx))
        legal, valid = self._ragged(self.legal, idx, -1)
        out = dict(
            pieces=np.asarray(self.pieces[idx]).astype(np.int64),
            geometry=np.asarray(self.geometry[idx]),
            attacks=np.unpackbits(np.asarray(self.attacks[idx]), axis=-1, bitorder="little"),
            legal=legal.astype(np.int64),
            target=self.target[idx].astype(np.int64),
            result=self.result[idx].astype(np.float32),
            game=self.game[idx],
        )
        if self.cp is not None:
            cp, _ = self._ragged(self.cp, idx, -32000)
            out["cp"] = cp.astype(np.float32)
            out["best_cp"] = self.best_cp[idx].astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m chess_policy.v6_data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("positions")
    p.add_argument("pgn", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--force-test", action="store_true")
    p.add_argument("--exclude", action="append", help="positions file to exclude (repeatable)")
    p.add_argument("--max-games", type=int, default=0)
    p.add_argument("--skip-games", type=int, default=0)
    l = sub.add_parser("label")
    l.add_argument("positions")
    l.add_argument("--out", required=True)
    l.add_argument("--engine", required=True)
    l.add_argument("--depth", type=int, default=10)
    l.add_argument("--workers", type=int, default=0)
    l.add_argument("--chunk", type=int, default=1000)
    l.add_argument("--shard", type=int, default=0)
    l.add_argument("--of", type=int, default=1)
    l.add_argument("--splits", default="train,dev,test")
    l.add_argument("--limit", type=int, default=0)
    l.add_argument("--stride", type=int, default=1, help="label every Nth position only")
    a = sub.add_parser("assemble")
    a.add_argument("positions")
    a.add_argument("--out", required=True)
    a.add_argument("--split", required=True, choices=list(SPLIT_CODES))
    a.add_argument("--labels")
    a.add_argument("--workers", type=int, default=0)
    a.add_argument("--max-rows", type=int, default=0)
    c = sub.add_parser("concat")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "positions":
        print(json.dumps(build_positions(args.pgn, args.out, force_test=args.force_test,
                                         exclude=args.exclude, max_games=args.max_games,
                                         skip_games=args.skip_games), indent=2))
    elif args.cmd == "label":
        label(args.positions, args.out, args.engine, args.depth, args.workers, args.chunk,
              args.shard, args.of, tuple(args.splits.split(",")), args.limit, stride=args.stride)
    elif args.cmd == "concat":
        print(json.dumps(concat(args.out, *args.dirs), indent=2))
    else:
        print(json.dumps(assemble(args.positions, args.out, args.split, args.labels,
                                  args.workers, args.max_rows), indent=2))


if __name__ == "__main__":
    main()
