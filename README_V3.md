# Geometric Chess Policy System — v0.3

A current chess position becomes a probability distribution over its legal moves.
The network uses learned Q/K/V attention, dense SwiGLU feed-forward layers and
trainable top-two mixture-of-experts routing. This is a conditional next-token
model. Its output token is a complete UCI move, such as `e2e4` or `a7a8n`.

**Release evidence is in `RESULTS.md`.** The included checkpoint is a trained CPU
pilot. Larger configurations are provided; their weights have not been trained.
The code is a tested engineering foundation, not a claim of production deployment
history, a particular Elo, or reproduction of undisclosed GPT-4 internals.

## Run the shipped model

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,serve]'
chess-policy predict --show-tokens
chess-policy uci
```

For a CPU-only install, install PyTorch from its CPU wheel index before installing
this project. For an NVIDIA GPU, install a PyTorch build compatible with the GPU
and driver. The tested dependency versions are in `results/environment.json`.

```bash
chess-policy predict --fen 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1' --top-k 5
chess-policy serve --host 127.0.0.1 --port 8000
```

The HTTP service exposes `GET /health` and `POST /predict` with
`{"fen":"...", "top_k":5}`. Health includes the loaded checkpoint hash.
`top_k=0` returns all legal candidates. Output probabilities always refer to the
full legal basket; a truncated displayed list need not sum to one. Temperature
zero chooses the maximum, while positive temperature samples with an optional seed.

The UCI adapter chooses immediately; it does not spend the requested `movetime`
searching. It supports `position startpos`, complete FENs, move histories,
`go searchmoves`, terminal positions and readiness. It has no ponder worker.

## Exact architecture

1. **Position tokens:** 64 square tokens, including occupied and empty squares,
   plus one state token. Token identity, file, rank and D4 orbit have embeddings.
2. **Rule features:** attack counts by color, legal outgoing/incoming counts,
   en-passant square, castling rights, side to move and halfmove count.
3. **Attention geometry:** relative displacement, file/rank/diagonal alignment,
   king/knight displacement relations, directed attack edges and legal-move edges.
4. **Full board attention:** every square can attend to every square through
   learned Q, K and V projections. There is no arbitrary causal barrier across
   files or ranks. The board is the fully visible conditioning prefix.
5. **Residual layers:** RMSNorm, multihead attention, dense SwiGLU FFNs, alternating
   routed MoE FFNs, and optional dropout. The router uses FP32 softmax, top-k
   dispatch, a load-balance objective and router z loss. No tokens are dropped.
6. **Legal candidate decoder:** source and destination representations, relative
   displacement, promotion identity and global state form each move query. The
   queries cross-attend to the encoded board and pass through an FFN to logits.
7. **Legal softmax:** the model scores only supplied legal candidates. Padding
   receives negative infinity and zero probability. Reordering candidates merely
   reorders their scores; it does not change the selected move.

The global move identifier spans 64 × 64 × 5 = 20,480 combinations, but no
20,480-way output matrix is evaluated. Each position gets its actual variable
basket. No legal move is removed for having a low network score before inference.

Each move's number is a **logit**, not a centipawn evaluation or winning
probability. Stockfish is used as a source of recorded moves and, separately, an
arena opponent. Prediction imports no Stockfish engine, minimax, MCTS or value head.

## Rebuild the data and train

The archive includes compact PGNs: the first 2,500 games and the following 800
games from one official Stockfish Fishtest LTC archive. Their provenance and the
original archive checksum are in `data/provenance.json`. Annotations and variations
were removed; moves and headers were preserved. These are Stockfish development
engine games, not AlphaZero games. No engine evaluation is a training target.

```bash
chess-policy build-data --pgn data/fishtest_train.pgn.gz --out corpus
chess-policy build-data --pgn data/fishtest_holdout.pgn.gz --out holdout --force-test --exclude corpus
python scripts/audit_corpus.py corpus holdout
chess-policy train --data corpus --out runs/my_pilot --config configs/pilot.json
chess-policy evaluate --data holdout --checkpoint runs/my_pilot/best.pt --out results/my_test.json
```

Generated uncompressed arrays require approximately 516 MiB for these two corpora.
The zip omits them because the included compact PGNs reproduce them. The pipeline
validates PGNs, splits by initial opening FEN, globally deduplicates position keys,
stores attack graphs as packed bits, writes mmap arrays in shards, and commits a
manifest only after completion. Source and shard hashes are retained. SQLite holds
the deduplication index; shard/row shuffling is streamed and mmap handles are bounded.
Metadata still grows with the number of games and shards.

**Splitting limitation:** this corpus has FEN-defined opening starts. A collection
whose games all start from the ordinary initial position shares one opening group
and will not create usable independent partitions with this splitter. Prepare
opening-group FEN excerpts or externally curated independent corpora first. Never
fall back to randomly splitting adjacent positions to make validation appear easy.

Training uses AdamW, warmup/cosine learning rate, gradient clipping, correctly
weighted gradient accumulation, optional FP16/BF16 AMP, reproducible seeds, and
development-only checkpoint selection. `last.pt` includes optimizer, scaler, RNG,
epoch and schedule; `best.pt` contains inference weights. Checksums are mandatory.

Resume the same interrupted schedule and dataset:

```bash
chess-policy train --data corpus --out runs/my_pilot --config configs/pilot.json --resume runs/my_pilot/last.pt
```

Resume is at completed epoch boundaries. Exact recovery of partially processed
epochs is not implemented. Changing the configuration, data hash, or run directory
is rejected. Training is single-process, single-device. There is no claimed
distributed or expert-parallel support.

## Scale deliberately

`configs/pilot.json` is the measured CPU run. `configs/rtx3050.json` is a conservative
starting configuration for a roughly 4 GB GPU: small microbatches with accumulation.
Its CUDA execution and peak memory have not been measured here. Reduce the
microbatch if memory is insufficient; measure throughput before increasing width.

`configs/medium.json` and `configs/large.json` increase depth, width and expert
capacity. Inspect the exact counts without allocating the large weights:

```bash
chess-policy inspect-config --config configs/large.json
chess-policy train --data corpus --out runs/gpu --config configs/rtx3050.json
```

The large configuration has approximately 284 million parameters. It requires
substantially more training data and hardware than the pilot. Configuration support
does not mean a trained large model exists. MoE avoids executing every expert on
every token, but all expert weights and optimizer state still occupy memory.

To recover the original 159 MB archive and prepare larger experiments:

```bash
python scripts/download_source.py
python scripts/prepare_source.py
```

`build-data` accepts repeated `--pgn` arguments and streams `.pgn` or `.pgn.gz`.
Keep new source runs separate for a stronger external test; the provided train and
holdout still come from the same Fishtest experiment.

## Verify behavior and measure games

```bash
python -m pytest -q
python -m ruff check src scripts tests
python scripts/arena.py --stockfish /usr/games/stockfish --elo 1600 --pairs 2 --out results/arena.json
```

The arena alternates colors for paired openings, records every move and replays it,
writes PGNs, logs opponent settings and model hashes, and leaves ply-capped games
unfinished. It does not count unfinished games as draws. The opponent's time budget
and the policy's measured latency are reported separately. Small matches against
`UCI_Elo=1600` do not establish a calibrated 1600 rating.

Read `docs/ARCHITECTURE.md` for the mathematics and representation decisions,
`docs/OPERATIONS.md` for operational limits, and `RESULTS.md` for measured outcomes.

## Core study order

1. `encoding.py`: position contract, exact moves and geometric meaning.
2. `model.py`: QKV equations, residual blocks, SwiGLU, routing and candidate decoder.
3. `training.py`: loss, gradients, schedule, accumulation and checkpoint selection.
4. `data.py`: split integrity, deduplication and scalable input handling.
5. `inference.py`, `evaluation.py`, `scripts/arena.py`: behavior and evidence.

The dependency `python-chess` supplies the rules engine. Its move generator is
tested here with perft and special positions; this project does not claim to have
invented a second independent legality implementation. See `THIRD_PARTY.md`.
