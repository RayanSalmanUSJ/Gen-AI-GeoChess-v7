"""
Method: a local general-purpose LLM (via Ollama) asked to play chess.

Run on a machine with Ollama (https://ollama.com) and the models pulled, e.g.
  ollama pull llama3.2:3b
  ollama pull qwen3.5:9b        # use whatever tag your Ollama lists for Qwen
then:
  python llm_player.py bench --model llama3.2:3b --every 4 --out results/llm_llama3.2-3b.json
  python llm_player.py bench --model llama3.2:3b --every 4 --no-legal-list --out results/llm_llama3.2-3b_nolist.json

It scores the LLM on exactly the same held-out positions as every other method
(Stockfish 15.1 depth-12 prices from Prototype 2), and also reports what only
LLMs need: how often the first answer is a legal move, and after retries.

Prompt modes:
  default         FEN + side to move + recent moves + the list of legal moves (UCI)
  --no-legal-list FEN + side to move + recent moves only (tests rules knowledge)
If an answer is illegal or unparseable, the model is told so and asked again
(up to --retries times); after that a random legal move is played and the
position is counted as a failure.

The same LLMPlayer class plugs into tournament.py ("llm:<model>").
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request

import chess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(HERE, "..", "02_value_distillation")
ROOT = os.path.join(HERE, "..", "..")
for p in (P2, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")
SAN_RE = re.compile(r"\b(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

SYSTEM = ("You are a strong chess player. Reply with exactly one legal move in UCI notation "
          "(for example e2e4, g1f3, e7e8q) and nothing else.")


def ollama_chat(host, model, messages, temperature=0.0, timeout=300):
    body = json.dumps({"model": model, "messages": messages, "stream": False, "think": False,
                       "options": {"temperature": temperature, "num_predict": 64}}).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def describe(board, legal_list=True, history_plies=12):
    side = "White" if board.turn == chess.WHITE else "Black"
    lines = [f"Position (FEN): {board.fen()}", f"{side} to move."]
    if board.move_stack:
        tmp = board.copy()
        recent = []
        for _ in range(min(history_plies, len(tmp.move_stack))):
            recent.append(tmp.pop())
        recent.reverse()
        san, b2 = [], tmp
        for m in recent:
            san.append(b2.san(m))
            b2.push(m)
        lines.append("Recent moves: " + " ".join(san))
    if legal_list:
        lines.append("Legal moves: " + " ".join(m.uci() for m in board.legal_moves))
    lines.append("Your move (UCI only):")
    return "\n".join(lines)


def parse_move(board, text):
    text = THINK_RE.sub("", text)
    for tok in UCI_RE.findall(text):
        try:
            m = chess.Move.from_uci(tok)
        except ValueError:
            continue
        if m in board.legal_moves:
            return m
    for tok in SAN_RE.findall(text):  # some models answer in SAN anyway
        try:
            return board.parse_san(tok)
        except ValueError:
            continue
    return None


class LLMPlayer:
    def __init__(self, model, host="http://localhost:11434", legal_list=True, retries=2, seed=0):
        self.model, self.host, self.legal_list, self.retries = model, host, legal_list, retries
        self.rng = random.Random(seed)
        self.name = f"llm({model}{'' if legal_list else ', no list'})"
        self.stats = dict(moves=0, first_try_legal=0, legal_after_retries=0, fallbacks=0, seconds=0.0)

    def move(self, board):
        t = time.time()
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": describe(board, self.legal_list)}]
        mv = None
        for attempt in range(self.retries + 1):
            try:
                reply = ollama_chat(self.host, self.model, msgs)
            except Exception as e:  # server down, timeout: count as a failure
                reply = f"(error: {e})"
            mv = parse_move(board, reply)
            if mv is not None:
                if attempt == 0:
                    self.stats["first_try_legal"] += 1
                self.stats["legal_after_retries"] += 1
                break
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "That is not a legal move here. "
                      "Answer with one legal move in UCI notation only."}]
        if mv is None:
            self.stats["fallbacks"] += 1
            mv = self.rng.choice(list(board.legal_moves))
        self.stats["moves"] += 1
        self.stats["seconds"] += time.time() - t
        return mv


def bench(a):
    import encoding as enc
    from evaluate import choice_metrics, summarize
    from train import load_npz
    d = load_npz(os.path.join(P2, "data", "test.npz"))
    # rebuild move history so the prompt can show recent moves
    sys.path.insert(0, HERE)
    from gpt_movelm import game_sequences
    seqs = game_sequences(os.path.join(P2, "data", "test_games.jsonl"))
    sel = np.arange(0, len(d["fen"]), a.every)[: a.limit or None]
    player = LLMPlayer(a.model, a.host, legal_list=not a.no_legal_list, retries=a.retries)
    choice = []
    for n, i in enumerate(sel):
        b = chess.Board()
        for m in seqs[int(d["game"][i])][: int(d["ply"][i])]:
            b.push(m)
        mv = player.move(b)
        a_idx = enc.action_index(b, mv)
        choice.append(int(np.where(d["idx"][i] == a_idx)[0][0]))
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(sel)} positions, first-try legal "
                  f"{player.stats['first_try_legal'] / player.stats['moves']:.2f}", flush=True)
    n_legal = (d["idx"][sel] >= 0).sum(1)
    m = choice_metrics(d["cp"][sel], n_legal, np.array(choice))
    s = summarize(m, d["game"][sel])
    st = player.stats
    out = dict(player=player.name, model=a.model, legal_list=not a.no_legal_list,
               positions=int(len(sel)), every=a.every,
               cp_loss=s["cp_loss"], best_rate=s["best_rate"], blunder_rate=s["blunder_rate"],
               es_loss=s["es_loss"],
               first_try_legal_rate=st["first_try_legal"] / st["moves"],
               legal_after_retries_rate=st["legal_after_retries"] / st["moves"],
               fallback_rate=st["fallbacks"] / st["moves"],
               seconds_per_move=st["seconds"] / st["moves"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else round(v["mean"], 3)) for k, v in out.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--model", required=True)
    b.add_argument("--host", default="http://localhost:11434")
    b.add_argument("--every", type=int, default=4, help="use every n-th test position (4 -> 328)")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--retries", type=int, default=2)
    b.add_argument("--no-legal-list", action="store_true")
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    bench(a)


if __name__ == "__main__":
    main()
