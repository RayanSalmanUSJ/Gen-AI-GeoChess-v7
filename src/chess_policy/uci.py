"""Synchronous, immediate-policy UCI. History is retained solely for chess rules."""

import sys
import chess


def parse_position(tokens):
    if tokens and tokens[0] == "startpos":
        board, cursor = chess.Board(), 1
    elif len(tokens) >= 7 and tokens[0] == "fen":
        board, cursor = chess.Board(" ".join(tokens[1:7])), 7
    else:
        raise ValueError("expected startpos or six-field FEN")
    if not board.is_valid():
        raise ValueError("invalid position")
    if cursor < len(tokens):
        if tokens[cursor] != "moves":
            raise ValueError("expected moves")
        for text in tokens[cursor + 1 :]:
            board.push_uci(text)
    return board


def serve_uci(policy, source=None, sink=None):
    source, sink = source or sys.stdin, sink or sys.stdout
    board = chess.Board()

    def emit(text):
        print(text, file=sink, flush=True)

    for line in source:
        tokens = line.split()
        if not tokens:
            continue
        try:
            command = tokens[0]
            if command == "uci":
                emit("id name Geometric Move Transformer 0.5")
                emit("id author Chess Policy Project")
                emit("uciok")
            elif command == "isready":
                emit("readyok")
            elif command == "ucinewgame":
                board = chess.Board()
            elif command == "position":
                board = parse_position(tokens[1:])
            elif command == "go":
                result = policy.predict(board, top_k=0)
                move = result["move"]
                if "searchmoves" in tokens and move is not None:
                    allowed = []
                    for text in tokens[tokens.index("searchmoves") + 1 :]:
                        try:
                            parsed = chess.Move.from_uci(text)
                        except ValueError:
                            break
                        if parsed in board.legal_moves:
                            allowed.append(text)
                    move = next((x["uci"] for x in result["moves"] if x["uci"] in allowed), None)
                emit("info string direct policy search_nodes=0")
                emit("bestmove " + (move or "0000"))
            elif command == "quit":
                break
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            emit("info string error " + str(exc).replace("\n", " "))
            if tokens[0] == "go":
                emit("bestmove 0000")
