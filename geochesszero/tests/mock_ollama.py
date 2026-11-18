"""A fake Ollama /api/chat server for testing llm_player.py without a model.

It reads the legal-move list from the prompt and answers like a weak model:
sometimes rambling text, sometimes an illegal move, otherwise a legal move.
"""

import json
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

RNG = random.Random(0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = "\n".join(m["content"] for m in body["messages"] if m["role"] == "user")
        legal = re.findall(r"Legal moves: (.*)", prompt)
        moves = legal[-1].split() if legal else ["e2e4"]
        r = RNG.random()
        if r < 0.15:
            text = "<think>hmm</think> I think the best plan is to improve my position."
        elif r < 0.30:
            text = "a1a8"  # usually illegal
        else:
            text = f"Best move: {RNG.choice(moves)}"
        out = json.dumps({"message": {"role": "assistant", "content": text}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def start(port=0):
    srv = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
