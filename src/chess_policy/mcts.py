"""Batched PUCT Monte Carlo Tree Search (MCTS) for GeoChess.

Integrates V6Policy (policy prior and value head) with:
1. Batched leaf evaluations using virtual loss.
2. Terminal state detection (checkmate, stalemate, draws).
3. PUCT selection with configurable c_puct, FPU, and simulation budgets.
4. Export of search statistics and AlphaZero-style visit distributions.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn.functional as F

from .encoding import collate, encode, move_id
from .v6_train import to_device


def _terminal_value(board: chess.Board) -> Optional[float]:
    """Terminal value from the perspective of the side to move.

    If side to move is checkmated: -1.0.
    If draw (stalemate, repetition, 50-move, insufficient material): 0.0.
    If game is ongoing: None.
    """
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    return None


class MCTSNode:
    """A node in the MCTS tree representing a board position."""

    def __init__(self, parent: Optional["MCTSNode"] = None, prior: float = 0.0):
        self.parent = parent
        self.prior = float(prior)
        self.visit_count = 0
        self.total_value = 0.0  # W (from this node's player perspective)
        self.virtual_loss = 0
        self.is_expanded = False
        self.terminal_value: Optional[float] = None
        # Mapping: chess.Move -> MCTSNode
        self.children: Dict[chess.Move, "MCTSNode"] = {}

    def q_value_for_parent(
        self, parent_q: float, fpu_reduction: float = 0.0, vl_weight: float = 0.1
    ) -> float:
        """Action-value Q from the perspective of the parent player.

        From parent perspective: child value is negated (-self.total_value).
        Unvisited nodes use First Play Urgency (FPU): parent_q - fpu_reduction.
        Virtual loss penalizes Q to prevent batch collisions.
        """
        total_n = self.visit_count + self.virtual_loss
        if total_n == 0:
            return parent_q - fpu_reduction
        return (-self.total_value - vl_weight * self.virtual_loss) / total_n

    def u_value(self, c_puct: float, sqrt_parent_visits: float) -> float:
        """PUCT exploration bonus U(s, a)."""
        return c_puct * self.prior * sqrt_parent_visits / (1.0 + self.visit_count + self.virtual_loss)

    def select_child(
        self, c_puct: float, parent_q: float, fpu_reduction: float = 0.0, vl_weight: float = 0.1
    ) -> Tuple[chess.Move, "MCTSNode"]:
        """Select child maximizing Q(s, a) + U(s, a) from current node's perspective."""
        sqrt_n = math.sqrt(max(1, self.visit_count))
        best_score = -float("inf")
        best_move = None
        best_child = None

        for move, child in self.children.items():
            q = child.q_value_for_parent(parent_q, fpu_reduction, vl_weight)
            u = child.u_value(c_puct, sqrt_n)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def expand(self, move_priors: List[Tuple[chess.Move, float]]):
        """Expand node with legal moves and their policy priors."""
        self.is_expanded = True
        for move, prior in move_priors:
            self.children[move] = MCTSNode(parent=self, prior=prior)


@dataclass
class MCTSConfig:
    c_puct: float = 2.0
    simulations: int = 64
    batch_size: int = 4
    virtual_loss_weight: float = 0.1
    fpu_reduction: float = 0.0
    temperature: float = 0.0  # 0.0 for argmax (evaluation), >0 for sampling
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0  # Set > 0 for root exploration noise in training


class MCTS:
    """Batched PUCT Monte Carlo Tree Search."""

    def __init__(self, model: torch.nn.Module, device: torch.device, config: Optional[MCTSConfig] = None):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.config = config or MCTSConfig()

    @torch.no_grad()
    def _evaluate_batch(self, boards: List[chess.Board]) -> Tuple[List[List[Tuple[chess.Move, float]]], List[float]]:
        """Batch-evaluate leaves with V6Policy: returns (list of (move, prior) per board, list of values)."""
        if not boards:
            return [], []

        examples = [encode(b, validate=False) for b in boards]
        batch = collate(examples)
        b = to_device(batch, self.device)

        self.model.eval()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.model(b["pieces"], b["geometry"], b["attacks"], b["legal"])

        logits = out["logits"].float()
        values = out["value"].float().cpu().numpy().tolist()

        batch_priors = []
        for i, board in enumerate(boards):
            legal_moves = sorted(board.legal_moves, key=move_id)
            n_legal = len(legal_moves)
            row_logits = logits[i, :n_legal]
            priors = F.softmax(row_logits, dim=-1).cpu().numpy()
            batch_priors.append([(m, float(p)) for m, p in zip(legal_moves, priors)])

        return batch_priors, values

    def search(self, root_board: chess.Board) -> Dict:
        """Run PUCT MCTS from root_board and return best move and search statistics."""
        legal_moves = list(root_board.legal_moves)
        if not legal_moves:
            return {"best_move": None, "visits": {}, "policy_target": {}, "root_value": 0.0, "status": "terminal"}

        if len(legal_moves) == 1:
            move = legal_moves[0]
            return {"best_move": move, "visits": {move.uci(): self.config.simulations},
                    "policy_target": {move.uci(): 1.0}, "root_value": 0.0, "status": "forced"}

        root = MCTSNode()
        term = _terminal_value(root_board)
        if term is not None:
            root.terminal_value = term
            return {"best_move": None, "visits": {}, "policy_target": {}, "root_value": term, "status": "terminal"}

        # Expand root immediately
        root_priors, root_vals = self._evaluate_batch([root_board])
        priors = root_priors[0]

        # Optional Dirichlet noise at root for exploration during self-play
        if self.config.dirichlet_epsilon > 0 and len(priors) > 1:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(priors))
            eps = self.config.dirichlet_epsilon
            priors = [(m, (1.0 - eps) * p + eps * n) for (m, p), n in zip(priors, noise)]

        root.expand(priors)
        root.total_value = root_vals[0]
        root.visit_count = 1

        sims_done = 1
        batch_size = max(1, self.config.batch_size)

        while sims_done < self.config.simulations:
            pending_leaves: List[Tuple[MCTSNode, chess.Board, List[Tuple[MCTSNode, chess.Move]]]] = []

            for _ in range(min(batch_size, self.config.simulations - sims_done)):
                node = root
                board = root_board.copy(stack=False)
                path: List[Tuple[MCTSNode, chess.Move]] = []

                # Selection: descend tree until reaching an unexpanded or terminal node
                while node.is_expanded and not node.terminal_value is not None and node.children:
                    parent_q = node.total_value / max(1, node.visit_count)
                    move, child = node.select_child(
                        self.config.c_puct,
                        parent_q,
                        self.config.fpu_reduction,
                        self.config.virtual_loss_weight,
                    )
                    path.append((node, move))
                    # Apply virtual loss to selected child
                    child.virtual_loss += 1
                    board.push(move)
                    node = child

                # Terminal check
                term = _terminal_value(board)
                if term is not None:
                    node.terminal_value = term
                    # Immediately backpropagate terminal value and clear virtual loss
                    curr_val = term
                    for p_node, p_move in reversed(path):
                        child_node = p_node.children[p_move]
                        child_node.virtual_loss -= 1
                        child_node.visit_count += 1
                        child_node.total_value += curr_val
                        curr_val = -curr_val
                    root.visit_count += 1
                    sims_done += 1
                else:
                    pending_leaves.append((node, board, path))

            if not pending_leaves:
                continue

            # Batched leaf evaluation
            leaf_boards = [item[1] for item in pending_leaves]
            leaf_priors, leaf_values = self._evaluate_batch(leaf_boards)

            # Expansion and backpropagation
            for (leaf_node, _, path), priors, leaf_val in zip(pending_leaves, leaf_priors, leaf_values):
                if not leaf_node.is_expanded:
                    leaf_node.expand(priors)

                # Backpropagate: leaf_val is from the perspective of the player to move at leaf
                curr_val = leaf_val
                for p_node, p_move in reversed(path):
                    child_node = p_node.children[p_move]
                    child_node.virtual_loss -= 1
                    child_node.visit_count += 1
                    child_node.total_value += curr_val
                    curr_val = -curr_val

                root.visit_count += 1
                sims_done += 1

        # Extract visit distribution from root
        visits = {move.uci(): child.visit_count for move, child in root.children.items()}
        total_visits = sum(visits.values())

        if total_visits == 0:
            best_move = legal_moves[0]
            policy_target = {best_move.uci(): 1.0}
        elif self.config.temperature == 0.0:
            # Deterministic selection: highest visit count, tie-break by prior
            best_move = max(root.children.items(), key=lambda item: (item[1].visit_count, item[1].prior))[0]
            policy_target = {uci: cnt / total_visits for uci, cnt in visits.items()}
        else:
            # Temperature-scaled visit distribution
            t = self.config.temperature
            weights = {uci: cnt ** (1.0 / t) for uci, cnt in visits.items()}
            sum_w = sum(weights.values())
            policy_target = {uci: w / sum_w for uci, w in weights.items()}
            ucis = list(policy_target.keys())
            probs = [policy_target[u] for u in ucis]
            chosen_uci = np.random.choice(ucis, p=probs)
            best_move = chess.Move.from_uci(chosen_uci)

        root_value = float(root.total_value / max(1, root.visit_count))

        return {
            "best_move": best_move,
            "visits": visits,
            "policy_target": policy_target,
            "root_value": root_value,
            "simulations": sims_done,
            "status": "ok",
        }
