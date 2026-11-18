# A geometric conditional language model for chess

The machine implements

\[
\pi_\theta(a\mid s)=\frac{\exp z_\theta(s,a)}
{\sum_{b\in\mathcal L(s)}\exp z_\theta(s,b)},\quad a\in\mathcal L(s).
\]

The labels are moves recorded in elite engine games. The objective is
`-log pi(recorded_move | current_position)`, plus router regularization.
This is behavioral cloning with a structured output vocabulary. A probability of
0.4 means 40% of the model's categorical probability mass, not a 40% chance of winning.

## Discrete input, continuous learning, discrete output

The square `e4`, a white pawn, and `e2e4` are discrete objects. Embeddings map their
identities into vectors. Attention, nonlinear FFNs and gradient descent operate on
those vectors. Softmax turns final logits into a distribution; argmax selects a
legal discrete move. The legality engine is the boundary that keeps the output in
the allowed set. It does not tell the learner which legal move is strategically good.

This implements the useful computational content of the proposed continuous
analogy. It is not the gamma function, a differentiable minimax solver, or a proof
that continuous optimization finds the optimal chess move. More precisely,
`Gamma(n+1)=n!`; Bohr–Mollerup characterizes the normalized positive log-convex
solution on the positive reals. Complex gamma has poles, and complex log-convexity
is not the condition used in that theorem.

## Why a fully visible board prefix?

Ordering the square tokens from a1 to h8 is a serialization convention. It should
not prevent a token on a1 from seeing h8. Every input square therefore attends to
every other square. No future *move label* is in that context. The conditional
model predicts one move token after the whole position has been provided.

Across a game, the next position is produced by exact chess rules after each
selected move. This induces a product of next-move probabilities along the game,
while avoiding a requirement to infer the board from a long move transcript.
It is a structured conditional language model, not a raw character model trained
to reconstruct FEN syntax. Its readable tokens are printed by `--show-tokens`.

## QKV and the geometric bias

For head h, normalized board states X produce learned projections:

\[
Q_h=XW^Q_h,\quad K_h=XW^K_h,\quad V_h=XW^V_h,
\qquad A_h=\operatorname{softmax}(Q_hK_h^T/\sqrt{d_h}+G_h).
\]

`G_h` combines learned weights on known relations: relative displacement,
alignments, attacks, reverse attacks, legal edges and reverse legal edges.
These are biases, not masks excluding interaction between unrelated pieces.
The board can learn a relation between a pawn and a distant king even when
neither attacks the other directly. Q, K, V and all geometric bias weights train.

Square tokens include empty squares because an outpost, escape square, open file
or landing square can matter before any piece occupies it. Occupied tokens still
carry the requested tuple: piece type, color, square and its attack/reach information.
Defended squares and pinned-piece attacks are retained in the attack graph; exact
legal moves are encoded separately. Confusing these graphs would break king safety.

## Nonlinearity and experts

Each block has residual attention followed by a residual FFN. Dense SwiGLU computes
`W_down(silu(W_gate x) * W_up x)`. Alternating FFN blocks use the same function
inside several separately parameterized experts. A learned router chooses two
experts per token and combines their outputs using normalized selected probabilities.

Routing uses FP32 probabilities even under AMP. No capacity limit discards tokens.
The auxiliary balance loss is `E * sum(mean_probability * assignment_fraction)`;
router z loss is the mean squared `logsumexp` of routing logits. The balance loss
is approximately one under balanced routing, not zero. Reported expert loads are
fractions of assignments, summing to one across experts; each token has two assignments.

Neither a head nor an expert is labeled "king safety" or "pawn structure". They
may learn such features, but gradients and routing counts alone do not establish
that interpretation. No expert is a hidden minimax searcher.

## Move decoding and its efficiency

For each exact legal move, the decoder gathers its source and destination states,
adds promotion and displacement embeddings and the global state representation,
then cross-attends to the board. A shared FFN produces its logit. The legal list
is variable-length and candidate order is irrelevant. Promotions are distinct tokens.

The board representation is computed once per position. A basket of 30 moves
reuses that computation instead of evaluating 30 successor boards independently.
The design avoids both a value head and a large unconstrained output matrix.
All legal candidates are scored, including quiet moves, sacrifices and underpromotions.

## What the geometry does and does not guarantee

The board is a finite grid, not a torus with movement modulo eight. Bitboards are
64-bit characteristic vectors; XOR is GF(2) addition, while AND/OR are set operations.
The D4 group has eight square permutations, with 80 fixed-square incidences and
10 orbits by Burnside's lemma. Orbit IDs are supplied as geometric features.

This model is not mathematically D4-invariant: absolute file/rank and directional
pawn/castling information are also supplied. No unsupported factor-of-eight sample
efficiency claim follows. Arbitrarily rotating games would introduce wrong labels.

The neural input includes side to move, castling rights and en-passant state.
A FEN alone cannot encode every repetition fact. UCI preserves the move stack for
rule adjudication; the neural model is deliberately history-free. Automatic terminal
draws are handled before prediction. Claimable draws are handled by the arena.

## What this learns and what remains unproved

Stockfish's recorded action is a supervision signal, not a unique proven optimum.
Several moves can be equally good; a different legal prediction need not be a blunder.
Conversely, high average imitation accuracy can hide rare catastrophic mistakes.
Accuracy, log loss, calibration and actual game results answer different questions.

The architecture is inspired by published transformer and sparse-expert work.
GPT-4's report does not disclose the detailed architecture needed to reproduce it.
The shared next-token loss makes these techniques applicable; it does not establish
a mathematical isomorphism between all aspects of language and adversarial chess,
or a guarantee that scaling parameters alone yields a specified Elo.

## Primary references

- Vaswani et al., Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Fedus et al., Switch Transformers: https://arxiv.org/abs/2101.03961
- Shazeer, GLU Variants Improve Transformer: https://arxiv.org/abs/2002.05202
- OpenAI, GPT-4 Technical Report: https://arxiv.org/abs/2303.08774
- Ruoss et al., Amortized Planning with Large-Scale Transformers: A Case Study on Chess:
  https://arxiv.org/abs/2402.04494
- Official Stockfish Fishtest games: https://huggingface.co/datasets/official-stockfish/fishtest_pgns

Ruoss et al. demonstrate strong search-free chess after large-scale distillation,
including action-value modeling. Their reported rating is not evidence for the
strength of this different, direct behavioral-cloning checkpoint.
