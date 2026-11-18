// Minimal UCI adapter around the team's own Module 12 engine (november.zip).
// The engine code is used unchanged: alpha-beta with TT, killers/history,
// null-move pruning, late-move reductions and the trained handcrafted evaluator
// (make_trained_params). Only this file is new.
//
// Build: ./build.sh <folder with the original module*.cpp files>
// (build.sh copies module12 with its main() renamed, since module10 resets the
//  main-renaming macro and module12's own main cannot be renamed from outside.)
// Supports: uci, isready, ucinewgame, position [startpos|fen ...] [moves ...],
//           go depth N (default depth 4), quit.
// The opening book is not used: it keys on the move history from the
// initial position, which positions given as FEN do not carry.

#include <iostream>
#include <sstream>
#include <string>

#include "module12_lib.cpp"  // = module12_strengthened.cpp with main() renamed

using namespace chess;

static bool apply_uci(Position& pos, const std::string& u) {
    MoveList ml;
    generate_legal(pos.b, ml);
    for (int i = 0; i < ml.n; ++i) {
        if (ml.moves[i].to_uci() == u) {
            apply_move_hash(pos, ml.moves[i]);
            return true;
        }
    }
    return false;
}

int main() {
    std::ios::sync_with_stdio(false);
    init_zobrist();
    g_tt.assign((size_t)TT_SIZE, TTEntry{});
    const std::string START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    Position pos;
    pos.load_fen(START);
    EvalParamsA params = make_trained_params();
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd == "uci") {
            std::cout << "id name TeamEngine-M12 (november.zip)\nid author team\nuciok" << std::endl;
        } else if (cmd == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (cmd == "ucinewgame") {
            g_tt.assign((size_t)TT_SIZE, TTEntry{});
        } else if (cmd == "position") {
            std::string tok, fen;
            ss >> tok;
            if (tok == "startpos") {
                fen = START;
                ss >> tok;  // maybe "moves"
            } else if (tok == "fen") {
                std::string part;
                for (int i = 0; i < 6 && ss >> part; ++i) {
                    if (part == "moves") { tok = part; break; }
                    fen += (fen.empty() ? "" : " ") + part;
                }
                if (tok != "moves") ss >> tok;
            }
            pos = Position();
            pos.load_fen(fen);
            if (tok == "moves") {
                std::string mv;
                while (ss >> mv) apply_uci(pos, mv);
            }
        } else if (cmd == "go") {
            int depth = 4;
            std::string tok;
            while (ss >> tok) {
                if (tok == "depth") ss >> depth;
            }
            MoveList ml;
            generate_legal(pos.b, ml);
            if (ml.n == 0) { std::cout << "bestmove 0000" << std::endl; continue; }
            SearchResult r = iterative_deepening_m12(pos, depth, 1e18, params, SearchOptions());
            std::cout << "info depth " << r.depth << " score cp " << r.score
                      << " nodes " << r.nodes << std::endl;
            std::cout << "bestmove " << r.best_move.to_uci() << std::endl;
        } else if (cmd == "quit") {
            break;
        }
    }
    return 0;
}
