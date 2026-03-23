#include "fjace.h"

#include "tt.h"

namespace Stockfish {

namespace {

int cho_matches = 0;
int cho_total = 0;
int han_matches = 0;
int han_total = 0;

}  // namespace

void fjace_reset() {
  cho_matches = 0;
  cho_total = 0;
  han_matches = 0;
  han_total = 0;
}

void fjace_analyze_played_move(const Position& pos, Move m) {
  bool ttHit;
  TTEntry* tte = TT.probe(pos.key(), ttHit);
  Move engineBest = ttHit ? tte->move() : MOVE_NONE;

  if (pos.side_to_move() == WHITE) {
    cho_total++;
    if (engineBest != MOVE_NONE && m == engineBest)
      cho_matches++;
  } else {
    han_total++;
    if (engineBest != MOVE_NONE && m == engineBest)
      han_matches++;
  }
}

double fjace_get_cho_els() {
  return cho_total > 0 ? double(cho_matches) / cho_total * 100.0 : 0.0;
}

double fjace_get_han_els() {
  return han_total > 0 ? double(han_matches) / han_total * 100.0 : 0.0;
}

}  // namespace Stockfish
