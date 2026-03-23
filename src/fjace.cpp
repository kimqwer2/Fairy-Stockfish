#include "fjace.h"

#include "tt.h"
#include "evaluate.h"

#include <algorithm>

namespace Stockfish {

namespace {

int cho_count = 0;
int han_count = 0;
double cho_loss_sum = 0.0;
double han_loss_sum = 0.0;

}  // namespace

void fjace_reset() {
  cho_count = 0;
  han_count = 0;
  cho_loss_sum = 0.0;
  han_loss_sum = 0.0;
}

void fjace_analyze_played_move(const Position& pos, Move m) {
  bool ttHit;
  TTEntry* tte = TT.probe(pos.key(), ttHit);
  if (ttHit && tte->move() != MOVE_NONE) {
      int loss = (m == tte->move()) ? 0 : 20;

      if (pos.side_to_move() == WHITE) {
          cho_count++;
          cho_loss_sum += loss;
      } else {
          han_count++;
          han_loss_sum += loss;
      }
  }
}

double fjace_get_cho_els() {
  if (cho_count == 0)
      return 0.0;
  return std::max(0.0, 100.0 - (cho_loss_sum / cho_count * 2.0));
}

double fjace_get_han_els() {
  if (han_count == 0)
      return 0.0;
  return std::max(0.0, 100.0 - (han_loss_sum / han_count * 2.0));
}

}  // namespace Stockfish
