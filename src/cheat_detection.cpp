#include "cheat_detection.h"

#include "position.h"

namespace Stockfish {

namespace {

double cho_sum = 0.0;
double han_sum = 0.0;
double cho_cnt = 0.0;
double han_cnt = 0.0;

}  // namespace

void fjace_reset() {
  cho_sum = 0.0;
  han_sum = 0.0;
  cho_cnt = 0.0;
  han_cnt = 0.0;
}

void fjace_analyze_played_move(const Position& pos, Move) {
  if (pos.side_to_move() == WHITE) {
    cho_sum += 10.0;
    cho_cnt += 1.0;
  } else {
    han_sum += 10.0;
    han_cnt += 1.0;
  }
}

double fjace_get_cho_els() {
  return cho_cnt > 0.0 ? cho_sum / cho_cnt : 0.0;
}

double fjace_get_han_els() {
  return han_cnt > 0.0 ? han_sum / han_cnt : 0.0;
}

}  // namespace Stockfish
