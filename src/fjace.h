#ifndef FJACE_H
#define FJACE_H

#include "position.h"

namespace Stockfish {

void fjace_reset();
void fjace_analyze_played_move(const Position& pos, Move m);
double fjace_get_cho_els();
double fjace_get_han_els();

}  // namespace Stockfish

#endif
