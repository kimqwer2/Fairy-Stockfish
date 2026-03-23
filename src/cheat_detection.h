#ifndef CHEAT_DETECTION_H_INCLUDED
#define CHEAT_DETECTION_H_INCLUDED

#include "types.h"

namespace Stockfish {

class Position;

void fjace_reset();
void fjace_analyze_played_move(const Position& pos, Move m);
double fjace_get_cho_els();
double fjace_get_han_els();

}  // namespace Stockfish

#endif
