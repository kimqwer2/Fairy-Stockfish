#ifndef NNUE_ERROR_LEARNING_H_INCLUDED
#define NNUE_ERROR_LEARNING_H_INCLUDED

#include <string>
#include "types.h"

namespace Stockfish {

class Position;

namespace NnueErrorLearning {

void init();
void on_options_changed();
void save();
void collect(const Position& pos, Value staticEval, Value searchEval, const std::string& result = "");
Value correction(const Position& pos, Value uncorrectedEval);
bool active_for(const Position& pos);

}
}

#endif
