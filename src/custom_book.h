#ifndef CUSTOM_BOOK_H_INCLUDED
#define CUSTOM_BOOK_H_INCLUDED

#include <string>
#include <unordered_map>

namespace Stockfish {

extern std::unordered_map<std::string, std::string> openingBook;

void load_custom_book();

} // namespace Stockfish

#endif
