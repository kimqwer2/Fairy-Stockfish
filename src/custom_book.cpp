#include "custom_book.h"

#include <cctype>
#include <deque>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

#include "movegen.h"
#include "position.h"
#include "thread.h"
#include "uci.h"
#include "variant.h"

namespace Stockfish {

std::unordered_map<std::string, std::string> openingBook;

namespace {

std::string strip_comments(const std::string& line) {
    std::string out;
    bool inBrace = false;
    for (char c : line) {
        if (c == '{') inBrace = true;
        else if (c == '}') inBrace = false;
        else if (!inBrace) out += c;
    }
    return out;
}

bool is_result_token(const std::string& t) {
    return t == "*" || t == "1-0" || t == "0-1" || t == "1/2-1/2";
}

Move janggi_san_to_move(const Position& pos, const std::string& san) {
    static const std::regex moveRe("^([RHEACKP]?)([a-i0-9]{0,2})([a-i])(10|[0-9])$");
    std::smatch m;
    if (!std::regex_match(san, m, moveRe))
        return MOVE_NONE;

    const std::string pieceDesignator = m[1].str();
    const std::string hint = m[2].str();
    const std::string toFile = m[3].str();
    const std::string toRank = m[4].str();
    const std::string dst = toFile + toRank;

    Move match = MOVE_NONE;
    for (const auto& mv : MoveList<LEGAL>(pos)) {
        std::string u = UCI::move(pos, mv);
        if (u.size() < 4)
            continue;

        std::string from = u.substr(0, u.size() > 5 && u[1] == '@' ? 3 : 2);
        std::string to = u.substr(from.size(), 2);
        if (to != dst)
            continue;

        Piece pc = pos.moved_piece(mv);
        char designator = static_cast<char>(std::toupper(pos.piece_to_char()[pc]));
        if (!pieceDesignator.empty() && designator != pieceDesignator[0])
            continue;

        if (!hint.empty()) {
            bool ok = false;
            if (hint.size() == 1) {
                ok = from[0] == hint[0] || (from.size() == 3 && from[2] == hint[0]) || from[1] == hint[0];
            } else {
                ok = from == hint;
            }
            if (!ok)
                continue;
        }

        if (match != MOVE_NONE)
            return MOVE_NONE;
        match = mv;
    }

    return match;
}

} // namespace

void load_custom_book() {
    openingBook.clear();

    std::ifstream in("book.txt");
    if (!in)
        return;

    const Variant* janggi = variants.find("janggi")->second;
    std::string line;
    std::string currentFen = janggi->startFen;
    std::string movetext;

    while (std::getline(in, line)) {
        if (line.empty())
            continue;

        if (line[0] == '[') {
            if (line.rfind("[FEN \"", 0) == 0) {
                size_t firstQuote = line.find('"');
                size_t lastQuote = line.rfind('"');
                if (firstQuote != std::string::npos && lastQuote > firstQuote)
                    currentFen = line.substr(firstQuote + 1, lastQuote - firstQuote - 1);
            }
            continue;
        }

        movetext += " " + strip_comments(line);
    }

    std::istringstream is(movetext);
    Position pos;
    std::deque<StateInfo> states(1);
    pos.set(janggi, currentFen, false, &states.back(), Threads.main());

    std::string token;
    while (is >> token) {
        if (token.find('.') != std::string::npos || is_result_token(token))
            continue;

        Move m = janggi_san_to_move(pos, token);
        if (m == MOVE_NONE)
            continue;

        openingBook[pos.fen()] = UCI::move(pos, m);
        states.emplace_back();
        pos.do_move(m, states.back());
    }

    sync_cout << "info string custom book loaded entries " << openingBook.size() << sync_endl;
}

} // namespace Stockfish
