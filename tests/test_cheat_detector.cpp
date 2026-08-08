#include <cassert>
#include <fstream>
#include <iostream>

#include "../src/cheat_detection.h"

using namespace CheatDetection;

class StubEngine final : public EngineAdapter {
  public:
    std::vector<EngineMove> AnalyzeTopN(const std::vector<std::string>& movesSoFar,
                                        const std::string&, int, int) override {
        if (movesSoFar.size() % 2 == 0)
            return {{"a1a2", 100}, {"a1a3", 20}, {"a1a4", -10}};
        return {{"b1b2", 90}, {"b1b3", 40}, {"b1b4", -30}};
    }
    int LegalMoveCount(const std::vector<std::string>&) override { return 20; }
};

int main() {
    {
        std::ofstream f("/tmp/cd.json");
        f << "{\"moves\":[\"a1a2\",\"b1b3\"]}";
        std::vector<std::string> w;
        auto g = ParseGameFile("/tmp/cd.json", w);
        assert(g.moves.size() == 2);
    }
    {
        std::ofstream f("/tmp/cd.pgn");
        f << "1. a1a2 b1b2 2. a2a3 b2b3";
        std::vector<std::string> w;
        auto g = ParseGameFile("/tmp/cd.pgn", w);
        assert(g.moves.size() == 4);
    }

    GameRecord game;
    game.moves = {"a1a2", "b1b4", "a1a2", "b1b2"};
    AnalyzeOptions opt;
    opt.opening_exclusion = 0;
    opt.difficulty_threshold = 10.0;
    StubEngine engine;
    auto report = AnalyzeGame(game, engine, opt);
    assert(report.cho.features.counted_moves == 2);
    assert(report.han.features.counted_moves == 2);
    assert(report.cho.els >= 0.0 && report.cho.els <= 100.0);
    assert(report.han.els >= 0.0 && report.han.els <= 100.0);
    std::cout << "test_cheat_detector passed\n";
}
