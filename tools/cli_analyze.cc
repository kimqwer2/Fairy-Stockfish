#include <fstream>
#include <iostream>

#include "../src/cheat_detection.h"

int main(int argc, char* argv[]) {
    std::string input, out = "game_report.json";
    CheatDetection::AnalyzeOptions options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) input = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out = argv[++i];
        else if (arg == "--depth" && i + 1 < argc) options.depth = std::stoi(argv[++i]);
        else if (arg == "--nnue-path" && i + 1 < argc) options.nnue_path = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) options.threads = std::stoi(argv[++i]);
    }
    if (input.empty()) {
        std::cerr << "Usage: fstockfish-cheat --input game.kif [--depth 18] [--threads 1] [--nnue-path file.nnue] [--out report.json]\n";
        return 1;
    }

    std::vector<std::string> warnings;
    auto game = CheatDetection::ParseGameFile(input, warnings);
    CheatDetection::RealEngineAdapter engine(options);
    auto report = CheatDetection::AnalyzeGame(game, engine, options);
    report.warnings.insert(report.warnings.end(), warnings.begin(), warnings.end());

    std::ofstream of(out);
    of << CheatDetection::ReportToJson(report);
    std::cout << "Wrote " << out << "\n";
    return 0;
}
