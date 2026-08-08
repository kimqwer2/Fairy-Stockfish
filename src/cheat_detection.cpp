#include "cheat_detection.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <memory>

#include "bitboard.h"
#include "endgame.h"
#include "evaluate.h"
#include "piece.h"
#include "position.h"
#include "psqt.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "tune.h"
#include "uci.h"
#include "variant.h"

namespace CheatDetection {
namespace {

double Clamp(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }

std::string ReadFile(const std::string& path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> ParseMovesFromPgnLike(const std::string& text) {
    std::vector<std::string> moves;
    std::regex moveRegex("([a-z][0-9]{1,2}[a-z][0-9]{1,2}[+=-]?)");
    for (std::sregex_iterator i(text.begin(), text.end(), moveRegex), e; i != e; ++i)
        moves.push_back((*i)[1]);
    return moves;
}

std::vector<std::string> ParseMovesFromJson(const std::string& text) {
    std::vector<std::string> out;
    std::regex re("\"moves\"\\s*:\\s*\\[(.*?)\\]");
    std::smatch m;
    if (!std::regex_search(text, m, re))
        return out;
    std::regex itemRe("\"([^\"]+)\"");
    std::string body = m[1];
    for (std::sregex_iterator i(body.begin(), body.end(), itemRe), e; i != e; ++i)
        out.push_back((*i)[1]);
    return out;
}

std::string JsonEscape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

double Mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double Variance(const std::vector<double>& v, double mean) {
    if (v.size() < 2) return 0.0;
    double s = 0.0;
    for (double x : v) s += (x - mean) * (x - mean);
    return s / static_cast<double>(v.size());
}

double Corr(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.size() < 2) return 0.0;
    double ma = Mean(a), mb = Mean(b);
    double cov = 0.0, va = 0.0, vb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double da = a[i] - ma, db = b[i] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    if (va <= 1e-9 || vb <= 1e-9) return 0.0;
    return cov / std::sqrt(va * vb);
}

struct SideAcc {
    std::size_t top1 = 0;
    std::size_t top3 = 0;
    std::size_t criticalTop1 = 0;
    std::vector<double> cplValues;
    std::vector<double> engineDelta;
    std::vector<double> playerDelta;
};

void FillSideReport(const AnalyzeOptions& options, const SideAcc& acc, SideReport& side) {
    const double counted = static_cast<double>(std::max<std::size_t>(1, side.features.counted_moves));
    side.features.top1_rate = 100.0 * static_cast<double>(acc.top1) / counted;
    side.features.top3_rate = 100.0 * static_cast<double>(acc.top3) / counted;
    side.features.avg_cpl = Mean(acc.cplValues);
    side.features.cpl_variance = Variance(acc.cplValues, side.features.avg_cpl);
    side.features.critical_accuracy = side.features.critical_positions
        ? 100.0 * static_cast<double>(acc.criticalTop1) / side.features.critical_positions : 0.0;
    side.features.engine_correlation = Corr(acc.engineDelta, acc.playerDelta);

    const double corrScaled = 50.0 * (side.features.engine_correlation + 1.0);
    side.els = options.weights.top3 * (side.features.top3_rate / 100.0)
             + options.weights.avg_cpl * (1.0 / (1.0 + side.features.avg_cpl))
             + options.weights.critical * (side.features.critical_accuracy / 100.0)
             + options.weights.variance * (1.0 / (1.0 + side.features.cpl_variance))
             + options.weights.correlation * (corrScaled / 100.0);
    side.els = Clamp(side.els, 0.0, 100.0);
}

void BuildPosition(const std::vector<std::string>& movesSoFar,
                   std::vector<Stockfish::StateInfo>& states,
                   Stockfish::Position& pos,
                   std::string& warning) {
    using namespace Stockfish;
    const auto it = variants.find("janggi");
    pos.set(it->second, it->second->startFen, false, &states[0], Threads.main());
    for (const auto& moveStr : movesSoFar) {
        std::string ms = moveStr;
        Move mv = UCI::to_move(pos, ms);
        if (mv == MOVE_NONE) {
            warning = "Illegal move while replaying prefix: " + moveStr;
            break;
        }
        states.emplace_back();
        pos.do_move(mv, states.back());
    }
}

void EnsureEngineInit(const AnalyzeOptions& options) {
    using namespace Stockfish;
    static std::once_flag once;
    std::call_once(once, [&]() {
        static char app[] = "fstockfish-cheat";
        static char* argv[] = {app, nullptr};
        pieceMap.init();
        variants.init();
        CommandLine::init(1, argv);
        UCI::init(Options);
        Options["UCI_Variant"] = std::string("janggi");
        Tune::init();
        PSQT::init(variants.find("janggi")->second);
        Bitboards::init();
        Position::init();
        Bitbases::init();
        Endgames::init();
        Threads.set(static_cast<size_t>(std::max(1, options.threads)));
        Search::init();
        Search::clear();
    });

    Stockfish::Options["UCI_Variant"] = std::string("janggi");
    Stockfish::Options["Threads"] = std::to_string(std::max(1, options.threads));
    if (!options.nnue_path.empty())
        Stockfish::Options["EvalFile"] = options.nnue_path;
    Stockfish::Eval::NNUE::init();
}

} // namespace

RealEngineAdapter::RealEngineAdapter(const AnalyzeOptions& options) : options_(options) {
    EnsureEngineInit(options_);
}

std::vector<EngineMove> RealEngineAdapter::AnalyzeTopN(const std::vector<std::string>& movesSoFar,
                                                       const std::string&,
                                                       int depth,
                                                       int topN) {
    using namespace Stockfish;
    std::vector<StateInfo> states(1);
    std::string warning;
    Position pos;
    BuildPosition(movesSoFar, states, pos, warning);
    if (!warning.empty())
        return {};

    (void)depth;
    std::vector<EngineMove> out;
    std::vector<EngineMove> all;
    for (const auto& m : MoveList<LEGAL>(pos)) {
        StateInfo st;
        pos.do_move(m, st);
        const double score = -static_cast<double>(Eval::evaluate(pos));
        pos.undo_move(m);
        all.push_back({UCI::move(pos, m), score});
    }
    std::stable_sort(all.begin(), all.end(), [](const EngineMove& a, const EngineMove& b) {
        return a.eval > b.eval;
    });
    out.assign(all.begin(), all.begin() + std::min(static_cast<size_t>(topN), all.size()));
    return out;
}

int RealEngineAdapter::LegalMoveCount(const std::vector<std::string>& movesSoFar) {
    using namespace Stockfish;
    std::vector<StateInfo> states(1);
    std::string warning;
    Position pos;
    BuildPosition(movesSoFar, states, pos, warning);
    if (!warning.empty())
        return 0;
    return static_cast<int>(MoveList<LEGAL>(pos).size());
}

GameRecord ParseGameFile(const std::string& path, std::vector<std::string>& warnings) {
    GameRecord record;
    std::string text = ReadFile(path);
    if (text.find("\"moves\"") != std::string::npos)
        record.moves = ParseMovesFromJson(text);
    if (record.moves.empty())
        record.moves = ParseMovesFromPgnLike(text);
    if (record.moves.empty()) warnings.push_back("No moves parsed from input file");
    return record;
}

GameReport AnalyzeGame(const GameRecord& record, EngineAdapter& engine, const AnalyzeOptions& options) {
    GameReport report;
    std::vector<std::string> prefix;
    SideAcc choAcc, hanAcc;

    for (size_t i = 0; i < record.moves.size(); ++i) {
        MoveRecord mr;
        mr.move_number = i + 1;
        mr.move = record.moves[i];
        mr.counted = i + 1 > static_cast<size_t>(options.opening_exclusion);

        auto topN = engine.AnalyzeTopN(prefix, record.moves[i], options.depth, options.top_n);
        mr.topN_list = topN;
        if (topN.empty()) {
            report.warnings.push_back("Engine returned no lines for move " + std::to_string(i + 1));
            report.moves.push_back(mr);
            prefix.push_back(record.moves[i]);
            continue;
        }

        mr.eval_best = topN[0].eval;
        mr.eval_second_best = topN.size() > 1 ? topN[1].eval : topN[0].eval;
        mr.difficulty = mr.eval_best - mr.eval_second_best;
        mr.is_critical = mr.difficulty >= options.difficulty_threshold;

        for (size_t r = 0; r < topN.size(); ++r)
            if (topN[r].move == record.moves[i]) {
                mr.engine_rank = static_cast<int>(r + 1);
                mr.eval_played = topN[r].eval;
                break;
            }
        if (mr.engine_rank < 0) {
            mr.eval_played = topN.back().eval - 100.0;
            mr.engine_rank = options.top_n + 1;
        }
        mr.cpl = mr.eval_best - mr.eval_played;

        if (engine.LegalMoveCount(prefix) <= options.forced_move_threshold || mr.difficulty < options.difficulty_threshold)
            mr.counted = false;

        const bool isCho = (mr.move_number % 2 == 1);
        SideReport& side = isCho ? report.cho : report.han;
        SideAcc& acc = isCho ? choAcc : hanAcc;

        if (mr.counted) {
            ++side.features.counted_moves;
            acc.cplValues.push_back(mr.cpl);
            acc.engineDelta.push_back(mr.eval_best - mr.eval_second_best);
            acc.playerDelta.push_back(mr.eval_best - mr.eval_played);
            if (mr.engine_rank == 1) ++acc.top1;
            if (mr.engine_rank <= 3) ++acc.top3;
            if (mr.is_critical) {
                ++side.features.critical_positions;
                if (mr.engine_rank == 1) ++acc.criticalTop1;
            }
        }

        report.moves.push_back(mr);
        prefix.push_back(record.moves[i]);
    }

    FillSideReport(options, choAcc, report.cho);
    FillSideReport(options, hanAcc, report.han);
    report.els_total = (report.cho.els + report.han.els) / 2.0;

    std::ostringstream reason;
    reason << "Cho(top3=" << report.cho.features.top3_rate << "%, cpl=" << report.cho.features.avg_cpl
           << ") Han(top3=" << report.han.features.top3_rate << "%, cpl=" << report.han.features.avg_cpl << ")";
    report.detection_reasoning = reason.str();
    return report;
}

std::string ReportToJson(const GameReport& report) {
    std::ostringstream out;
    out << "{\n  \"els_cho\": " << report.cho.els
        << ",\n  \"els_han\": " << report.han.els
        << ",\n  \"els_total\": " << report.els_total
        << ",\n  \"cho\": {\"top1_rate\": " << report.cho.features.top1_rate
        << ", \"top3_rate\": " << report.cho.features.top3_rate
        << ", \"avg_cpl\": " << report.cho.features.avg_cpl
        << ", \"cpl_variance\": " << report.cho.features.cpl_variance
        << ", \"critical_accuracy\": " << report.cho.features.critical_accuracy
        << ", \"engine_correlation\": " << report.cho.features.engine_correlation
        << "},\n  \"han\": {\"top1_rate\": " << report.han.features.top1_rate
        << ", \"top3_rate\": " << report.han.features.top3_rate
        << ", \"avg_cpl\": " << report.han.features.avg_cpl
        << ", \"cpl_variance\": " << report.han.features.cpl_variance
        << ", \"critical_accuracy\": " << report.han.features.critical_accuracy
        << ", \"engine_correlation\": " << report.han.features.engine_correlation
        << "},\n  \"detection_reasoning\": \"" << JsonEscape(report.detection_reasoning) << "\",\n";

    out << "  \"moves\": [\n";
    for (size_t i = 0; i < report.moves.size(); ++i) {
        const auto& m = report.moves[i];
        out << "    {\"move_number\": " << m.move_number
            << ", \"move\": \"" << JsonEscape(m.move) << "\", \"engine_rank\": " << m.engine_rank
            << ", \"eval_best\": " << m.eval_best << ", \"eval_played\": " << m.eval_played
            << ", \"eval_second_best\": " << m.eval_second_best << ", \"difficulty\": " << m.difficulty
            << ", \"is_critical\": " << (m.is_critical ? "true" : "false")
            << ", \"cpl\": " << m.cpl << "}" << (i + 1 == report.moves.size() ? "\n" : ",\n");
    }
    out << "  ],\n  \"warnings\": [";
    for (size_t i = 0; i < report.warnings.size(); ++i)
        out << (i ? ", " : "") << "\"" << JsonEscape(report.warnings[i]) << "\"";
    out << "]\n}\n";
    return out.str();
}

} // namespace CheatDetection
