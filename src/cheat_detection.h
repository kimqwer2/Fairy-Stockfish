#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace CheatDetection {

struct EngineMove {
    std::string move;
    double eval = 0.0;
};

struct MoveRecord {
    std::size_t move_number = 0;
    std::string move;
    int engine_rank = -1;
    double eval_best = 0.0;
    double eval_played = 0.0;
    double eval_second_best = 0.0;
    std::vector<EngineMove> topN_list;
    double difficulty = 0.0;
    bool is_critical = false;
    double cpl = 0.0;
    bool counted = true;
};

struct GameRecord {
    std::vector<std::string> moves;
    std::string variant = "janggi";
};

struct ModelWeights {
    double top3 = 30.0;
    double avg_cpl = 25.0;
    double critical = 30.0;
    double variance = 10.0;
    double correlation = 5.0;
};

struct AnalyzeOptions {
    int depth = 18;
    int top_n = 3;
    int opening_exclusion = 10;
    double difficulty_threshold = 70.0;
    int forced_move_threshold = 2;
    int threads = 1;
    std::string nnue_path;
    ModelWeights weights;
};

struct GameFeatures {
    double top1_rate = 0.0;
    double top3_rate = 0.0;
    double avg_cpl = 0.0;
    double cpl_variance = 0.0;
    double critical_accuracy = 0.0;
    double engine_correlation = 0.0;
    std::size_t counted_moves = 0;
    std::size_t critical_positions = 0;
};

struct SideReport {
    GameFeatures features;
    double els = 0.0;
};

struct GameReport {
    std::vector<MoveRecord> moves;
    SideReport cho;
    SideReport han;
    double els_total = 0.0;
    std::string detection_reasoning;
    std::vector<std::string> warnings;
};

class EngineAdapter {
  public:
    virtual ~EngineAdapter() = default;
    virtual std::vector<EngineMove> AnalyzeTopN(const std::vector<std::string>& movesSoFar,
                                                const std::string& playedMove,
                                                int depth,
                                                int topN) = 0;
    virtual int LegalMoveCount(const std::vector<std::string>& movesSoFar) = 0;
};

class RealEngineAdapter : public EngineAdapter {
  public:
    explicit RealEngineAdapter(const AnalyzeOptions& options);
    std::vector<EngineMove> AnalyzeTopN(const std::vector<std::string>& movesSoFar,
                                        const std::string& playedMove,
                                        int depth,
                                        int topN) override;
    int LegalMoveCount(const std::vector<std::string>& movesSoFar) override;

  private:
    AnalyzeOptions options_;
};

GameRecord ParseGameFile(const std::string& path, std::vector<std::string>& warnings);
GameReport AnalyzeGame(const GameRecord& record, EngineAdapter& engine, const AnalyzeOptions& options);
std::string ReportToJson(const GameReport& report);

} // namespace CheatDetection
