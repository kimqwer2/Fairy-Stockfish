#ifndef CHEAT_DETECTION_H_INCLUDED
#define CHEAT_DETECTION_H_INCLUDED

#include <string>
#include <vector>

#include "types.h"

namespace Stockfish {

class Variant;
class Thread;

class FjaceTracker {
 public:
  void reset();

  void on_position_command(const Variant* variant,
                           const std::string& variantName,
                           const std::string& fen,
                           bool sfen,
                           const std::vector<std::string>& moves,
                           bool enabled,
                           bool chess960,
                           Thread* th);

  void emit_current_info() const;

 private:
  enum SideId { CHO = 0, HAN = 1 };

  struct SideAcc {
    size_t considered = 0;
    size_t top1 = 0;
    size_t top3 = 0;
    size_t criticalTotal = 0;
    size_t criticalMatch = 0;
    double cplSum = 0.0;
    double cplSqSum = 0.0;
    double playedSum = 0.0;
    double bestSum = 0.0;
    double playedSqSum = 0.0;
    double bestSqSum = 0.0;
    double crossSum = 0.0;
  };

  struct UpdateResult {
    bool updated = false;
    double lastCpl = 0.0;
  };

  UpdateResult evaluate_last_move(const Variant* variant,
                                  const std::string& fen,
                                  bool sfen,
                                  const std::vector<std::string>& moves,
                                  bool chess960,
                                  Thread* th);

  static double score_for_side(const SideAcc& acc);
  static double variance(const SideAcc& acc);
  static double correlation(const SideAcc& acc);

  std::string baseFen;
  bool baseSfen = false;
  std::vector<std::string> moveHistory;
  SideAcc sides[2];
  double lastCpl = 0.0;
};

void fjace_on_position_command(const Variant* variant,
                               const std::string& variantName,
                               const std::string& fen,
                               bool sfen,
                               const std::vector<std::string>& moves,
                               bool enabled,
                               bool chess960,
                               Thread* th);

void fjace_reset();

std::string fjace_info_string(bool enabled, const std::string& variantName);
std::string fjace_debug_string(bool enabled, const std::string& variantName);

}  // namespace Stockfish

#endif
