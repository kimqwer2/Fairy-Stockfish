#include "cheat_detection.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <limits>
#include <iostream>

#include "evaluate.h"
#include "misc.h"
#include "movegen.h"
#include "position.h"
#include "thread.h"
#include "uci.h"
#include "variant.h"

namespace Stockfish {

namespace {

constexpr double CRITICAL_GAP_CP = 120.0;

inline double clamp01(double x) {
  return std::max(0.0, std::min(1.0, x));
}

inline double value_to_cp(Value v) {
  return double(v) * 100.0 / PawnValueEg;
}

}  // namespace


std::string format_pv_comment(double choEls, double hanEls) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "(Cho:%.1f%% Han:%.1f%%)", choEls, hanEls);
  return std::string(buffer);
}

std::string format_debug_string(double choEls, double hanEls) {
  char buffer[80];
  std::snprintf(buffer, sizeof(buffer), "[FJACE] Cho: %.1f%% Han: %.1f%%", choEls, hanEls);
  return std::string(buffer);
}

bool is_supported_variant(const std::string& variantName) {
  return variantName == "janggi" || variantName == "janggimodern";
}

FjaceTracker g_tracker;

double g_choEls = 0.0;
double g_hanEls = 0.0;

void FjaceTracker::reset() {
  baseFen.clear();
  baseSfen = false;
  moveHistory.clear();
  sides[CHO] = SideAcc{};
  sides[HAN] = SideAcc{};
  lastCpl = 0.0;
}

void FjaceTracker::on_position_command(const Variant* variant,
                                       const std::string& variantName,
                                       const std::string& fen,
                                       bool sfen,
                                       const std::vector<std::string>& moves,
                                       bool enabled,
                                       bool chess960,
                                       Thread* th) {
  if (!enabled || !variant || !is_supported_variant(variantName))
    return;

  if (baseFen != fen || baseSfen != sfen || moves.size() < moveHistory.size()
      || !std::equal(moveHistory.begin(), moveHistory.end(), moves.begin())) {
    reset();
    baseFen = fen;
    baseSfen = sfen;
  }

  if (moves.size() == moveHistory.size())
    return;

  UpdateResult res = evaluate_last_move(variant, fen, sfen, moves, chess960, th);
  moveHistory = moves;

  if (res.updated) {
    lastCpl = res.lastCpl;
    emit_current_info();
  }
}

FjaceTracker::UpdateResult FjaceTracker::evaluate_last_move(const Variant* variant,
                                                            const std::string& fen,
                                                            bool sfen,
                                                            const std::vector<std::string>& moves,
                                                            bool chess960,
                                                            Thread* th) {
  UpdateResult result;

  std::deque<StateInfo> states(1);
  Position pos;
  pos.set(variant, fen, chess960, &states.back(), th, sfen);

  for (size_t i = 0; i + 1 < moves.size(); ++i) {
    std::string uciMove = moves[i];
    Move m = UCI::to_move(pos, uciMove);
    if (m == MOVE_NONE)
      return result;
    states.emplace_back();
    pos.do_move(m, states.back());
  }

  std::string playedStr = moves.back();
  Move played = UCI::to_move(pos, playedStr);
  if (played == MOVE_NONE)
    return result;

  const size_t legalMoves = MoveList<LEGAL>(pos).size();
  if (legalMoves <= 2)
    return result;

  struct ScoredMove { Move m; double score; };
  std::vector<ScoredMove> scored;
  scored.reserve(legalMoves);

  for (const auto& em : MoveList<LEGAL>(pos)) {
    Move m = em;
    states.emplace_back();
    pos.do_move(m, states.back());
    const double cp = -value_to_cp(Eval::evaluate(pos));
    pos.undo_move(m);
    states.pop_back();
    scored.push_back({m, cp});
  }

  std::sort(scored.begin(), scored.end(), [](const ScoredMove& a, const ScoredMove& b) {
    return a.score > b.score;
  });

  if (scored.empty())
    return result;

  const double best = scored[0].score;
  double second = best;
  if (scored.size() > 1)
    second = scored[1].score;

  double playedScore = best;
  size_t playedRank = scored.size();
  for (size_t i = 0; i < scored.size(); ++i) {
    if (scored[i].m == played) {
      playedScore = scored[i].score;
      playedRank = i;
      break;
    }
  }

  if (playedRank == scored.size())
    return result;

  const double cpl = std::max(0.0, best - playedScore);
  const bool critical = (best - second) >= CRITICAL_GAP_CP;

  // odd plies -> Cho, even plies -> Han
  const SideId side = (moves.size() % 2 == 1) ? CHO : HAN;
  SideAcc& acc = sides[side];

  acc.considered++;
  acc.top1 += (playedRank == 0 ? 1 : 0);
  acc.top3 += (playedRank < 3 ? 1 : 0);
  if (critical) {
    acc.criticalTotal++;
    acc.criticalMatch += (playedRank == 0 ? 1 : 0);
  }

  acc.cplSum += cpl;
  acc.cplSqSum += cpl * cpl;
  acc.playedSum += playedScore;
  acc.bestSum += best;
  acc.playedSqSum += playedScore * playedScore;
  acc.bestSqSum += best * best;
  acc.crossSum += playedScore * best;

  result.updated = true;
  result.lastCpl = cpl;
  return result;
}

void FjaceTracker::emit_current_info() const {
  g_choEls = score_for_side(sides[CHO]);
  g_hanEls = score_for_side(sides[HAN]);

  if (CurrentProtocol == XBOARD) {
      char message[96];
      std::snprintf(message, sizeof(message), "telluser [FJACE] Cho: %.1f%% Han: %.1f%%", g_choEls, g_hanEls);
      sync_cout << message << sync_endl;
  }
}

double FjaceTracker::score_for_side(const SideAcc& acc) {
  if (!acc.considered)
    return 0.0;

  const double top3Rate = double(acc.top3) / acc.considered;
  const double avgCpl = acc.cplSum / acc.considered;
  const double criticalAcc = acc.criticalTotal ? double(acc.criticalMatch) / acc.criticalTotal : 0.0;
  const double cplVar = variance(acc);
  const double corr = (correlation(acc) + 1.0) / 2.0;

  const double w1 = 30.0;
  const double w2 = 25.0;
  const double w3 = 30.0;
  const double w4 = 10.0;
  const double w5 = 5.0;

  const double score = w1 * top3Rate
                     + w2 * (1.0 / (1.0 + avgCpl))
                     + w3 * criticalAcc
                     + w4 * (1.0 / (1.0 + cplVar))
                     + w5 * clamp01(corr);

  return std::max(0.0, std::min(100.0, score));
}

double FjaceTracker::variance(const SideAcc& acc) {
  if (!acc.considered)
    return 0.0;
  const double n = double(acc.considered);
  const double mean = acc.cplSum / n;
  return std::max(0.0, (acc.cplSqSum / n) - mean * mean);
}

double FjaceTracker::correlation(const SideAcc& acc) {
  if (acc.considered < 2)
    return 0.0;

  const double n = double(acc.considered);
  const double num = n * acc.crossSum - acc.playedSum * acc.bestSum;
  const double left = n * acc.playedSqSum - acc.playedSum * acc.playedSum;
  const double right = n * acc.bestSqSum - acc.bestSum * acc.bestSum;
  const double den = std::sqrt(std::max(0.0, left * right));

  if (den <= std::numeric_limits<double>::epsilon())
    return 0.0;
  return std::max(-1.0, std::min(1.0, num / den));
}


void fjace_on_position_command(const Variant* variant,
                               const std::string& variantName,
                               const std::string& fen,
                               bool sfen,
                               const std::vector<std::string>& moves,
                               bool enabled,
                               bool chess960,
                               Thread* th) {
  g_tracker.on_position_command(variant, variantName, fen, sfen, moves, enabled, chess960, th);
}

void fjace_reset() {
  g_tracker.reset();
  g_choEls = 0.0;
  g_hanEls = 0.0;
}

std::string fjace_info_string(bool enabled, const std::string& variantName) {
  if (!enabled || !is_supported_variant(variantName))
      return "";
  return format_pv_comment(g_choEls, g_hanEls);
}

std::string fjace_debug_string(bool enabled, const std::string& variantName) {
  if (!enabled || !is_supported_variant(variantName))
      return "";
  return format_debug_string(g_choEls, g_hanEls);
}

}  // namespace Stockfish
