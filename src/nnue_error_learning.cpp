#include "nnue_error_learning.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "position.h"
#include "uci.h"

namespace Stockfish::NnueErrorLearning {

namespace {

constexpr uint32_t CorrectionMagic = 0x43474a46; // FJGC
constexpr uint32_t DatasetMagic    = 0x45474a46; // FJGE
constexpr uint16_t FormatVersion   = 1;
constexpr int MaterialSlots = COLOR_NB * PIECE_TYPE_NB;

struct CorrectionEntry {
    int32_t sum = 0;
    uint32_t count = 0;
};

struct Header {
    uint32_t magic;
    uint16_t version;
    uint16_t recordSize;
    uint64_t count;
};

struct CorrectionRecord {
    uint64_t key;
    int32_t sum;
    uint32_t count;
};

struct DatasetRecord {
    uint64_t key;
    uint64_t materialKey;
    int16_t staticEval;
    int16_t searchEval;
    int16_t diff;
    int16_t materialResult;
    uint16_t gamePly;
    uint8_t sideToMove;
    uint8_t result; // 0 unknown, 1 white win, 2 draw, 3 black win
    uint8_t material[MaterialSlots];
    char variant[24];
    char fen[192];
};

std::mutex mutex;
std::unordered_map<Key, CorrectionEntry> corrections;
std::string loadedCorrectionFile;
bool loaded = false;

bool enabled_for_janggi() {
    return Options.count("UCI_Variant")
        && std::string(Options["UCI_Variant"]) == "janggimodern";
}

int16_t clamp_i16(Value v) {
    return int16_t(std::clamp(int(v), -32767, 32767));
}

uint8_t encode_result(const std::string& result) {
    if (result == "1-0") return 1;
    if (result == "1/2-1/2" || result == "0.5-0.5") return 2;
    if (result == "0-1") return 3;
    return 0;
}

void fill_text(char* dst, size_t n, const std::string& s) {
    std::memset(dst, 0, n);
    std::memcpy(dst, s.data(), std::min(n - 1, s.size()));
}

void load_file(const std::string& file) {
    corrections.clear();
    loadedCorrectionFile = file;
    loaded = true;

    if (file.empty() || file == "<empty>")
        return;

    std::ifstream in(file, std::ios::binary);
    if (!in)
        return;

    Header h{};
    if (!in.read(reinterpret_cast<char*>(&h), sizeof(h))
        || h.magic != CorrectionMagic
        || h.version != FormatVersion
        || h.recordSize != sizeof(CorrectionRecord))
        return;

    for (uint64_t i = 0; i < h.count; ++i)
    {
        CorrectionRecord r{};
        if (!in.read(reinterpret_cast<char*>(&r), sizeof(r)))
        {
            corrections.clear();
            return;
        }
        if (r.count)
            corrections[Key(r.key)] = {r.sum, r.count};
    }
}

} // namespace

bool active_for(const Position& pos) {
    return enabled_for_janggi() && pos.variant();
}

void init() { on_options_changed(); }

void on_options_changed() {
    std::lock_guard<std::mutex> lock(mutex);
    std::string file = Options.count("Janggi Correction File") ? std::string(Options["Janggi Correction File"]) : "";
    if (!loaded || file != loadedCorrectionFile)
        load_file(file);
}

void save() {
    std::lock_guard<std::mutex> lock(mutex);
    std::string file = Options.count("Janggi Correction File") ? std::string(Options["Janggi Correction File"]) : loadedCorrectionFile;
    if (file.empty() || file == "<empty>" || corrections.empty())
        return;

    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out)
        return;

    Header h{CorrectionMagic, FormatVersion, uint16_t(sizeof(CorrectionRecord)), corrections.size()};
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    for (const auto& it : corrections)
    {
        CorrectionRecord r{uint64_t(it.first), it.second.sum, it.second.count};
        out.write(reinterpret_cast<const char*>(&r), sizeof(r));
    }
}

Value correction(const Position& pos) {
    if (!Options.count("Janggi Correction Enable") || !bool(Options["Janggi Correction Enable"]) || !active_for(pos))
        return VALUE_ZERO;
    on_options_changed();
    std::lock_guard<std::mutex> lock(mutex);
    auto it = corrections.find(pos.key());
    if (it == corrections.end() || !it->second.count)
        return VALUE_ZERO;
    int maxValue = Options.count("Janggi Correction Maximum Value") ? int(Options["Janggi Correction Maximum Value"]) : 0;
    return Value(std::clamp(it->second.sum / int(it->second.count), -maxValue, maxValue));
}

void collect(const Position& pos, Value staticEval, Value searchEval, const std::string& result) {
    if (!Options.count("NNUE Error Collection") || !bool(Options["NNUE Error Collection"]) || !active_for(pos))
        return;
    if (std::abs(int(searchEval)) >= VALUE_MATE_IN_MAX_PLY)
        return;

    on_options_changed();
    int diff = int(searchEval) - int(staticEval);
    int maxValue = Options.count("Janggi Correction Maximum Value") ? int(Options["Janggi Correction Maximum Value"]) : 1000;
    diff = std::clamp(diff, -maxValue, maxValue);

    DatasetRecord rec{};
    rec.key = uint64_t(pos.key());
    rec.materialKey = uint64_t(pos.material_key());
    rec.staticEval = clamp_i16(staticEval);
    rec.searchEval = clamp_i16(searchEval);
    rec.diff = int16_t(diff);
    rec.materialResult = clamp_i16(pos.material_counting_result());
    rec.gamePly = uint16_t(std::min(pos.game_ply(), 65535));
    rec.sideToMove = uint8_t(pos.side_to_move());
    rec.result = encode_result(result);
    for (Color c : { WHITE, BLACK })
        for (PieceType pt = NO_PIECE_TYPE; pt < PIECE_TYPE_NB; ++pt)
            rec.material[c * PIECE_TYPE_NB + pt] = uint8_t(std::min(pos.count(c, pt), 255));
    fill_text(rec.variant, sizeof(rec.variant), std::string(Options["UCI_Variant"]));
    fill_text(rec.fen, sizeof(rec.fen), pos.fen());

    std::string dataset = Options.count("NNUE Error Dataset File") ? std::string(Options["NNUE Error Dataset File"]) : "";
    if (!dataset.empty() && dataset != "<empty>")
    {
        bool writeHeader = !std::ifstream(dataset, std::ios::binary).good();
        std::ofstream out(dataset, std::ios::binary | std::ios::app);
        if (out)
        {
            if (writeHeader)
            {
                Header h{DatasetMagic, FormatVersion, uint16_t(sizeof(DatasetRecord)), 0};
                out.write(reinterpret_cast<const char*>(&h), sizeof(h));
            }
            out.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
        }
    }

    if (Options.count("Janggi Correction Enable") && bool(Options["Janggi Correction Enable"]))
    {
        double lr = Options.count("Janggi Correction Learning Rate") ? double(Options["Janggi Correction Learning Rate"]) : 0.0;
        int learned = int(std::round(diff * lr / 100.0));
        std::lock_guard<std::mutex> lock(mutex);
        CorrectionEntry& e = corrections[pos.key()];
        e.sum += learned;
        e.count += 1;
    }
}

} // namespace Stockfish::NnueErrorLearning
