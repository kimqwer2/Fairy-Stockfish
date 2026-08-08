#include "nnue_error_learning.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <mutex>
#include <string>
#include <vector>

#include "position.h"
#include "uci.h"

namespace Stockfish::NnueErrorLearning {

namespace {

constexpr uint32_t CorrectionMagic = 0x43474a46; // FJGC
constexpr uint32_t DatasetMagic    = 0x45474a46; // FJGE
constexpr uint16_t FormatVersion   = 1;
constexpr int MaterialSlots = COLOR_NB * PIECE_TYPE_NB;
constexpr int MaxCorrectedPieces = 32;
constexpr int CorrectionEvalWindow = 1200;
constexpr size_t DatasetFlushRecords = 64;
constexpr uint64_t CorrectionSaveSamples = 64;

struct CorrectionEntry {
    Key key = 0;
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
std::vector<CorrectionEntry> corrections;
std::vector<DatasetRecord> datasetBuffer;
std::string loadedCorrectionFile;
std::string datasetFile;
int maxCorrection = 300;
int learningRate = 0;
bool loaded = false;
bool variantActive = false;
bool correctionEnabled = false;
bool collectionEnabled = false;
bool dirty = false;
uint64_t loadedEntries = 0;
uint64_t datasetRecords = 0;
uint64_t collectedRecords = 0;
std::atomic<uint64_t> correctionLookups{0};
std::atomic<uint64_t> correctionHits{0};
std::atomic<uint64_t> correctionApplied{0};
uint64_t correctionLearned = 0;
uint64_t lastSavedLearned = 0;
std::string loadStatus = "not loaded";

size_t next_power_of_two(size_t n) {
    size_t p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

size_t index_for(Key key) {
    return (uint64_t(key) * 11400714819323198485ull) & (corrections.size() - 1);
}

CorrectionEntry* find_entry(Key key) {
    if (corrections.empty())
        return nullptr;

    for (size_t idx = index_for(key), probes = 0; probes < corrections.size(); ++probes, idx = (idx + 1) & (corrections.size() - 1))
    {
        CorrectionEntry& e = corrections[idx];
        if (!e.count)
            return nullptr;
        if (e.key == key)
            return &e;
    }

    return nullptr;
}

CorrectionEntry& find_or_insert(Key key) {
    if (corrections.empty())
        corrections.resize(1024);

    while (true)
    {
        for (size_t idx = index_for(key), probes = 0; probes < corrections.size(); ++probes, idx = (idx + 1) & (corrections.size() - 1))
        {
            CorrectionEntry& e = corrections[idx];
            if (!e.count || e.key == key)
            {
                if (!e.count)
                    e.key = key;
                return e;
            }
        }

        std::vector<CorrectionEntry> old;
        old.swap(corrections);
        corrections.assign(old.size() * 2, {});
        for (const CorrectionEntry& e : old)
            if (e.count)
                find_or_insert(e.key) = e;
    }
}

bool enabled_for_janggi() {
    return Options.count("UCI_Variant")
        && std::string(Options["UCI_Variant"]) == "janggimodern";
}

bool has_correction_potential(const Position& pos, Value uncorrectedEval) {
    return variantActive
        && correctionEnabled
        && !corrections.empty()
        && std::abs(int(uncorrectedEval)) <= CorrectionEvalWindow
        && pos.count<ALL_PIECES>() <= MaxCorrectedPieces
        && pos.material_counting() == JANGGI_MATERIAL;
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

    loadedEntries = 0;
    loadStatus = "disabled or empty file";

    if (file.empty() || file == "<empty>")
        return;

    std::ifstream in(file, std::ios::binary);
    if (!in)
    {
        loadStatus = "file not found";
        return;
    }

    Header h{};
    if (!in.read(reinterpret_cast<char*>(&h), sizeof(h))
        || h.magic != CorrectionMagic
        || h.version != FormatVersion
        || h.recordSize != sizeof(CorrectionRecord))
    {
        loadStatus = "invalid header";
        return;
    }

    corrections.assign(next_power_of_two(std::max<uint64_t>(1024, h.count * 2)), {});
    for (uint64_t i = 0; i < h.count; ++i)
    {
        CorrectionRecord r{};
        if (!in.read(reinterpret_cast<char*>(&r), sizeof(r)))
        {
            corrections.clear();
            loadedEntries = 0;
            loadStatus = "truncated file";
            return;
        }
        if (r.count)
        {
            CorrectionEntry& e = find_or_insert(Key(r.key));
            e.sum = r.sum;
            e.count = r.count;
            loadedEntries++;
        }
    }
    loadStatus = "loaded";
}

void flush_dataset() {
    if (datasetFile.empty() || datasetFile == "<empty>" || datasetBuffer.empty())
        return;

    bool writeHeader = !std::ifstream(datasetFile, std::ios::binary).good();
    std::ofstream out(datasetFile, std::ios::binary | std::ios::app);
    if (!out)
        return;

    if (writeHeader)
    {
        Header h{DatasetMagic, FormatVersion, uint16_t(sizeof(DatasetRecord)), 0};
        out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    }

    out.write(reinterpret_cast<const char*>(datasetBuffer.data()), std::streamsize(datasetBuffer.size() * sizeof(DatasetRecord)));
    if (out)
    {
        datasetRecords += datasetBuffer.size();
        datasetBuffer.clear();
    }
}

bool save_correction_file() {
    std::string file = Options.count("Janggi Correction File") ? std::string(Options["Janggi Correction File"]) : loadedCorrectionFile;
    if (file.empty() || file == "<empty>" || corrections.empty() || !dirty)
        return false;

    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out)
        return false;

    uint64_t used = 0;
    for (const CorrectionEntry& e : corrections)
        used += e.count != 0;

    Header h{CorrectionMagic, FormatVersion, uint16_t(sizeof(CorrectionRecord)), used};
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    for (const CorrectionEntry& e : corrections)
        if (e.count)
        {
            CorrectionRecord r{uint64_t(e.key), e.sum, e.count};
            out.write(reinterpret_cast<const char*>(&r), sizeof(r));
        }

    if (!out)
        return false;

    dirty = false;
    return true;
}

} // namespace

bool active_for(const Position& pos) {
    return variantActive && pos.variant();
}

void init() { on_options_changed(); }

void on_options_changed() {
    std::lock_guard<std::mutex> lock(mutex);

    variantActive = enabled_for_janggi();
    correctionEnabled = Options.count("Janggi Correction Enable") && bool(Options["Janggi Correction Enable"]);
    collectionEnabled = Options.count("NNUE Error Collection") && bool(Options["NNUE Error Collection"]);
    maxCorrection = Options.count("Janggi Correction Maximum Value") ? int(Options["Janggi Correction Maximum Value"]) : 300;
    learningRate = Options.count("Janggi Correction Learning Rate") ? int(Options["Janggi Correction Learning Rate"]) : 0;
    datasetFile = Options.count("NNUE Error Dataset File") ? std::string(Options["NNUE Error Dataset File"]) : "";

    std::string file = Options.count("Janggi Correction File") ? std::string(Options["Janggi Correction File"]) : "";
    if (loaded && file != loadedCorrectionFile)
    {
        flush_dataset();
        save_correction_file();
    }
    if (!loaded || file != loadedCorrectionFile)
        load_file(file);
}

void flush() {
    std::lock_guard<std::mutex> lock(mutex);
    flush_dataset();
}

void save() {
    std::lock_guard<std::mutex> lock(mutex);
    flush_dataset();
    save_correction_file();
}

Value correction(const Position& pos, Value uncorrectedEval) {
    if (!has_correction_potential(pos, uncorrectedEval))
        return VALUE_ZERO;

    correctionLookups++;
    const CorrectionEntry* e = find_entry(pos.key());
    if (!e)
        return VALUE_ZERO;

    correctionHits++;
    Value v = Value(std::clamp(e->sum / int(e->count), -maxCorrection, maxCorrection));
    if (v != VALUE_ZERO)
        correctionApplied++;
    return v;
}

void collect(const Position& pos, Value staticEval, Value searchEval, const std::string& result) {
    if (!collectionEnabled || !active_for(pos) || std::abs(int(searchEval)) >= VALUE_MATE_IN_MAX_PLY)
        return;

    int diff = int(searchEval) - int(staticEval);
    diff = std::clamp(diff, -maxCorrection, maxCorrection);

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

    std::lock_guard<std::mutex> lock(mutex);
    datasetBuffer.push_back(rec);
    collectedRecords++;
    if (datasetBuffer.size() >= DatasetFlushRecords)
        flush_dataset();

    if (correctionEnabled && learningRate)
    {
        int learned = int(std::round(diff * learningRate / 100.0));
        CorrectionEntry& e = find_or_insert(pos.key());
        e.sum = std::clamp<int64_t>(int64_t(e.sum) + learned, INT32_MIN, INT32_MAX);
        if (e.count != UINT32_MAX)
            e.count += 1;
        dirty = true;
        correctionLearned++;
        if (correctionLearned - lastSavedLearned >= CorrectionSaveSamples)
        {
            lastSavedLearned = correctionLearned;
            flush_dataset();
            save_correction_file();
        }
    }
}

std::string status() {
    std::lock_guard<std::mutex> lock(mutex);
    uint64_t active = 0;
    for (const CorrectionEntry& e : corrections)
        active += e.count != 0;

    std::ostringstream os;
    os << "Janggi Correction Enabled: " << (correctionEnabled ? "true" : "false") << "\n"
       << "Janggi Correction Variant Active: " << (variantActive ? "true" : "false") << "\n"
       << "Janggi Correction File: " << loadedCorrectionFile << "\n"
       << "Janggi Correction Load Status: " << loadStatus << "\n"
       << "Janggi Correction Loaded: " << loadedEntries << "\n"
       << "Janggi Correction Active Entries: " << active << "\n"
       << "Janggi Correction Lookups: " << correctionLookups.load() << "\n"
       << "Janggi Correction Hits: " << correctionHits.load() << "\n"
       << "Janggi Correction Applied: " << correctionApplied.load() << "\n"
       << "Janggi Correction Learned: " << correctionLearned << "\n"
       << "NNUE Error Collection Enabled: " << (collectionEnabled ? "true" : "false") << "\n"
       << "NNUE Error Dataset File: " << datasetFile << "\n"
       << "NNUE Error Dataset Buffered: " << datasetBuffer.size() << "\n"
       << "NNUE Error Dataset Flushed: " << datasetRecords << "\n"
       << "NNUE Error Dataset Collected: " << collectedRecords;
    return os.str();
}

} // namespace Stockfish::NnueErrorLearning
