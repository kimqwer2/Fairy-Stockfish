#ifndef EXPERIENCE_H_INCLUDED
#define EXPERIENCE_H_INCLUDED

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "incbin/incbin.h"

namespace Stockfish {

#pragma pack(push, 1)
struct ExperienceEntry {
  uint64_t key;
  uint16_t move;
};
#pragma pack(pop)

static_assert(sizeof(ExperienceEntry) == 10, "ExperienceEntry must be exactly 10 bytes");

#if !defined(_MSC_VER)
INCBIN(EmbeddedExperience, "experience.bin");
#else
const unsigned char        gEmbeddedExperienceData[1] = {0x0};
[[maybe_unused]]
const unsigned char *const gEmbeddedExperienceEnd = &gEmbeddedExperienceData[1];
const unsigned int         gEmbeddedExperienceSize = 0;
#endif

inline bool is_bad_experience(uint64_t key, uint16_t move) {
  const auto* begin = reinterpret_cast<const ExperienceEntry*>(gEmbeddedExperienceData);
  const auto* end = begin + (gEmbeddedExperienceSize / sizeof(ExperienceEntry));
  const ExperienceEntry needle{key, move};
  return std::binary_search(begin, end, needle,
                            [](const ExperienceEntry& a, const ExperienceEntry& b) {
                              return a.key < b.key || (a.key == b.key && a.move < b.move);
                            });
}

}  // namespace Stockfish

#endif  // EXPERIENCE_H_INCLUDED
