# Fairy-Stockfish Janggi Anti-Cheat Module (FJACE)

This adds an offline, move-list-only cheat likelihood analyzer for janggi records.

## Build

```bash
cd src
make -j2 ARCH=x86-64 build
make -j2 ARCH=x86-64 cheat-tools
```

## CLI

```bash
./tools/fstockfish-cheat --input tests/data/sample_human.pgn --depth 18 --threads 4 --nnue-path nn-xxxx.nnue --out report.json
```

Output JSON includes:
- per-move engine-rank/eval/cpl/difficulty/critical flags
- side-specific features for Cho and Han
- `els_cho`, `els_han`, and `els_total` in [0,100]
- textual `detection_reasoning` and warnings

## Default model

`ELS = clamp( w1*top3 + w2*(1/(1+avg_cpl)) + w3*critical + w4*(1/(1+var)) + w5*corr_scaled, 0, 100 )`

Defaults: `w1=30, w2=25, w3=30, w4=10, w5=5`.

## Notes

- Current engine adapter exposes a pluggable `EngineAdapter`; tests use a stub adapter.
- For operational use, calibrate thresholds against human corpora and synthetic injection sets.
- Expected runtime at depth 18 is hardware-dependent; target remains 5-15s / 100 moves.
