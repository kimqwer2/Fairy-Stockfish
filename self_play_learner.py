#!/usr/bin/env python3
"""Multi-threaded self-play blunder logger for Fairy-Stockfish janggimodern.

Output format: little-endian <QH>
  - uint64: Zobrist key before the blunder move
  - uint16: raw Fairy-Stockfish move encoding
"""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MATE_CP = 30000
SCORE_RE = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)")
BESTMOVE_RE = re.compile(r"^bestmove\s+(\S+)")
KEY_RE = re.compile(r"\bKey:\s*([0-9A-Fa-f]+)")


# Keep exactly as requested (LARGEBOARDS janggimodern mapping)
def square_to_index(sq: str) -> int:
    """장기(janggimodern) 10x9 보드 좌표를 Fairy-Stockfish 내부 인덱스로 변환"""
    file_idx = ord(sq[0]) - ord("a")
    rank_idx = int(sq[1:]) - 1
    return rank_idx * 12 + file_idx


# Keep exactly as requested (LARGEBOARDS raw move encoding)
def move_to_raw16(uci_move: str) -> int:
    """UCI 이동 좌표를 16비트 Raw 정수로 변환"""
    m = re.match(r"^([a-i](?:10|[1-9]))([a-i](?:10|[1-9]))", uci_move)
    if not m:
        return 0
    from_sq, to_sq = m.groups()
    return ((square_to_index(from_sq) << 7) + square_to_index(to_sq)) & 0xFFFF


@dataclass
class SearchResult:
    bestmove: str
    score_cp: int


class EngineProtocolError(RuntimeError):
    pass


class RawUCIEngine:
    def __init__(self, engine_path: str, variant: str, extra_options: dict[str, str]) -> None:
        self.engine_path = engine_path
        self.variant = variant
        self.extra_options = extra_options
        self.proc: Optional[subprocess.Popen[str]] = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise EngineProtocolError("Failed to create engine pipes")

        self.send("uci")
        self.read_until("uciok")

        # Required immediately after startup for this workflow
        self.send(f"setoption name UCI_Variant value {self.variant}")
        for name, value in self.extra_options.items():
            self.send(f"setoption name {name} value {value}")

        self.send("isready")
        self.read_until("readyok")
        self.send("ucinewgame")
        self.send("isready")
        self.read_until("readyok")

    def send(self, cmd: str) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise EngineProtocolError("Engine not started")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix: str) -> list[str]:
        if self.proc is None or self.proc.stdout is None:
            raise EngineProtocolError("Engine not started")
        out: list[str] = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise EngineProtocolError("Engine terminated unexpectedly")
            line = line.rstrip("\n")
            out.append(line)
            if line.startswith(prefix):
                return out

    def set_position(self, moves: list[str]) -> None:
        cmd = "position startpos"
        if moves:
            cmd += " moves " + " ".join(moves)
        self.send(cmd)

    def get_key(self, moves: list[str]) -> int:
        self.set_position(moves)
        self.send("d")
        key: Optional[int] = None
        for line in self.read_until("Checkers:"):
            m = KEY_RE.search(line)
            if m:
                key = int(m.group(1), 16)
        if key is None:
            raise EngineProtocolError("Failed to parse 'Key:' from engine output")
        return key

    def analyse(self, moves: list[str], depth: Optional[int], nodes: Optional[int], wtime: int, btime: int, inc: int) -> SearchResult:
        self.set_position(moves)

        if depth is not None:
            go_cmd = f"go depth {depth}"
        elif nodes is not None:
            go_cmd = f"go nodes {nodes}"
        else:
            go_cmd = f"go wtime {wtime} btime {btime} winc {inc} binc {inc}"

        self.send(go_cmd)

        score_cp: Optional[int] = None
        bestmove: Optional[str] = None

        for line in self.read_until("bestmove"):
            s = SCORE_RE.search(line)
            if s:
                kind, value_text = s.groups()
                value = int(value_text)
                if kind == "cp":
                    score_cp = value
                else:
                    score_cp = (MATE_CP - min(abs(value), 1000)) * (1 if value >= 0 else -1)

            b = BESTMOVE_RE.match(line)
            if b:
                bestmove = b.group(1)

        if not bestmove or bestmove in {"(none)", "0000"}:
            raise EngineProtocolError("Engine returned no legal move")

        return SearchResult(bestmove=bestmove, score_cp=score_cp if score_cp is not None else 0)

    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None


class SelfPlayBlunderLogger:
    def __init__(self, args: argparse.Namespace) -> None:
        self.engine_path = args.engine
        self.variant = args.variant
        self.games = args.games
        self.threads = args.threads
        self.base_time = args.time
        self.inc = args.inc
        self.depth = args.depth
        self.nodes = args.nodes
        self.max_plies = args.max_plies
        self.drop_threshold = args.drop_threshold
        self.out_bin = Path(args.out_bin)

        self.engine_options = dict(args.engine_options or [])

        self._records: list[tuple[int, int]] = []
        self._records_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._games_started = 0

    def _reserve_game(self) -> Optional[int]:
        with self._counter_lock:
            if self._games_started >= self.games:
                return None
            self._games_started += 1
            return self._games_started

    def _append_record(self, key: int, move_raw: int) -> None:
        with self._records_lock:
            self._records.append((key, move_raw))

    def _run_one_game(self, engine: RawUCIEngine) -> None:
        moves: list[str] = []

        for _ in range(self.max_plies):
            key_before = engine.get_key(moves)
            before = engine.analyse(
                moves,
                depth=self.depth,
                nodes=self.nodes,
                wtime=self.base_time,
                btime=self.base_time,
                inc=self.inc,
            )

            raw_move = move_to_raw16(before.bestmove)
            if raw_move == 0:
                break

            moves.append(before.bestmove)

            # Evaluate the resulting position to measure drop for the side that moved.
            after = engine.analyse(
                moves,
                depth=self.depth,
                nodes=self.nodes,
                wtime=self.base_time,
                btime=self.base_time,
                inc=self.inc,
            )

            drop = before.score_cp - (-after.score_cp)
            if drop >= self.drop_threshold:
                self._append_record(key_before, raw_move)

    def _worker(self, worker_id: int) -> None:
        engine = RawUCIEngine(self.engine_path, self.variant, self.engine_options)
        try:
            engine.start()
            while True:
                game_no = self._reserve_game()
                if game_no is None:
                    return

                try:
                    self._run_one_game(engine)
                except Exception as exc:
                    # Gracefully recover from engine instability by restarting this worker engine.
                    print(f"[worker {worker_id}] engine/game failure on game {game_no}: {exc}")
                    engine.close()
                    engine = RawUCIEngine(self.engine_path, self.variant, self.engine_options)
                    engine.start()
        finally:
            engine.close()

    def run(self) -> None:
        workers = [threading.Thread(target=self._worker, args=(i,), daemon=True) for i in range(self.threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        unique_sorted = sorted(set(self._records), key=lambda x: (x[0], x[1]))
        with self.out_bin.open("wb") as f:
            for key, move_raw in unique_sorted:
                f.write(struct.pack("<QH", key, move_raw))

        print(f"Finished {self.games} games")
        print(f"Collected {len(self._records)} raw records, {len(unique_sorted)} unique")
        print(f"Saved {len(unique_sorted)} entries to {self.out_bin}")


def parse_engine_options(items: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in items:
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"Invalid --engine-option '{item}', expected Name:Value")
        name, value = item.split(":", 1)
        parsed.append((name.strip(), value.strip()))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-threaded self-play blunder logger for janggimodern")
    parser.add_argument("--engine", required=True, help="Path to Fairy-Stockfish executable")
    parser.add_argument("--variant", default="janggimodern")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--time", type=int, default=1000, help="Base time in ms (used when --depth/--nodes are not set)")
    parser.add_argument("--inc", type=int, default=10, help="Increment in ms (used when --depth/--nodes are not set)")
    parser.add_argument("--depth", type=int, default=None, help="Fixed search depth (overrides time control)")
    parser.add_argument("--nodes", type=int, default=None, help="Fixed node budget (used when --depth is not set)")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--drop-threshold", type=int, default=250)
    parser.add_argument("--out-bin", default="experience.bin")
    parser.add_argument(
        "--engine-option",
        action="append",
        default=[],
        help="Extra engine option in Name:Value format (can be repeated)",
    )

    args = parser.parse_args()
    args.engine_options = parse_engine_options(args.engine_option)

    logger = SelfPlayBlunderLogger(args)
    logger.run()


if __name__ == "__main__":
    main()
