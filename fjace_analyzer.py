#!/usr/bin/env python3
"""FJACE PGN analyzer for WinBoard-style Janggi PGN.

This script:
1. Parses a compact WinBoard move list such as:
      ih3 Hg7 2. Hg2 Hc7 3. Hc2 Che7
2. Heuristically translates those moves to UCI coordinates using a lightweight
   Janggi board tracker.
3. Feeds each growing prefix to Fairy-Stockfish / fstockfish through UCI.
4. Extracts FJACE percentages from engine output and prints a report.

The translator is intentionally lightweight. It supports the standard initial
Janggi setup and the common piece families needed for typical WinBoard logs:
rook/chariot, cannon, horse, elephant, advisor, king, and pawn.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

FILES = "abcdefghi"
BOARD_FILES = range(9)
BOARD_RANKS = range(10)
MOVE_RE = re.compile(r"^([RHEACKPrheackp]?)([a-i0-9]{0,2})([a-i])(10|[0-9])$")
ELS_RE = re.compile(
    r"(?:Cho[:_ ](?P<cho>\d+(?:\.\d+)?)%.*?Han[:_ ](?P<han>\d+(?:\.\d+)?)%)|"
    r"(?:\[Cho:(?P<cho2>\d+(?:\.\d+)?)%/Han:(?P<han2>\d+(?:\.\d+)?)%\])"
)


@dataclass(frozen=True)
class Piece:
    color: str  # 'w' or 'b'
    kind: str   # R C H E A K P


@dataclass
class HistoryEntry:
    ply: int
    san: str
    uci: str
    cho_els: Optional[float]
    han_els: Optional[float]


class JanggiBoard:
    def __init__(self) -> None:
        self.board: Dict[Tuple[int, int], Piece] = {}
        self._setup_initial_position()

    @staticmethod
    def parse_square(text: str) -> Tuple[int, int]:
        file_char = text[0].lower()
        rank_text = text[1:]
        if file_char not in FILES:
            raise ValueError(f"Invalid file in square: {text}")
        rank = int(rank_text)
        return FILES.index(file_char), rank

    @staticmethod
    def square_name(square: Tuple[int, int]) -> str:
        f, r = square
        return f"{FILES[f]}{r}"

    @staticmethod
    def in_bounds(square: Tuple[int, int]) -> bool:
        f, r = square
        return f in BOARD_FILES and r in BOARD_RANKS

    def _place(self, sq: str, color: str, kind: str) -> None:
        self.board[self.parse_square(sq)] = Piece(color, kind)

    def _setup_initial_position(self) -> None:
        # Bottom side (Cho / Red / white in engine orientation)
        for sq, kind in {
            "a0": "R", "b0": "H", "c0": "E", "d0": "A", "e1": "K",
            "f0": "A", "g0": "E", "h0": "H", "i0": "R",
            "b2": "C", "h2": "C",
            "a3": "P", "c3": "P", "e3": "P", "g3": "P", "i3": "P",
        }.items():
            self._place(sq, "w", kind)

        # Top side (Han / Green / black in engine orientation)
        for sq, kind in {
            "a9": "R", "b9": "H", "c9": "E", "d9": "A", "e8": "K",
            "f9": "A", "g9": "E", "h9": "H", "i9": "R",
            "b7": "C", "h7": "C",
            "a6": "P", "c6": "P", "e6": "P", "g6": "P", "i6": "P",
        }.items():
            self._place(sq, "b", kind)

    def piece_at(self, square: Tuple[int, int]) -> Optional[Piece]:
        return self.board.get(square)

    def side_to_move(self, ply_index: int) -> str:
        return "w" if ply_index % 2 == 0 else "b"

    def legal_moves_for_piece(self, origin: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
        piece = self.board.get(origin)
        if not piece:
            return []
        return list(self._legal_moves(origin, piece))

    def _legal_moves(self, origin: Tuple[int, int], piece: Piece) -> Iterable[Tuple[int, int]]:
        if piece.kind == "R":
            yield from self._sliding_moves(origin, piece, orthogonal=True, diagonal_palace=True)
        elif piece.kind == "C":
            yield from self._cannon_moves(origin, piece)
        elif piece.kind == "H":
            yield from self._horse_moves(origin, piece)
        elif piece.kind == "E":
            yield from self._elephant_moves(origin, piece)
        elif piece.kind == "A":
            yield from self._advisor_king_moves(origin, piece, advisor=True)
        elif piece.kind == "K":
            yield from self._advisor_king_moves(origin, piece, advisor=False)
        elif piece.kind == "P":
            yield from self._pawn_moves(origin, piece)

    def _sliding_moves(self, origin: Tuple[int, int], piece: Piece, orthogonal: bool, diagonal_palace: bool) -> Iterable[Tuple[int, int]]:
        directions = []
        if orthogonal:
            directions.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])
        for df, dr in directions:
            f, r = origin
            while True:
                f += df
                r += dr
                sq = (f, r)
                if not self.in_bounds(sq):
                    break
                blocker = self.piece_at(sq)
                if blocker is None:
                    yield sq
                    continue
                if blocker.color != piece.color:
                    yield sq
                break

        if diagonal_palace:
            yield from self._palace_diagonal_steps(origin, piece)

    def _cannon_moves(self, origin: Tuple[int, int], piece: Piece) -> Iterable[Tuple[int, int]]:
        for df, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            f, r = origin
            jumped = False
            while True:
                f += df
                r += dr
                sq = (f, r)
                if not self.in_bounds(sq):
                    break
                blocker = self.piece_at(sq)
                if not jumped:
                    if blocker is None:
                        continue
                    if blocker.kind == "C":
                        break
                    jumped = True
                    continue
                if blocker is None:
                    yield sq
                    continue
                if blocker.color != piece.color and blocker.kind != "C":
                    yield sq
                break
        # Simplified: omit diagonal palace cannon jumps.

    def _horse_moves(self, origin: Tuple[int, int], piece: Piece) -> Iterable[Tuple[int, int]]:
        patterns = [
            ((0, 1), (-1, 2)), ((0, 1), (1, 2)),
            ((0, -1), (-1, -2)), ((0, -1), (1, -2)),
            ((1, 0), (2, -1)), ((1, 0), (2, 1)),
            ((-1, 0), (-2, -1)), ((-1, 0), (-2, 1)),
        ]
        for leg, dest_delta in patterns:
            leg_sq = (origin[0] + leg[0], origin[1] + leg[1])
            dest = (origin[0] + dest_delta[0], origin[1] + dest_delta[1])
            if not self.in_bounds(dest) or self.piece_at(leg_sq) is not None:
                continue
            blocker = self.piece_at(dest)
            if blocker is None or blocker.color != piece.color:
                yield dest

    def _elephant_moves(self, origin: Tuple[int, int], piece: Piece) -> Iterable[Tuple[int, int]]:
        patterns = [
            ((0, 1), (-1, 2), (-2, 3)), ((0, 1), (1, 2), (2, 3)),
            ((0, -1), (-1, -2), (-2, -3)), ((0, -1), (1, -2), (2, -3)),
            ((1, 0), (2, -1), (3, -2)), ((1, 0), (2, 1), (3, 2)),
            ((-1, 0), (-2, -1), (-3, -2)), ((-1, 0), (-2, 1), (-3, 2)),
        ]
        for leg1, leg2_delta, dest_delta in patterns:
            leg1_sq = (origin[0] + leg1[0], origin[1] + leg1[1])
            leg2_sq = (origin[0] + leg2_delta[0], origin[1] + leg2_delta[1])
            dest = (origin[0] + dest_delta[0], origin[1] + dest_delta[1])
            if not self.in_bounds(dest):
                continue
            if self.piece_at(leg1_sq) is not None or self.piece_at(leg2_sq) is not None:
                continue
            blocker = self.piece_at(dest)
            if blocker is None or blocker.color != piece.color:
                yield dest

    def _advisor_king_moves(self, origin: Tuple[int, int], piece: Piece, advisor: bool) -> Iterable[Tuple[int, int]]:
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if advisor:
            deltas = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        else:
            deltas += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for df, dr in deltas:
            dest = (origin[0] + df, origin[1] + dr)
            if not self._in_palace(dest, piece.color):
                continue
            blocker = self.piece_at(dest)
            if blocker is None or blocker.color != piece.color:
                yield dest

    def _pawn_moves(self, origin: Tuple[int, int], piece: Piece) -> Iterable[Tuple[int, int]]:
        direction = 1 if piece.color == "w" else -1
        for dest in [(origin[0], origin[1] + direction), (origin[0] - 1, origin[1]), (origin[0] + 1, origin[1])]:
            if not self.in_bounds(dest):
                continue
            blocker = self.piece_at(dest)
            if blocker is None or blocker.color != piece.color:
                yield dest
        # Palace diagonals for pawns are omitted in this lightweight version.

    def _palace_diagonal_steps(self, origin: Tuple[int, int], piece: Piece) -> Iterable[Tuple[int, int]]:
        for df, dr in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            dest = (origin[0] + df, origin[1] + dr)
            if not self._in_palace(origin, piece.color) or not self._in_palace(dest, piece.color):
                continue
            blocker = self.piece_at(dest)
            if blocker is None or blocker.color != piece.color:
                yield dest

    def _in_palace(self, sq: Tuple[int, int], color: str) -> bool:
        f, r = sq
        if color == "w":
            return 3 <= f <= 5 and 0 <= r <= 2
        return 3 <= f <= 5 and 7 <= r <= 9

    def move_to_uci(self, san: str, ply_index: int) -> str:
        token = san.strip()
        if not token:
            raise ValueError("Empty move token")

        match = MOVE_RE.match(token)
        if not match:
            raise ValueError(f"Unsupported move token: {token}")

        piece_prefix, disambiguator, dest_file, dest_rank = match.groups()
        piece_kind = piece_prefix.upper() if piece_prefix else "P"
        color = self.side_to_move(ply_index)
        dest = self.parse_square(dest_file + dest_rank)

        candidates: List[Tuple[Tuple[int, int], str]] = []
        for square, piece in self.board.items():
            if piece.color != color or piece.kind != piece_kind:
                continue
            if disambiguator:
                if len(disambiguator) == 1:
                    if disambiguator in FILES and FILES[square[0]] != disambiguator.lower():
                        continue
                    if disambiguator.isdigit() and square[1] != int(disambiguator):
                        continue
                elif len(disambiguator) == 2:
                    if square != self.parse_square(disambiguator.lower()):
                        continue
            if dest in self.legal_moves_for_piece(square):
                candidates.append((square, self.square_name(square) + self.square_name(dest)))

        if not candidates:
            raise ValueError(f"Could not resolve move '{token}' for side {color}")
        if len(candidates) > 1:
            # Prefer the leftmost / lowest-file candidate when ambiguous.
            candidates.sort(key=lambda item: (item[0][0], item[0][1]))

        origin, uci = candidates[0]
        self.push(origin, dest)
        return uci

    def push(self, origin: Tuple[int, int], dest: Tuple[int, int]) -> None:
        piece = self.board.pop(origin)
        self.board.pop(dest, None)
        self.board[dest] = piece


def tokenize_pgn_moves(text: str) -> List[str]:
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    tokens = []
    for raw in text.replace("\n", " ").split():
        if raw.endswith(".") or raw in {"1-0", "0-1", "1/2-1/2", "*"}:
            continue
        if raw[:-1].isdigit() and raw.endswith("."):
            continue
        if raw.isdigit():
            continue
        tokens.append(raw)
    return tokens


def extract_els(line: str) -> Optional[Tuple[float, float]]:
    m = ELS_RE.search(line)
    if not m:
        return None
    cho = m.group("cho") or m.group("cho2")
    han = m.group("han") or m.group("han2")
    if cho is None or han is None:
        return None
    return float(cho), float(han)


class EngineSession:
    def __init__(self, engine_path: Path, depth: int) -> None:
        self.engine_path = engine_path
        self.depth = depth
        self.proc = subprocess.Popen(
            [str(engine_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def readline(self) -> str:
        assert self.proc.stdout is not None
        return self.proc.stdout.readline()

    def initialize(self) -> None:
        self.send("uci")
        self._wait_for("uciok")
        self.send("setoption name UCI_Variant value janggimodern")
        self.send("setoption name Enable_Cheat_Detector value true")
        self.send("isready")
        self._wait_for("readyok")

    def _wait_for(self, needle: str) -> None:
        while True:
            line = self.readline()
            if not line:
                raise RuntimeError(f"Engine closed before '{needle}'")
            if needle in line:
                return

    def analyze_prefix(self, moves: Sequence[str]) -> Tuple[Optional[float], Optional[float]]:
        joined = " ".join(moves)
        self.send(f"position startpos moves {joined}" if joined else "position startpos")
        self.send(f"go depth {self.depth}")

        cho: Optional[float] = None
        han: Optional[float] = None
        while True:
            line = self.readline()
            if not line:
                raise RuntimeError("Engine closed during analysis")
            parsed = extract_els(line)
            if parsed is not None:
                cho, han = parsed
            if line.startswith("bestmove"):
                return cho, han

    def close(self) -> None:
        try:
            self.send("quit")
        except Exception:
            pass
        self.proc.terminate()
        self.proc.wait(timeout=5)


def analyze_game(engine_path: Path, pgn_text: str, depth: int) -> List[HistoryEntry]:
    board = JanggiBoard()
    tokens = tokenize_pgn_moves(pgn_text)
    uci_moves: List[str] = []
    history: List[HistoryEntry] = []
    engine = EngineSession(engine_path, depth)
    engine.initialize()
    try:
        for ply, san in enumerate(tokens):
            uci = board.move_to_uci(san, ply)
            uci_moves.append(uci)
            cho, han = engine.analyze_prefix(uci_moves)
            history.append(HistoryEntry(ply=ply, san=san, uci=uci, cho_els=cho, han_els=han))
    finally:
        engine.close()
    return history


def print_report(history: Sequence[HistoryEntry], depth: int) -> None:
    final_cho = next((e.cho_els for e in reversed(history) if e.cho_els is not None), 0.0)
    final_han = next((e.han_els for e in reversed(history) if e.han_els is not None), 0.0)

    print("=" * 43)
    print("           FJACE ANALYSIS REPORT           ")
    print("=" * 43)
    print(f"Total Moves : {len(history)}")
    print(f"Engine Depth: {depth}")
    print("-" * 43)
    print("[MOVE HISTORY]")
    for entry in history:
        move_no = entry.ply // 2 + 1
        side_prefix = f"{move_no}." if entry.ply % 2 == 0 else f"{move_no}..."
        if entry.ply % 2 == 0:
            suffix = f"Cho ELS: {entry.cho_els if entry.cho_els is not None else 0.0:.1f}%"
        else:
            suffix = f"Han ELS: {entry.han_els if entry.han_els is not None else 0.0:.1f}%"
        print(f"{side_prefix:>4} {entry.san} ({entry.uci}) -> {suffix}")
    print("-" * 43)
    print("[FINAL CHEAT LIKELIHOOD]")
    print(f"CHO (Red)   : {final_cho:.1f}%")
    print(f"HAN (Green) : {final_han:.1f}%")
    print("=" * 43)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch FJACE analyzer for WinBoard-style Janggi PGN")
    parser.add_argument("input", help="Path to a PGN/move-list text file, or '-' to read stdin")
    parser.add_argument("--engine", default="./src/stockfish", help="Path to Fairy-Stockfish/fstockfish executable")
    parser.add_argument("--depth", type=int, default=15, help="Search depth for each batch step")
    args = parser.parse_args(argv)

    input_text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    engine_path = Path(args.engine)
    if not engine_path.exists():
        raise SystemExit(f"Engine not found: {engine_path}")

    history = analyze_game(engine_path, input_text, args.depth)
    print_report(history, args.depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
