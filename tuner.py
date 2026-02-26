#!/usr/bin/env python3
import argparse
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import optuna

PIECE_SCORES = {
    "r": 13.0,  # Rook
    "c": 7.0,   # Cannon
    "n": 5.0,   # Horse
    "b": 3.0,   # Elephant
    "a": 3.0,   # Guard (Wazir)
    "p": 2.0,   # Soldier
}


class UCIEngine:
    def __init__(self, engine_path: str, nnue_file: str, threads: int, variant: str = "janggimodern"):
        self.variant = variant
        self.nnue_file = nnue_file
        self.threads = threads
        self.process = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

    def _send(self, command: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Engine stdin is not available")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_until(self, token_prefix: str) -> List[str]:
        if self.process.stdout is None:
            raise RuntimeError("Engine stdout is not available")

        lines: List[str] = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError(f"Engine terminated while waiting for '{token_prefix}'")
            line = line.strip()
            lines.append(line)
            if line.startswith(token_prefix):
                return lines

    def _setoption(self, name: str, value: str, wait_ready: bool = True) -> None:
        self._send(f"setoption name {name} value {value}")
        if wait_ready:
            self._send("isready")
            self._read_until("readyok")

    def initialize(self) -> None:
        self._send("uci")
        self._read_until("uciok")

        self._setoption("Threads", str(self.threads))
        self._setoption("UCI_Variant", self.variant)
        self._setoption("Use NNUE", "true")
        self._setoption("EvalFile", self.nnue_file)

    def set_options(self, options: Dict[str, int]) -> None:
        for name, value in options.items():
            self._setoption(name, str(value), wait_ready=True)

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def bestmove(self, start_fen: str, moves: List[str], nodes: int) -> str:
        position_cmd = f"position fen {start_fen}"
        if moves:
            position_cmd += " moves " + " ".join(moves)
        self._send(position_cmd)
        self._send(f"go nodes {nodes}")

        lines = self._read_until("bestmove")
        parts = lines[-1].split()
        if len(parts) < 2:
            raise RuntimeError(f"Malformed bestmove line: {lines[-1]}")
        return parts[1]

    def quit(self) -> None:
        if self.process.poll() is None:
            try:
                self._send("quit")
            except Exception:
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()


@dataclass
class GameResult:
    score_for_a: float
    reason: str


class JanggiMatchManager:
    def __init__(
        self,
        engine_path: str,
        nnue_file: str,
        threads: int,
        nodes: int,
        max_plies: int,
        fen_book: List[str],
        rng: random.Random,
    ):
        self.engine_path = engine_path
        self.nnue_file = nnue_file
        self.threads = threads
        self.nodes = nodes
        self.max_plies = max_plies
        self.fen_book = fen_book
        self.rng = rng

    @staticmethod
    def normalize_fen(raw_line: str) -> Optional[str]:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            return None

        if ";" in line:
            line = line.split(";", 1)[0].strip()
            if not line:
                return None

        fields = line.split()
        if len(fields) >= 6:
            return " ".join(fields[:6])
        if len(fields) == 4:
            return " ".join(fields + ["0", "1"])
        return None

    @classmethod
    def load_book(cls, book_path: str) -> List[str]:
        path = Path(book_path)
        if not path.exists():
            raise FileNotFoundError(f"Book file not found: {book_path}")

        fens: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            fen = cls.normalize_fen(line)
            if fen is not None:
                fens.append(fen)

        if not fens:
            raise ValueError(f"No valid FEN entries found in {book_path}")
        return fens

    @staticmethod
    def board_from_fen(fen: str) -> Dict[str, str]:
        board_part = fen.split()[0]
        ranks = board_part.split("/")
        max_rank = len(ranks)
        board: Dict[str, str] = {}

        for rank_idx, rank_data in enumerate(ranks):
            file_idx = 0
            for ch in rank_data:
                if ch.isdigit():
                    file_idx += int(ch)
                else:
                    square = chr(ord("a") + file_idx) + str(max_rank - rank_idx)
                    board[square] = ch
                    file_idx += 1
        return board

    @staticmethod
    def apply_uci_move(board: Dict[str, str], move: str) -> None:
        if move in ("0000", "(none)"):
            return
        if len(move) < 4:
            return

        src = move[:2]
        dst = move[2:4]
        if src == dst:
            return

        piece = board.pop(src, None)
        if piece is not None:
            board[dst] = piece

    @staticmethod
    def score_material(board: Dict[str, str]) -> Tuple[float, float]:
        cho = 0.0
        han = 1.5  # Han komi

        for piece in board.values():
            v = PIECE_SCORES.get(piece.lower(), 0.0)
            if piece.isupper():
                cho += v
            else:
                han += v

        return cho, han

    def adjudicate(self, board: Dict[str, str], a_is_cho: bool) -> float:
        cho, han = self.score_material(board)
        if cho == han:
            return 0.5
        cho_wins = cho > han
        a_wins = cho_wins if a_is_cho else not cho_wins
        return 1.0 if a_wins else 0.0

    def play_game(self, options_for_a: Dict[str, int], a_is_cho: bool, start_fen: str) -> GameResult:
        engine_a = UCIEngine(self.engine_path, self.nnue_file, self.threads)
        engine_b = UCIEngine(self.engine_path, self.nnue_file, self.threads)

        try:
            engine_a.initialize()
            engine_b.initialize()
            engine_a.set_options(options_for_a)

            engine_a.new_game()
            engine_b.new_game()

            board = self.board_from_fen(start_fen)
            moves: List[str] = []

            for ply in range(self.max_plies):
                a_to_move = (ply % 2 == 0 and a_is_cho) or (ply % 2 == 1 and not a_is_cho)
                engine = engine_a if a_to_move else engine_b

                move = engine.bestmove(start_fen, moves, self.nodes)
                if move in ("0000", "(none)"):
                    return GameResult(0.0 if a_to_move else 1.0, "no-legal-move")

                moves.append(move)
                self.apply_uci_move(board, move)

            return GameResult(self.adjudicate(board, a_is_cho), "move-limit-material")
        finally:
            engine_a.quit()
            engine_b.quit()

    def play_symmetric_pair(self, options_for_a: Dict[str, int], start_fen: str) -> float:
        score_a_as_cho = self.play_game(options_for_a, a_is_cho=True, start_fen=start_fen).score_for_a
        score_a_as_han = self.play_game(options_for_a, a_is_cho=False, start_fen=start_fen).score_for_a
        return score_a_as_cho + score_a_as_han


def suggest_params(trial: optuna.Trial) -> Dict[str, int]:
    return {
        "Janggi_Rook_MG": trial.suggest_int("Janggi_Rook_MG", 1100, 1550),
        "Janggi_Rook_EG": trial.suggest_int("Janggi_Rook_EG", 1200, 1700),
        "Janggi_Cannon_MG": trial.suggest_int("Janggi_Cannon_MG", 600, 850),
        "Janggi_Cannon_EG": trial.suggest_int("Janggi_Cannon_EG", 550, 850),
        "Janggi_Horse_MG": trial.suggest_int("Janggi_Horse_MG", 400, 800),
        "Janggi_Horse_EG": trial.suggest_int("Janggi_Horse_EG", 450, 900),
        "Janggi_Elephant_MG": trial.suggest_int("Janggi_Elephant_MG", 250, 400),
        "Janggi_Elephant_EG": trial.suggest_int("Janggi_Elephant_EG", 250, 450),
        "Janggi_Guard_MG": trial.suggest_int("Janggi_Guard_MG", 250, 400),
        "Janggi_Guard_EG": trial.suggest_int("Janggi_Guard_EG", 250, 450),
        "Janggi_Soldier_MG": trial.suggest_int("Janggi_Soldier_MG", 150, 280),
        "Janggi_Soldier_EG": trial.suggest_int("Janggi_Soldier_EG", 180, 350),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fairy-Stockfish Janggi Optuna tuner")
    parser.add_argument("--engine", required=True, help="Path to Fairy-Stockfish binary")
    parser.add_argument("--book", default="startpos.epd", help="Book file with FEN/EPD lines")
    parser.add_argument("--nnue", default="janggimodern-17.nnue", help="NNUE filename or path")
    parser.add_argument("--threads", type=int, default=1, help="Engine threads")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials")
    parser.add_argument("--games", type=int, default=4, help="Total games per trial (uses symmetric pairs)")
    parser.add_argument("--nodes", type=int, default=5000, help="Nodes per move")
    parser.add_argument("--max-plies", type=int, default=200, help="Adjudicate after this many plies")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    fen_book = JanggiMatchManager.load_book(args.book)
    rng = random.Random(args.seed)

    manager = JanggiMatchManager(
        engine_path=args.engine,
        nnue_file=args.nnue,
        threads=args.threads,
        nodes=args.nodes,
        max_plies=args.max_plies,
        fen_book=fen_book,
        rng=rng,
    )

    pair_count = max(1, args.games // 2)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        total = 0.0

        for _ in range(pair_count):
            start_fen = manager.rng.choice(manager.fen_book)
            total += manager.play_symmetric_pair(params, start_fen)

        avg = total / float(pair_count * 2)
        trial.set_user_attr("score", avg)
        trial.set_user_attr("params", params)
        return avg

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("Best score:", study.best_value)
    print("Best params:")
    for key in sorted(study.best_params):
        print(f"  {key} = {study.best_params[key]}")


if __name__ == "__main__":
    main()
