#!/usr/bin/env python3
import argparse
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import optuna

PIECE_SCORES = {
    "r": 13.0,
    "c": 7.0,
    "n": 5.0,
    "b": 3.0,
    "a": 3.0,
    "p": 2.0,
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
            self._setoption(name, str(value), wait_ready=False)
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def bestmove(self, start_fen: str, moves: List[str], nodes: int) -> str:
        cmd = f"position fen {start_fen}"
        if moves:
            cmd += " moves " + " ".join(moves)
        self._send(cmd)
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
    def __init__(self, engine_path: str, nnue_file: str, threads: int, nodes: int, max_plies: int, fen_book: List[str], rng: random.Random):
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
        fens = [fen for fen in (cls.normalize_fen(line) for line in path.read_text(encoding="utf-8").splitlines()) if fen]
        if not fens:
            raise ValueError(f"No valid FEN entries found in {book_path}")
        return fens

    @staticmethod
    def board_from_fen(fen: str) -> Dict[str, str]:
        board: Dict[str, str] = {}
        ranks = fen.split()[0].split("/")
        for ridx, rank_data in enumerate(ranks):
            file_idx = 0
            rank_no = len(ranks) - ridx
            for ch in rank_data:
                if ch.isdigit():
                    file_idx += int(ch)
                else:
                    sq = f"{chr(ord('a') + file_idx)}{rank_no}"
                    board[sq] = ch
                    file_idx += 1
        return board

    @staticmethod
    def score_material(fen: str) -> float:
        board = JanggiMatchManager.board_from_fen(fen)
        score = 0.0
        for p in board.values():
            w = PIECE_SCORES.get(p.lower(), 0.0)
            score += w if p.isupper() else -w
        return score

    def adjudicate(self, fen: str) -> Optional[GameResult]:
        m = self.score_material(fen)
        if m > 18:
            return GameResult(1.0, "material")
        if m < -18:
            return GameResult(0.0, "material")
        return None

    def play_game(self, params_a: Dict[str, int], params_b: Dict[str, int], start_fen: str) -> GameResult:
        a = UCIEngine(self.engine_path, self.nnue_file, self.threads, variant="janggimodern")
        b = UCIEngine(self.engine_path, self.nnue_file, self.threads, variant="janggimodern")
        try:
            a.initialize()
            b.initialize()
            a.set_options(params_a)
            b.set_options(params_b)

            moves: List[str] = []
            side_a_white = True
            for ply in range(self.max_plies):
                eng = a if (ply % 2 == 0) == side_a_white else b
                mv = eng.bestmove(start_fen, moves, self.nodes)
                if mv in ("(none)", "0000"):
                    side_to_move_a = (ply % 2 == 0) == side_a_white
                    return GameResult(0.0 if side_to_move_a else 1.0, "terminal")
                moves.append(mv)

            return GameResult(0.5, "max_plies")
        finally:
            a.quit()
            b.quit()

    def play_symmetric_pair(self, params_a: Dict[str, int], params_b: Dict[str, int]) -> float:
        fen = self.rng.choice(self.fen_book)
        g1 = self.play_game(params_a, params_b, fen)
        g2 = self.play_game(params_b, params_a, fen)
        # g2 score is from B perspective when swapped, convert to A perspective
        return (g1.score_for_a + (1.0 - g2.score_for_a)) / 2.0


def suggest_params(trial: optuna.Trial) -> Dict[str, int]:
    return {
        "Janggi_Rook_MG": trial.suggest_int("Janggi_Rook_MG", 200, 2000),
        "Janggi_Rook_EG": trial.suggest_int("Janggi_Rook_EG", 200, 2200),
        "Janggi_Cannon_MG": trial.suggest_int("Janggi_Cannon_MG", 100, 1500),
        "Janggi_Cannon_EG": trial.suggest_int("Janggi_Cannon_EG", 100, 1500),
        "Janggi_Horse_MG": trial.suggest_int("Janggi_Horse_MG", 100, 1500),
        "Janggi_Horse_EG": trial.suggest_int("Janggi_Horse_EG", 100, 1800),
        "Janggi_Elephant_MG": trial.suggest_int("Janggi_Elephant_MG", 50, 1200),
        "Janggi_Elephant_EG": trial.suggest_int("Janggi_Elephant_EG", 50, 1200),
        "Janggi_Guard_MG": trial.suggest_int("Janggi_Guard_MG", 50, 1000),
        "Janggi_Guard_EG": trial.suggest_int("Janggi_Guard_EG", 50, 1000),
        "Janggi_Soldier_MG": trial.suggest_int("Janggi_Soldier_MG", 20, 600),
        "Janggi_Soldier_EG": trial.suggest_int("Janggi_Soldier_EG", 20, 700),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optuna tuner for Fairy-Stockfish janggimodern material values")
    p.add_argument("--engine", default="./src/stockfish")
    p.add_argument("--nnue-file", default="<empty>")
    p.add_argument("--book", required=True)
    p.add_argument("--study", default="janggi_optuna")
    p.add_argument("--storage", default="sqlite:///janggi_optuna.db")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--games-per-trial", type=int, default=2)
    p.add_argument("--nodes", type=int, default=3000)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    return p


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    fen_book = JanggiMatchManager.load_book(args.book)
    manager = JanggiMatchManager(args.engine, args.nnue_file, args.threads, args.nodes, args.max_plies, fen_book, rng)

    baseline = {
        "Janggi_Rook_MG": 1276,
        "Janggi_Rook_EG": 1380,
        "Janggi_Cannon_MG": 800,
        "Janggi_Cannon_EG": 600,
        "Janggi_Horse_MG": 520,
        "Janggi_Horse_EG": 800,
        "Janggi_Elephant_MG": 340,
        "Janggi_Elephant_EG": 350,
        "Janggi_Guard_MG": 400,
        "Janggi_Guard_EG": 350,
        "Janggi_Soldier_MG": 200,
        "Janggi_Soldier_EG": 270,
    }

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        score = 0.0
        for _ in range(args.games_per_trial):
            score += manager.play_symmetric_pair(params, baseline)
        return score / args.games_per_trial

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials)

    print("best_value:", study.best_value)
    print("best_params:", study.best_params)


if __name__ == "__main__":
    main()
