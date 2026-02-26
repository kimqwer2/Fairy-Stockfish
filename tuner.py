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
    def __init__(self, engine_path: str, nnue_file: str, variant: str = "janggimodern"):
        self.variant = variant
        self.nnue_file = nnue_file
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
                raise RuntimeError("Engine terminated unexpectedly while waiting for output")
            line = line.strip()
            lines.append(line)
            if line.startswith(token_prefix):
                return lines

    def initialize(self) -> None:
        self._send("uci")
        self._read_until("uciok")

        self._send(f"setoption name UCI_Variant value {self.variant}")
        self._send("setoption name Use NNUE value true")
        self._send(f"setoption name EvalFile value {self.nnue_file}")

        self._send("isready")
        self._read_until("readyok")

    def set_options(self, options: Dict[str, int]) -> None:
        for name, value in options.items():
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._read_until("readyok")

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
            self.process.wait(timeout=5)


@dataclass
class GameResult:
    score_for_engine_a: float
    reason: str
    start_fen: str


class JanggiMatchManager:
    def __init__(
        self,
        engine_path: str,
        nnue_file: str,
        nodes: int,
        max_plies: int,
        fen_book: List[str],
        rng: random.Random,
    ):
        self.engine_path = engine_path
        self.nnue_file = nnue_file
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
            raise ValueError(f"No valid FEN/EPD entries found in book: {book_path}")
        return fens

    @staticmethod
    def board_from_fen(fen: str) -> Dict[str, str]:
        board_part = fen.split()[0]
        ranks = board_part.split("/")
        board: Dict[str, str] = {}

        max_rank = len(ranks)
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

        frm = move[:2]
        to = move[2:4]
        if frm == to:
            return

        piece = board.pop(frm, None)
        if piece is None:
            return

        board[to] = piece

    @staticmethod
    def material_score(board: Dict[str, str]) -> Tuple[float, float]:
        cho = 0.0
        han = 1.5  # Han (second player / lowercase) komi bonus

        for piece in board.values():
            value = PIECE_SCORES.get(piece.lower(), 0.0)
            if piece.isupper():
                cho += value
            else:
                han += value

        return cho, han

    def adjudicate_by_material(self, board: Dict[str, str]) -> int:
        cho, han = self.material_score(board)
        if cho > han:
            return 1
        if cho < han:
            return -1
        return 0

    def play_game(self, options_for_a: Dict[str, int], a_is_cho: bool, start_fen: str) -> GameResult:
        engine_a = UCIEngine(self.engine_path, self.nnue_file)
        engine_b = UCIEngine(self.engine_path, self.nnue_file)

        try:
            engine_a.initialize()
            engine_b.initialize()

            if options_for_a:
                engine_a.set_options(options_for_a)

            engine_a.new_game()
            engine_b.new_game()

            board = self.board_from_fen(start_fen)
            moves: List[str] = []

            for ply in range(self.max_plies):
                a_to_move = (ply % 2 == 0 and a_is_cho) or (ply % 2 == 1 and not a_is_cho)
                current_engine = engine_a if a_to_move else engine_b

                bm = current_engine.bestmove(start_fen=start_fen, moves=moves, nodes=self.nodes)
                if bm == "(none)":
                    return GameResult(
                        score_for_engine_a=0.0 if a_to_move else 1.0,
                        reason="no-legal-move",
                        start_fen=start_fen,
                    )

                moves.append(bm)
                self.apply_uci_move(board, bm)

            adjudication = self.adjudicate_by_material(board)
            if adjudication == 0:
                return GameResult(0.5, "move-limit-draw", start_fen)

            cho_wins = adjudication > 0
            a_wins = cho_wins if a_is_cho else not cho_wins
            return GameResult(1.0 if a_wins else 0.0, "move-limit-material", start_fen)

        finally:
            engine_a.quit()
            engine_b.quit()

    def play_match(self, options_for_a: Dict[str, int], games: int) -> float:
        total_score = 0.0
        for game_index in range(games):
            start_fen = self.rng.choice(self.fen_book)
            a_is_cho = (game_index % 2 == 0)
            result = self.play_game(options_for_a=options_for_a, a_is_cho=a_is_cho, start_fen=start_fen)
            total_score += result.score_for_engine_a
        return total_score / float(games)


def suggest_params(trial: optuna.Trial) -> Dict[str, int]:
    return {
        "Janggi_Rook_MG": trial.suggest_int("Janggi_Rook_MG", 900, 2200),
        "Janggi_Rook_EG": trial.suggest_int("Janggi_Rook_EG", 900, 2200),
        "Janggi_Cannon_MG": trial.suggest_int("Janggi_Cannon_MG", 400, 1800),
        "Janggi_Cannon_EG": trial.suggest_int("Janggi_Cannon_EG", 300, 1800),
        "Janggi_Horse_MG": trial.suggest_int("Janggi_Horse_MG", 200, 1500),
        "Janggi_Horse_EG": trial.suggest_int("Janggi_Horse_EG", 200, 1500),
        "Janggi_Elephant_MG": trial.suggest_int("Janggi_Elephant_MG", 150, 1000),
        "Janggi_Elephant_EG": trial.suggest_int("Janggi_Elephant_EG", 150, 1000),
        "Janggi_Guard_MG": trial.suggest_int("Janggi_Guard_MG", 100, 1000),
        "Janggi_Guard_EG": trial.suggest_int("Janggi_Guard_EG", 100, 1000),
        "Janggi_Soldier_MG": trial.suggest_int("Janggi_Soldier_MG", 50, 700),
        "Janggi_Soldier_EG": trial.suggest_int("Janggi_Soldier_EG", 50, 900),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuner for Fairy-Stockfish janggimodern piece values")
    parser.add_argument("--engine", default="./src/stockfish", help="Path to Fairy-Stockfish binary")
    parser.add_argument("--book", default="startpos.epd", help="Path to opening book file (.epd or .fen lines)")
    parser.add_argument("--nnue", default="janggimodern-17.nnue", help="NNUE file name/path")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--games", type=int, default=4, help="Games per trial")
    parser.add_argument("--nodes", type=int, default=5000, help="Fixed nodes per move")
    parser.add_argument("--max-plies", type=int, default=200, help="Adjudication threshold in plies")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for book sampling")
    args = parser.parse_args()

    fen_book = JanggiMatchManager.load_book(args.book)
    rng = random.Random(args.seed)

    manager = JanggiMatchManager(
        engine_path=args.engine,
        nnue_file=args.nnue,
        nodes=args.nodes,
        max_plies=args.max_plies,
        fen_book=fen_book,
        rng=rng,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        score = manager.play_match(options_for_a=params, games=args.games)
        trial.set_user_attr("score", score)
        trial.set_user_attr("params", params)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("Best score:", study.best_value)
    print("Best params:")
    for key in sorted(study.best_params):
        print(f"  {key} = {study.best_params[key]}")


if __name__ == "__main__":
    main()
