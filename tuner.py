#!/usr/bin/env python3
import argparse
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import optuna

START_FEN_JANGGI = "rnba1abnr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RNBA1ABNR w - - 0 1"

PIECE_SCORES = {
    "r": 13.0,
    "c": 7.0,
    "n": 5.0,
    "b": 3.0,
    "a": 3.0,
    "p": 2.0,
}



class UCIEngine:
    def __init__(self, engine_path: str, variant: str = "janggimodern"):
        self.process = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        self.variant = variant

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

    def bestmove(self, moves: List[str], depth: Optional[int], movetime_ms: Optional[int]) -> str:
        pos = "position startpos" if not moves else "position startpos moves " + " ".join(moves)
        self._send(pos)
        if depth is not None:
            self._send(f"go depth {depth}")
        else:
            self._send(f"go movetime {movetime_ms}")
        lines = self._read_until("bestmove")
        best = lines[-1].split()
        if len(best) < 2:
            raise RuntimeError(f"Malformed bestmove line: {lines[-1]}")
        return best[1]

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


class JanggiMatchManager:
    def __init__(self, engine_path: str, depth: Optional[int], movetime_ms: Optional[int], max_plies: int = 200):
        self.engine_path = engine_path
        self.depth = depth
        self.movetime_ms = movetime_ms
        self.max_plies = max_plies

    @staticmethod
    def _initial_board() -> Dict[str, str]:
        board_part = START_FEN_JANGGI.split()[0]
        board: Dict[str, str] = {}
        ranks = board_part.split("/")
        for rank_idx, rank_data in enumerate(ranks):
            file_idx = 0
            for ch in rank_data:
                if ch.isdigit():
                    file_idx += int(ch)
                else:
                    square = chr(ord("a") + file_idx) + str(10 - rank_idx)
                    board[square] = ch
                    file_idx += 1
        return board

    @staticmethod
    def _apply_uci_move(board: Dict[str, str], move: str) -> None:
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
    def _material_score(board: Dict[str, str]) -> Tuple[float, float]:
        cho = 0.0
        han = 1.5
        for p in board.values():
            v = PIECE_SCORES.get(p.lower(), 0.0)
            if p.isupper():
                cho += v
            else:
                han += v
        return cho, han

    def _adjudicate(self, board: Dict[str, str]) -> int:
        cho, han = self._material_score(board)
        if cho > han:
            return 1
        if cho < han:
            return -1
        return 0

    def play_game(self, options_for_a: Dict[str, int], a_is_cho: bool) -> GameResult:
        engine_a = UCIEngine(self.engine_path)
        engine_b = UCIEngine(self.engine_path)
        try:
            engine_a.initialize()
            engine_b.initialize()
            engine_a.set_options(options_for_a)

            engine_a.new_game()
            engine_b.new_game()

            board = self._initial_board()
            moves: List[str] = []

            for ply in range(self.max_plies):
                a_to_move = (ply % 2 == 0 and a_is_cho) or (ply % 2 == 1 and not a_is_cho)
                side_engine = engine_a if a_to_move else engine_b

                bm = side_engine.bestmove(moves, self.depth, self.movetime_ms)
                if bm == "(none)":
                    return GameResult(score_for_engine_a=0.0 if a_to_move else 1.0, reason="no-legal-move")

                moves.append(bm)
                self._apply_uci_move(board, bm)

            adjudication = self._adjudicate(board)
            if adjudication == 0:
                return GameResult(0.5, "move-limit-draw")

            cho_wins = adjudication > 0
            a_wins = cho_wins if a_is_cho else not cho_wins
            return GameResult(1.0 if a_wins else 0.0, "move-limit-material")
        finally:
            engine_a.quit()
            engine_b.quit()

    def play_match(self, options_for_a: Dict[str, int], games: int) -> float:
        total = 0.0
        for g in range(games):
            a_is_cho = (g % 2 == 0)
            result = self.play_game(options_for_a, a_is_cho=a_is_cho)
            total += result.score_for_engine_a
        return total / games


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
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--games", type=int, default=4, help="Games per trial")
    parser.add_argument("--depth", type=int, default=6, help="Fixed search depth")
    parser.add_argument("--movetime", type=int, default=None, help="Move time in ms (used when depth is omitted)")
    parser.add_argument("--max-plies", type=int, default=200, help="Move-limit adjudication threshold")
    args = parser.parse_args()

    depth = args.depth if args.depth is not None else None
    movetime = args.movetime if depth is None else None

    manager = JanggiMatchManager(
        engine_path=args.engine,
        depth=depth,
        movetime_ms=movetime,
        max_plies=args.max_plies,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        score = manager.play_match(params, games=args.games)
        trial.set_user_attr("params", params)
        trial.set_user_attr("score", score)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("Best score:", study.best_value)
    print("Best params:")
    for k in sorted(study.best_params):
        print(f"  {k} = {study.best_params[k]}")


if __name__ == "__main__":
    main()
