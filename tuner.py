import argparse
import random
import subprocess
from pathlib import Path
from typing import Dict

import optuna

# 장기 점수 판정용 기물 가치
PIECE_SCORES = {"r": 13.0, "c": 7.0, "n": 5.0, "m": 5.0, "b": 3.0, "e": 3.0, "a": 3.0, "p": 2.0, "s": 2.0}

# 고정 기준값: check.py / DB 호환을 위해 키 이름 유지
BASELINE_MATERIAL_PARAMS = {
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

# Search 모드에서 고정 적용할 "최종 최적화" Material 값
FIXED_OPTIMIZED_MATERIAL_PARAMS = {
    "Janggi_Rook_MG": 1295,
    "Janggi_Rook_EG": 1500,
    "Janggi_Cannon_MG": 795,
    "Janggi_Cannon_EG": 605,
    "Janggi_Horse_MG": 495,
    "Janggi_Horse_EG": 825,
    "Janggi_Elephant_MG": 365,
    "Janggi_Elephant_EG": 335,
    "Janggi_Guard_MG": 345,
    "Janggi_Guard_EG": 335,
    "Janggi_Soldier_MG": 195,
    "Janggi_Soldier_EG": 275,
}

# Material 모드에서 고정 적용할 Search 기본값
BASELINE_SEARCH_PARAMS = {
    "LMR_Base": 520,
    "LMR_Div": 100,
    "Futility_Margin": 200,
}


class UCIEngine:
    def __init__(self, engine_path, nnue_file, threads, variant="janggimodern"):
        self.engine_path = engine_path
        self.nnue_file = nnue_file
        self.threads = threads
        self.variant = variant

        self.process = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name UCI_Variant value {self.variant}")
        self._send("setoption name Use NNUE value true")
        self._send(f"setoption name EvalFile value {self.nnue_file}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, command):
        if self.process.stdin:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()

    def _read_until(self, token):
        lines = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Engine died unexpectedly")
            line = line.strip()
            lines.append(line)
            if line.startswith(token):
                return lines

    def set_options(self, options):
        """옵션 설정은 한 번만 수행"""
        for name, value in options.items():
            self._send(f"setoption name {name} value {str(value)}")
        # C++ deep-override 경로를 확실히 타도록 강제
        self.new_game()

    def new_game(self):
        """매 게임마다 옵션을 다시 보내지 않고 게임 초기화만 수행"""
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def bestmove(self, start_fen, moves, nodes):
        cmd = f"position fen {start_fen}"
        if moves:
            cmd += " moves " + " ".join(moves)

        self._send(cmd)
        self._send(f"go nodes {nodes}")

        lines = self._read_until("bestmove")
        return lines[-1].split()[1]

    def quit(self):
        try:
            self._send("quit")
            self.process.wait(timeout=2)
        except Exception:
            self.process.kill()


class JanggiMatchManager:
    def __init__(self, nodes, max_plies, fen_book):
        self.nodes = nodes
        self.max_plies = max_plies
        self.fen_book = fen_book

    def play_game(self, engine_a, engine_b, start_fen, a_is_cho):
        moves = []
        board = self.board_from_fen(start_fen)

        for ply in range(self.max_plies):
            a_to_move = (ply % 2 == 0) == a_is_cho
            engine = engine_a if a_to_move else engine_b
            mv = engine.bestmove(start_fen, moves, self.nodes)

            if mv in ("(none)", "0000"):
                return 0.0 if a_to_move else 1.0

            moves.append(mv)
            src, dst = mv[:2], mv[2:4]

            if src in board:
                board[dst] = board.pop(src)

        cho_s, han_s = self.calculate_score(board)
        return 1.0 if (cho_s > han_s if a_is_cho else han_s > cho_s) else 0.0

    def board_from_fen(self, fen):
        board = {}
        ranks = fen.split()[0].split("/")
        for ridx, rdata in enumerate(ranks):
            file_idx = 0
            rno = len(ranks) - ridx
            for ch in rdata:
                if ch.isdigit():
                    file_idx += int(ch)
                else:
                    board[f"{chr(ord('a') + file_idx)}{rno}"] = ch
                    file_idx += 1
        return board

    def calculate_score(self, board):
        cho = 0.0
        han = 1.5
        for p in board.values():
            v = PIECE_SCORES.get(p.lower(), 0.0)
            if p.isupper():
                cho += v
            else:
                han += v
        return cho, han


def suggest_material_params(trial):
    return {
        "Janggi_Rook_MG": trial.suggest_int("Janggi_Rook_MG", 1280, 1310),
        "Janggi_Rook_EG": trial.suggest_int("Janggi_Rook_EG", 1485, 1515),
        "Janggi_Cannon_MG": trial.suggest_int("Janggi_Cannon_MG", 785, 805),
        "Janggi_Cannon_EG": trial.suggest_int("Janggi_Cannon_EG", 595, 615),
        "Janggi_Horse_MG": trial.suggest_int("Janggi_Horse_MG", 483, 507),
        "Janggi_Horse_EG": trial.suggest_int("Janggi_Horse_EG", 813, 837),
        "Janggi_Elephant_MG": trial.suggest_int("Janggi_Elephant_MG", 353, 377),
        "Janggi_Elephant_EG": trial.suggest_int("Janggi_Elephant_EG", 323, 347),
        "Janggi_Guard_MG": trial.suggest_int("Janggi_Guard_MG", 335, 355),
        "Janggi_Guard_EG": trial.suggest_int("Janggi_Guard_EG", 325, 345),
        "Janggi_Soldier_MG": trial.suggest_int("Janggi_Soldier_MG", 185, 205),
        "Janggi_Soldier_EG": trial.suggest_int("Janggi_Soldier_EG", 265, 285),
    }


def suggest_search_params(trial):
    return {
        "LMR_Base": trial.suggest_int("LMR_Base", 400, 700),
        "LMR_Div": trial.suggest_int("LMR_Div", 80, 120),
        "Futility_Margin": trial.suggest_int("Futility_Margin", 100, 300),
    }


def get_combined_params(trial: optuna.Trial, mode: str) -> Dict[str, int]:
    """
    mode에 따라 '튜닝 대상' + '고정값'을 합쳐 엔진에 전달할 전체 파라미터를 생성.
    항상 Material + Search 전체 키를 반환한다.
    """
    if mode == "material":
        tuned = suggest_material_params(trial)
        combined = {
            **tuned,
            **BASELINE_SEARCH_PARAMS,
        }
    else:  # mode == "search"
        tuned = suggest_search_params(trial)
        combined = {
            **FIXED_OPTIMIZED_MATERIAL_PARAMS,
            **tuned,
        }

    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="stockfishtune.exe")
    parser.add_argument("--nnue-file", default="janggimodern-17.nnue")
    parser.add_argument("--book", required=True)
    parser.add_argument("--study", default="janggi_optuna")
    parser.add_argument("--storage", default="sqlite:///janggi_optuna.db")
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--games-per-trial", type=int, default=22)
    parser.add_argument("--nodes", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--mode", choices=["material", "search"], default="material")
    args = parser.parse_args()

    fens = [l.strip() for l in Path(args.book).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study,
        storage=args.storage,
        load_if_exists=True,
    )

    def objective(trial):
        params = get_combined_params(trial, args.mode)

        # 비교군 baseline은 항상 "final optimized material + baseline search"로 고정
        baseline = {
            **FIXED_OPTIMIZED_MATERIAL_PARAMS,
            **BASELINE_SEARCH_PARAMS,
        }

        # Persistent Engine 구조 유지: Trial 당 엔진 2개를 띄워 모든 게임 재사용
        ea = UCIEngine(args.engine, args.nnue_file, 1)
        eb = UCIEngine(args.engine, args.nnue_file, 1)

        try:
            manager = JanggiMatchManager(args.nodes, args.max_plies, fens)
            total_score = 0.0

            ea.set_options(params)
            eb.set_options(baseline)

            for _ in range(args.games_per_trial // 2):
                fen = random.choice(fens)

                ea.new_game()
                eb.new_game()
                total_score += manager.play_game(ea, eb, fen, True)

                ea.new_game()
                eb.new_game()
                total_score += (1.0 - manager.play_game(eb, ea, fen, True))

            return total_score / args.games_per_trial

        finally:
            ea.quit()
            eb.quit()

    # n_jobs 동시성 유지
    study.optimize(objective, n_trials=args.trials, n_jobs=args.concurrency)


if __name__ == "__main__":
    main()
