#!/usr/bin/env python3
"""Simple Janggi material tuner driver for Fairy-Stockfish.

This script demonstrates the required command order before every game:
  setoption -> ucinewgame -> isready
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class JanggiValues:
    rook_mg: int = 600
    rook_eg: int = 650
    cannon_mg: int = 800
    cannon_eg: int = 600
    horse_mg: int = 520
    horse_eg: int = 800
    elephant_mg: int = 340
    elephant_eg: int = 350
    guard_mg: int = 400
    guard_eg: int = 350
    soldier_mg: int = 200
    soldier_eg: int = 270

    def as_uci_map(self) -> Dict[str, int]:
        return {
            "Janggi_Rook_MG": self.rook_mg,
            "Janggi_Rook_EG": self.rook_eg,
            "Janggi_Cannon_MG": self.cannon_mg,
            "Janggi_Cannon_EG": self.cannon_eg,
            "Janggi_Horse_MG": self.horse_mg,
            "Janggi_Horse_EG": self.horse_eg,
            "Janggi_Elephant_MG": self.elephant_mg,
            "Janggi_Elephant_EG": self.elephant_eg,
            "Janggi_Guard_MG": self.guard_mg,
            "Janggi_Guard_EG": self.guard_eg,
            "Janggi_Soldier_MG": self.soldier_mg,
            "Janggi_Soldier_EG": self.soldier_eg,
        }


class Engine:
    def __init__(self, path: str = "./src/stockfish") -> None:
        self.p = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def send(self, cmd: str) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def wait_for(self, token: str) -> None:
        assert self.p.stdout is not None
        for line in self.p.stdout:
            if token in line:
                return
        raise RuntimeError(f"Engine exited before seeing token: {token}")

    def handshake(self) -> None:
        self.send("uci")
        self.wait_for("uciok")

    def set_janggi_values(self, values: JanggiValues) -> None:
        self.send("setoption name UCI_Variant value janggi")
        for name, value in values.as_uci_map().items():
            self.send(f"setoption name {name} value {value}")

    def prepare_game(self, values: JanggiValues) -> None:
        # Required ordering for every game:
        # 1) setoption (values)
        # 2) ucinewgame (clear hash/tables)
        # 3) isready (barrier)
        self.set_janggi_values(values)
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok")

    def close(self) -> None:
        self.send("quit")
        self.p.wait(timeout=2)


def main() -> None:
    engine = Engine()
    engine.handshake()

    # Example: run two games with different tuned parameters.
    candidates = [
        JanggiValues(rook_mg=620, rook_eg=700),
        JanggiValues(rook_mg=650, rook_eg=740, cannon_eg=630),
    ]

    for i, params in enumerate(candidates, start=1):
        engine.prepare_game(params)
        engine.send("position startpos")
        engine.send("go depth 6")
        engine.wait_for("bestmove")
        print(f"Finished game {i} with params={params}")

    engine.close()


if __name__ == "__main__":
    main()
