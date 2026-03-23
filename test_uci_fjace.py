#!/usr/bin/env python3
import subprocess
import sys


def run_engine(commands: str, timeout: int = 30) -> str:
    proc = subprocess.run(
        ["./src/stockfish"],
        input=commands,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return proc.stdout + proc.stderr


def main() -> int:
    probe = run_engine("uci\nquit\n")
    if "janggimodern" not in probe:
        print("SKIP: janggimodern is not available in this build/options list")
        return 0

    script = "\n".join([
        "uci",
        "setoption name UCI_Variant value janggimodern",
        "setoption name Enable_Cheat_Detector value true",
        "isready",
        "ucinewgame",
        "position startpos moves a1a2 a10a9",
        "go depth 1",
        "quit",
        "",
    ])

    output = run_engine(script)
    assert "uciok" in output, output
    assert "readyok" in output, output
    assert "info string [FJACE]" in output, output
    assert "Cho ELS:" in output and "Han ELS:" in output and "Last Move CPL:" in output, output
    print("FJACE UCI integration test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
