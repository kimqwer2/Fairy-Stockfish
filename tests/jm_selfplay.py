#!/usr/bin/env python3
import argparse, subprocess, re, sys

def uci(cmds, proc):
    for c in cmds:
        proc.stdin.write(c+"\n")
    proc.stdin.flush()

def read_until(proc, pat):
    rg = re.compile(pat)
    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        if rg.search(line):
            return line.strip()

def new_engine(path, nnue, threads, h):
    p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    uci(["uci"], p); read_until(p, r"uciok")
    uci([f"setoption name Threads value {threads}", f"setoption name Hash value {h}", "setoption name UCI_Variant value janggi", "setoption name Use NNUE value true", f"setoption name EvalFile value {nnue}", "isready"], p)
    read_until(p, r"readyok")
    return p

def play_game(w, b, fen, movetime=60, maxply=240):
    moves=[]
    side=0
    for _ in range(maxply):
        eng = w if side==0 else b
        pos = f"position fen {fen}"
        if moves: pos += " moves " + " ".join(moves)
        uci([pos, f"go movetime {movetime}"], eng)
        bm = read_until(eng, r"^bestmove")
        if bm is None: return 0
        m = bm.split()[1]
        if m == "(none)":
            return -1 if side==0 else 1
        moves.append(m)
        side ^= 1
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--cand', required=True)
    ap.add_argument('--openings', required=True); ap.add_argument('--nnue', required=True)
    ap.add_argument('--rounds', type=int, default=2); ap.add_argument('--movetime', type=int, default=60)
    ap.add_argument('--threads', type=int, default=1); ap.add_argument('--hash', type=int, default=64)
    args = ap.parse_args()
    fens=[]
    for l in open(args.openings):
        l=l.strip()
        if not l or l.startswith('#'): continue
        fens.append(l.split(' bm ')[0].strip())
    base= new_engine(args.base,args.nnue,args.threads,args.hash)
    cand= new_engine(args.cand,args.nnue,args.threads,args.hash)
    w=d=l=0
    for _ in range(args.rounds):
        for fen in fens:
            r1 = play_game(cand, base, fen, args.movetime)
            if r1==1: w+=1
            elif r1==-1: l+=1
            else: d+=1
            r2 = play_game(base, cand, fen, args.movetime)
            if r2==-1: w+=1
            elif r2==1: l+=1
            else: d+=1
    n=w+d+l
    score=(w+0.5*d)/n if n else 0.5
    elo = -400 * __import__('math').log10((1-score)/score) if 0<score<1 else (999 if score==1 else -999)
    print(f"games={n} W={w} D={d} L={l} score={score:.4f} elo={elo:.1f}")
    for p in (base,cand):
        uci(["quit"], p); p.wait(timeout=2)

if __name__=='__main__': main()
