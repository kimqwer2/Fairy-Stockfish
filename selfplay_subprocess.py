#!/usr/bin/env python3
import argparse, subprocess, math, random, os

def read_epd(path):
    fens=[]
    with open(path) as f:
        for line in f:
            s=line.strip()
            if not s or s.startswith('#'): continue
            fen=s.split(';')[0].strip()
            fens.append(fen)
    return fens

class Engine:
    def __init__(self,path,variant,nnue,movetime,depth=None,extra_opts=None):
        self.p=subprocess.Popen([path],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
        self.movetime=movetime
        self.send('uci'); self.wait('uciok')
        self.setopt('Threads','1'); self.setopt('Hash','16'); self.setopt('Use NNUE','true');
        if nnue:
            if not os.path.exists(nnue):
                raise RuntimeError(f'NNUE file not found: {nnue}')
            self.setopt('EvalFile',nnue)
        self.setopt('UCI_Variant',variant)
        self.depth=depth
        if extra_opts:
            for k,v in extra_opts.items(): self.setopt(k,v)
        self.send('isready'); self.wait('readyok')
        if nnue:
            print(f'NNUE LOADED: YES ({path} -> {nnue})')
    def send(self,s): self.p.stdin.write(s+'\n'); self.p.stdin.flush()
    def wait(self,token):
        while True:
            l=self.p.stdout.readline()
            if not l: raise RuntimeError('engine died')
            if token in l: return l.strip()
    def bestmove(self,fen,moves):
        cmd='position fen '+fen
        if moves: cmd += ' moves ' + ' '.join(moves)
        self.send(cmd)
        if self.depth is not None:
            self.send(f'go depth {self.depth}')
        else:
            self.send(f'go movetime {self.movetime}')
        while True:
            l=self.p.stdout.readline()
            if not l: raise RuntimeError('engine died')
            if l.startswith('bestmove '):
                return l.split()[1]
    def bestmove_startpos(self,moves):
        cmd='position startpos'
        if moves: cmd += ' moves ' + ' '.join(moves)
        self.send(cmd)
        if self.depth is not None:
            self.send(f'go depth {self.depth}')
        else:
            self.send(f'go movetime {self.movetime}')
        while True:
            l=self.p.stdout.readline()
            if not l: raise RuntimeError('engine died')
            if l.startswith('bestmove '):
                return l.split()[1]
    def setopt(self,n,v): self.send(f'setoption name {n} value {v}')
    def close(self):
        try: self.send('quit')
        except: pass
        self.p.terminate()

def elo(score):
    score=max(1e-6,min(1-1e-6,score))
    return -400*math.log10(1/score-1)

def play_game(wa,ba,fen,maxplies):
    moves=[]
    for ply in range(maxplies):
        eng=wa if ply%2==0 else ba
        bm=eng.bestmove(fen,moves)
        if bm in ('0000','(none)','none'):
            # For janggi, a side with no legal move wins (pass-stalemate rule).
            return (1 if ply % 2 == 0 else 0), 'no_move'
        moves.append(bm)
    return 0.5, 'maxplies'

def generate_positions(args):
    rng = random.Random(args.seed)
    eng = Engine(args.engine, args.variant, args.nnue, args.movetime, args.depth)
    try:
        out = []
        attempts = 0
        while len(out) < args.count and attempts < args.count * args.max_retries:
            attempts += 1
            moves = []
            plies = rng.randint(args.min_plies, args.max_plies)
            failed = False
            for _ in range(plies):
                # Add noise by changing skill before each move
                eng.setopt('Skill Level', str(rng.randint(-20, 20)))
                bm = eng.bestmove_startpos(moves) if args.use_startpos else eng.bestmove(args.start_fen, moves)
                if bm in ('0000', '(none)', 'none'):
                    failed = True
                    break
                moves.append(bm)
            if failed or not moves:
                continue
            # Query resulting position via UCI "d" and parse "Fen:" line
            if args.use_startpos:
                eng.send('position startpos' + (' moves ' + ' '.join(moves) if moves else ''))
            else:
                eng.send('position fen ' + args.start_fen + (' moves ' + ' '.join(moves) if moves else ''))
            eng.send('d')
            fen = None
            while True:
                line = eng.p.stdout.readline()
                if not line:
                    break
                if line.startswith('Fen: '):
                    fen = line[len('Fen: '):].strip()
                if (line.strip() == '' or line.startswith('Checkers:')) and fen:
                    break
            if fen:
                out.append(fen)
        if len(out) < args.count:
            raise RuntimeError(f'Generated only {len(out)} positions out of requested {args.count}')
        with open(args.output, 'w') as f:
            for fen in out:
                f.write(fen + '\n')
        print(f'Generated {len(out)} positions -> {args.output}')
    finally:
        eng.close()

def run_match(args):
    fens=read_epd(args.epd)
    copt={}
    bopt={}
    if args.cand_skill is not None: copt['Skill Level']=str(args.cand_skill)
    if args.base_skill is not None: bopt['Skill Level']=str(args.base_skill)
    base_mt = args.base_movetime if args.base_movetime is not None else args.movetime
    cand_mt = args.cand_movetime if args.cand_movetime is not None else args.movetime
    base=Engine(args.base,args.variant,args.nnue,base_mt,args.base_depth,bopt)
    cand=Engine(args.cand,args.variant,args.nnue,cand_mt,args.cand_depth,copt)
    print(f'Candidate engine: {args.cand}')
    print(f'Baseline engine:  {args.base}')
    w=d=l=0
    ww=dw=lw=0
    wb=db=lb=0
    reason_counts = {'maxplies': 0, 'no_move': 0, 'abnormal': 0}
    try:
        for g in range(args.games):
            fen=fens[(g // 2) % len(fens)]
            try:
                if g%2==0:
                    cand_side='White'
                    r, reason = play_game(cand,base,fen,args.maxplies)
                else:
                    cand_side='Black'
                    r, reason = play_game(base,cand,fen,args.maxplies)
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            except Exception:
                reason_counts['abnormal'] += 1
                r = 0.5
                reason = 'abnormal'

            if g%2==0:
                if r==1: w+=1
                elif r==0: l+=1
                else: d+=1
                if r==1: ww+=1
                elif r==0: lw+=1
                else: dw+=1
            else:
                if r==0: w+=1
                elif r==1: l+=1
                else: d+=1
                if r==0: wb+=1
                elif r==1: lb+=1
                else: db+=1
            if r==0.5:
                outcome='Draw'
            elif (g%2==0 and r==1) or (g%2==1 and r==0):
                outcome='Candidate win'
            else:
                outcome='Candidate loss'
            print(f'game {g+1}: candidate {cand_side} -> {outcome} ({reason})')
            if (g+1)%10==0: print(f'games={g+1} W/D/L={w}/{d}/{l}')
    finally:
        base.close(); cand.close()
    total=w+d+l; score=(w+0.5*d)/total
    print(f'Total {total} W {w} D {d} L {l} Score {score:.3f} Elo {elo(score):.1f}')
    wt = ww + dw + lw
    bt = wb + db + lb
    if wt:
        print(f'Candidate as White: W {ww} D {dw} L {lw} Score {(ww + 0.5 * dw) / wt:.3f}')
    if bt:
        print(f'Candidate as Black: W {wb} D {db} L {lb} Score {(wb + 0.5 * db) / bt:.3f}')
    print('Termination breakdown:')
    print(f'  normal wins (terminal no-move): {reason_counts.get("no_move", 0)}')
    print(f'  forced draw (maxplies): {reason_counts.get("maxplies", 0)}')
    print(f'  abnormal termination: {reason_counts.get("abnormal", 0)}')

def main():
    ap=argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    m = sp.add_parser('match')
    m.add_argument('--base',required=True); m.add_argument('--cand',required=True)
    m.add_argument('--epd',default='janggimodern.epd'); m.add_argument('--variant',default='janggimodern')
    m.add_argument('--nnue',default='janggimodern-18.nnue'); m.add_argument('--games',type=int,default=80)
    m.add_argument('--movetime',type=int,default=30); m.add_argument('--maxplies',type=int,default=220)
    m.add_argument('--base-movetime',type=int); m.add_argument('--cand-movetime',type=int)
    m.add_argument('--cand-skill',type=int); m.add_argument('--base-skill',type=int)
    m.add_argument('--base-depth',type=int); m.add_argument('--cand-depth',type=int)
    g = sp.add_parser('gen')
    g.add_argument('--engine', required=True)
    g.add_argument('--output', default='janggimodern_midgame.epd')
    g.add_argument('--variant', default='janggi')
    g.add_argument('--nnue', default='janggimodern-18.nnue')
    g.add_argument('--count', type=int, default=24)
    g.add_argument('--min-plies', type=int, default=20)
    g.add_argument('--max-plies', type=int, default=40)
    g.add_argument('--movetime', type=int, default=50)
    g.add_argument('--depth', type=int)
    g.add_argument('--max-retries', type=int, default=8)
    g.add_argument('--seed', type=int, default=1)
    g.add_argument('--start-fen', default='rheagaehr/9/1c5c1/s1s1s1s1s/9/9/S1S1S1S1S/1C5C1/9/RHEAGAEHR w - - 0 1')
    g.add_argument('--use-startpos', action='store_true', default=True)
    args=ap.parse_args()
    if args.cmd == 'gen':
        generate_positions(args)
    else:
        run_match(args)

if __name__=='__main__': main()
